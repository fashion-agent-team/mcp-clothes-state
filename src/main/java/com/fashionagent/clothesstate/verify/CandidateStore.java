package com.fashionagent.clothesstate.verify;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Clock;
import java.time.LocalDate;
import java.time.ZoneId;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.UUID;
import javax.imageio.ImageIO;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.context.event.ApplicationReadyEvent;
import org.springframework.context.event.EventListener;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;
import tools.jackson.databind.json.JsonMapper;

/**
 * GPT가 얼룩·찢어짐이라고 한 조각만 다시 학습할 후보로 저장합니다(정지원 2026-11-09 "얼룩 사진만 저장").
 * 무늬 모델이 몰라서 GPT에게 물은 내려놓은 옷 박스도 GPT 답과 함께 저장합니다(정지원 2026-11-11, 무늬 재학습용, pattern-아이디.jpg).
 * 폴더/날짜/아이디.jpg 와 같은 이름의 .json(종류, 우리 모델 확신도, GPT 확신도, 모델). 사용자 정보는 넣지 않습니다.
 * 바로 학습하지 않고, 사람이 확인한 뒤 학습에 씁니다. retentionDays(기본 30일)가 지난 날짜 폴더는 지웁니다(docs/gpt-check.md).
 */
@Component
public class CandidateStore {

    private static final Logger log = LoggerFactory.getLogger(CandidateStore.class);
    private static final ZoneId KST = ZoneId.of("Asia/Seoul");

    private final GptCheckProperties props;
    private final Clock clock;
    private final JsonMapper json = JsonMapper.builder().build();

    @Autowired
    public CandidateStore(GptCheckProperties props) {
        this(props, Clock.system(KST));
    }

    CandidateStore(GptCheckProperties props, Clock clock) {
        this.props = props;
        this.clock = clock.withZone(KST);
    }

    public boolean enabled() {
        return props.candidateDir() != null && !props.candidateDir().isBlank();
    }

    /** 저장한 파일 경로, 저장 안 했으면 null. 실패해도 검사는 계속합니다. */
    public Path save(BufferedImage crop, String modelKind, double modelConfidence, CropVerifier.Verdict v) {
        Map<String, Object> meta = new LinkedHashMap<>();
        meta.put("label", v.label());
        meta.put("modelKind", modelKind);
        meta.put("modelConfidence", round(modelConfidence));
        meta.put("gptConfidence", round(v.confidence()));
        meta.put("gptModel", v.model());
        meta.put("reviewed", false);
        return write("", crop, meta, "학습 후보");
    }

    /**
     * 무늬 재학습 후보: 옷 박스 그림(보낸 것과 같은 크기)과 .json(task=pattern, GPT 답·확신도, 우리 모델 1등·확률).
     * GPT 답도 틀릴 수 있어 reviewed=false로 두고 사람이 확인한 뒤 학습에 씁니다.
     */
    public Path savePattern(BufferedImage garment, String gptPattern, double gptConfidence, String gptModel,
            String modelTop, double modelConfidence) {
        Map<String, Object> meta = new LinkedHashMap<>();
        meta.put("task", "pattern");
        meta.put("label", gptPattern);
        meta.put("gptConfidence", round(gptConfidence));
        meta.put("gptModel", gptModel);
        meta.put("modelTop", modelTop);
        meta.put("modelConfidence", round(modelConfidence));
        meta.put("reviewed", false);
        return write("pattern-", garment, meta, "무늬 학습 후보");
    }

    /**
     * 옷 실수 재학습 후보(정지원 2026-11-16 "지피티로 단추 엇갈린거 학습하면서"): GPT에게 보낸 입은 옷 조각(얼굴 쪽은 이미 회색)과
     * .json(task=wear, 실수 종류, GPT가 실수라고 했는지·확신도, 우리 모델 확률). wear-종류-아이디.jpg. reviewed=false.
     */
    public Path saveWear(BufferedImage piece, String kind, boolean gptSaysMistake, double gptConfidence, String gptModel,
            double modelProbability, String modelVersion) {
        Map<String, Object> meta = new LinkedHashMap<>();
        meta.put("task", "wear");
        meta.put("kind", kind);
        meta.put("gptMistake", gptSaysMistake);
        meta.put("gptConfidence", round(gptConfidence));
        meta.put("gptModel", gptModel);
        meta.put("modelProbability", round(modelProbability));
        meta.put("modelVersion", modelVersion);
        meta.put("reviewed", false);
        String safeKind = kind == null ? "unknown" : kind.replaceAll("[^A-Za-z0-9_-]", "_");
        return write("wear-" + safeKind + "-", piece, meta, "옷 실수 학습 후보");
    }

    /**
     * 오늘 날짜 폴더에 그림(.jpg)과 같은 이름의 .json을 씁니다. JSON은 JsonMapper로 만들어 GPT 답에 따옴표가 섞여도 깨지지 않습니다.
     * 새 날짜 폴더를 만들 때도 보관 기간 지난 폴더를 지웁니다(매일 새벽 지우기와 별도로).
     */
    private Path write(String prefix, BufferedImage image, Map<String, Object> meta, String what) {
        if (!enabled()) {
            return null;
        }
        try {
            LocalDate today = LocalDate.now(clock);
            Path dir = Path.of(props.candidateDir(), today.toString());
            if (Files.notExists(dir)) {
                prune(today);
            }
            Files.createDirectories(dir);
            String id = prefix + UUID.randomUUID();
            Path jpg = dir.resolve(id + ".jpg");
            ImageIO.write(image, "jpg", jpg.toFile());
            Files.writeString(dir.resolve(id + ".json"), json.writeValueAsString(meta) + System.lineSeparator());
            return jpg;
        } catch (IOException | RuntimeException e) {
            log.warn("{} 저장 실패: {}", what, e.getClass().getSimpleName());
            return null;
        }
    }

    private static double round(double v) {
        return Math.round(v * 1000) / 1000.0;
    }

    /**
     * 매일 새벽 3시 30분(한국 시각)과 서버를 켤 때 보관 기간 지난 날짜 폴더를 지웁니다.
     * 예전에는 새 후보를 저장할 때만 지워서, 저장이 없는 동안은 30일이 지나도 사진이 남았습니다(2026-11-20 점검).
     */
    @Scheduled(cron = "${clothes.gpt-check.prune-cron:0 30 3 * * *}", zone = "Asia/Seoul")
    @EventListener(ApplicationReadyEvent.class)
    public void pruneNow() {
        if (enabled()) {
            prune(LocalDate.now(clock));
        }
    }

    /** 날짜 폴더(yyyy-MM-dd) 중 retentionDays보다 오래된 것을 지움. */
    public void prune(LocalDate today) {
        Path root = Path.of(props.candidateDir());
        if (!Files.isDirectory(root)) {
            return;
        }
        LocalDate oldest = today.minusDays(props.retentionDays());
        try (var dirs = Files.list(root)) {
            for (Path d : dirs.toList()) {
                LocalDate date;
                try {
                    date = LocalDate.parse(d.getFileName().toString());
                } catch (java.time.format.DateTimeParseException e) {
                    continue;
                }
                if (date.isBefore(oldest)) {
                    try (var files = Files.walk(d)) {
                        for (Path f : files.sorted(java.util.Comparator.reverseOrder()).toList()) {
                            Files.deleteIfExists(f);
                        }
                    }
                }
            }
        } catch (IOException e) {
            log.warn("오래된 학습 후보 지우기 실패: {}", e.getClass().getSimpleName());
        }
    }
}
