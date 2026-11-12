package com.fashionagent.clothesstate.verify;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDate;
import java.time.ZoneId;
import java.util.Locale;
import java.util.UUID;
import javax.imageio.ImageIO;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

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

    public CandidateStore(GptCheckProperties props) {
        this.props = props;
    }

    public boolean enabled() {
        return props.candidateDir() != null && !props.candidateDir().isBlank();
    }

    /** 저장한 파일 경로, 저장 안 했으면 null. 실패해도 검사는 계속합니다. */
    public Path save(BufferedImage crop, String modelKind, double modelConfidence, CropVerifier.Verdict v) {
        if (!enabled()) {
            return null;
        }
        try {
            LocalDate today = LocalDate.now(KST);
            Path dir = Path.of(props.candidateDir(), today.toString());
            if (Files.notExists(dir)) {
                prune(today); // 하루 한 번(새 날짜 폴더를 만들 때) 보관 기간이 지난 폴더를 지움
            }
            Files.createDirectories(dir);
            String id = UUID.randomUUID().toString();
            Path jpg = dir.resolve(id + ".jpg");
            ImageIO.write(crop, "jpg", jpg.toFile());
            String meta = String.format(Locale.ROOT,
                    "{\"label\":\"%s\",\"modelKind\":\"%s\",\"modelConfidence\":%.3f,\"gptConfidence\":%.3f,\"gptModel\":\"%s\",\"reviewed\":false}%n",
                    v.label(), modelKind, modelConfidence, v.confidence(), v.model());
            Files.writeString(dir.resolve(id + ".json"), meta);
            return jpg;
        } catch (IOException | RuntimeException e) {
            log.warn("학습 후보 저장 실패: {}", e.getClass().getSimpleName());
            return null;
        }
    }

    /**
     * 무늬 재학습 후보: 옷 박스 그림(보낸 것과 같은 크기)과 .json(task=pattern, GPT 답·확신도, 우리 모델 1등·확률).
     * GPT 답도 틀릴 수 있어 reviewed=false로 두고 사람이 확인한 뒤 학습에 씁니다.
     */
    public Path savePattern(BufferedImage garment, String gptPattern, double gptConfidence, String gptModel,
            String modelTop, double modelConfidence) {
        if (!enabled()) {
            return null;
        }
        try {
            LocalDate today = LocalDate.now(KST);
            Path dir = Path.of(props.candidateDir(), today.toString());
            if (Files.notExists(dir)) {
                prune(today);
            }
            Files.createDirectories(dir);
            String id = "pattern-" + UUID.randomUUID();
            Path jpg = dir.resolve(id + ".jpg");
            ImageIO.write(garment, "jpg", jpg.toFile());
            String meta = String.format(Locale.ROOT,
                    "{\"task\":\"pattern\",\"label\":\"%s\",\"gptConfidence\":%.3f,\"gptModel\":\"%s\",\"modelTop\":\"%s\",\"modelConfidence\":%.3f,\"reviewed\":false}%n",
                    gptPattern, gptConfidence, gptModel, modelTop, modelConfidence);
            Files.writeString(dir.resolve(id + ".json"), meta);
            return jpg;
        } catch (IOException | RuntimeException e) {
            log.warn("무늬 학습 후보 저장 실패: {}", e.getClass().getSimpleName());
            return null;
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
