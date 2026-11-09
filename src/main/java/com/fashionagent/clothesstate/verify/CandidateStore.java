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
 * 폴더/날짜/아이디.jpg 와 같은 이름의 .json(종류, 우리 모델 확신도, GPT 확신도, 모델). 사용자 정보는 넣지 않습니다.
 * 바로 학습하지 않고, 사람이 확인한 뒤 학습에 씁니다(docs/gpt-check.md).
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
            Path dir = Path.of(props.candidateDir(), LocalDate.now(KST).toString());
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
}
