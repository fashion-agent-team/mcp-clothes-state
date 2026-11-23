package com.fashionagent.clothesstate.pattern;

import static org.assertj.core.api.Assertions.assertThat;

import tools.jackson.databind.JsonNode;
import tools.jackson.databind.json.JsonMapper;
import java.io.File;
import java.util.List;
import javax.imageio.ImageIO;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;
import org.springframework.core.io.ClassPathResource;

/**
 * 서버 입력 만들기(Java)가 학습 평가(PIL)와 같은 답을 내는지 실제 모델로 확인. 평소에는 건너뛰고,
 * PATTERN_PARITY_DIR(사진들 + expected.json [{file, top, probs[8]}])이 있을 때만 돕니다.
 * LoRA 모델(BICUBIC)은 PATTERN_LORA_PARITY_DIR + PATTERN_LORA_MODEL(ONNX 경로)이 있을 때만
 * (training/scripts/exp3/make_lora_parity.py로 만듦).
 */
class OnnxPatternParityTest {

    @Test
    @EnabledIfEnvironmentVariable(named = "PATTERN_LORA_PARITY_DIR", matches = ".+")
    @DisplayName("LoRA 모델: PIL BICUBIC으로 만든 입력과 1등이 같고 확률 차이가 작다")
    void loraSameAsPil() throws Exception {
        File dir = new File(System.getenv("PATTERN_LORA_PARITY_DIR"));
        JsonNode expected = JsonMapper.builder().build().readTree(new File(dir, "expected.json"));
        var props = new PatternProperties("", "v", List.of("animal", "check", "dot", "floral", "graphic", "other",
                "plain", "stripe"), 224, 0.0, 0.0);
        var model = new OnnxPatternClassifier(new ClassPathResource("models/yolo26n_pattern8.onnx"),
                System.getenv("PATTERN_LORA_MODEL"), props);
        assertThat(model.mixed()).isTrue();
        int same = 0;
        double maxDiff = 0;
        for (JsonNode e : expected) {
            var img = ImageIO.read(new File(dir, e.get("file").asString()));
            float[] p = model.loraProbs(img, 0, 0, 1, 1);
            int top = 0;
            for (int i = 1; i < p.length; i++) {
                top = p[i] > p[top] ? i : top;
            }
            for (int i = 0; i < p.length; i++) {
                maxDiff = Math.max(maxDiff, Math.abs(p[i] - e.get("probs").get(i).asDouble()));
            }
            if (props.classes().get(top).equals(e.get("top").asString())) {
                same++;
            }
        }
        System.out.printf("lora pattern parity: %d/%d same top, max prob diff %.4f%n", same, expected.size(), maxDiff);
        assertThat(same).isGreaterThanOrEqualTo(expected.size() - 1);
        assertThat(maxDiff).isLessThan(0.05);
    }

    @Test
    @EnabledIfEnvironmentVariable(named = "PATTERN_PARITY_DIR", matches = ".+")
    @DisplayName("PIL로 만든 입력과 1등이 같고 확률 차이가 작다")
    void sameAsPil() throws Exception {
        File dir = new File(System.getenv("PATTERN_PARITY_DIR"));
        JsonNode expected = JsonMapper.builder().build().readTree(new File(dir, "expected.json"));
        var props = new PatternProperties("", "v", List.of("animal", "check", "dot", "floral", "graphic", "other",
                "plain", "stripe"), 224, 0.0, 0.0);
        var model = new OnnxPatternClassifier(new ClassPathResource("models/yolo26n_pattern8.onnx"), props);
        assertThat(model.ready()).isTrue();
        int same = 0;
        double maxDiff = 0;
        for (JsonNode e : expected) {
            var img = ImageIO.read(new File(dir, e.get("file").asString()));
            var r = model.classify(img, 0, 0, 1, 1);
            int top = props.classes().indexOf(r.top());
            double diff = Math.abs(r.confidence() - e.get("probs").get(top).asDouble());
            maxDiff = Math.max(maxDiff, diff);
            if (r.top().equals(e.get("top").asString())) {
                same++;
            }
        }
        System.out.printf("pattern parity: %d/%d same top, max prob diff %.4f%n", same, expected.size(), maxDiff);
        assertThat(same).isEqualTo(expected.size());
        assertThat(maxDiff).isLessThan(0.03);
    }
}
