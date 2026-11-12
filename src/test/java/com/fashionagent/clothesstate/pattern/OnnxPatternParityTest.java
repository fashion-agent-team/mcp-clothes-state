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
 */
@EnabledIfEnvironmentVariable(named = "PATTERN_PARITY_DIR", matches = ".+")
class OnnxPatternParityTest {

    @Test
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
            var img = ImageIO.read(new File(dir, e.get("file").asText()));
            var r = model.classify(img, 0, 0, 1, 1);
            int top = props.classes().indexOf(r.top());
            double diff = Math.abs(r.confidence() - e.get("probs").get(top).asDouble());
            maxDiff = Math.max(maxDiff, diff);
            if (r.top().equals(e.get("top").asText())) {
                same++;
            }
        }
        System.out.printf("pattern parity: %d/%d same top, max prob diff %.4f%n", same, expected.size(), maxDiff);
        assertThat(same).isEqualTo(expected.size());
        assertThat(maxDiff).isLessThan(0.03);
    }
}
