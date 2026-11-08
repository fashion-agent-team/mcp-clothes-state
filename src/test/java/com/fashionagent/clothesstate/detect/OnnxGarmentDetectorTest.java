package com.fashionagent.clothesstate.detect;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.List;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

class OnnxGarmentDetectorTest {

    @Test
    @DisplayName("종류가 달라도 박스가 거의 같으면 확신도 높은 하나만, 떨어진 옷은 그대로")
    void mergeSameGarment() {
        var top = new Detection("top", 0.62, 0.1, 0.1, 0.5, 0.5);
        var dress = new Detection("dress", 0.41, 0.11, 0.1, 0.5, 0.52);
        var shoe = new Detection("shoes", 0.8, 0.4, 0.9, 0.1, 0.05);
        assertThat(OnnxGarmentDetector.mergeSameGarment(List.of(dress, top, shoe), 0.7))
                .containsExactly(shoe, top);
    }

    @Test
    @DisplayName("클래스 순서는 학습 순서 = 약속 문서 category")
    void kinds() {
        assertThat(OnnxGarmentDetector.KINDS).containsExactly("top", "bottom", "outer", "dress", "shoes");
    }

    @Test
    @DisplayName("모델 파일이 없으면 준비 안 됨")
    void missingModel() {
        var d = new OnnxGarmentDetector(new org.springframework.core.io.ClassPathResource("models/none.onnx"), 512,
                0.5, 0.7);
        assertThat(d.ready()).isFalse();
    }
}
