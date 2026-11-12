package com.fashionagent.clothesstate.pattern;

import static org.assertj.core.api.Assertions.assertThat;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.util.List;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.core.io.ByteArrayResource;

class OnnxPatternClassifierTest {

    private static final List<String> CLASSES = List.of("animal", "check", "dot", "floral", "graphic", "other", "plain",
            "stripe");

    private static float[] probs(int top, float p) {
        float[] out = new float[8];
        java.util.Arrays.fill(out, (1 - p) / 7);
        out[top] = p;
        return out;
    }

    @Test
    @DisplayName("무지는 확신이 낮아도 plain, 무늬는 기준 이상일 때만 이름, 아니면 null(모름)")
    void decide() {
        assertThat(OnnxPatternClassifier.decide(probs(6, 0.4f), CLASSES, 0.6).pattern()).isEqualTo("plain");
        assertThat(OnnxPatternClassifier.decide(probs(7, 0.8f), CLASSES, 0.6).pattern()).isEqualTo("stripe");
        var unsure = OnnxPatternClassifier.decide(probs(1, 0.5f), CLASSES, 0.6);
        assertThat(unsure.pattern()).isNull();
        assertThat(unsure.top()).isEqualTo("check");
        assertThat(unsure.confidence()).isEqualTo(0.5);
    }

    @Test
    @DisplayName("모델 파일이 없거나 깨졌으면 준비 안 됨")
    void noModel() {
        var props = new PatternProperties("none", "v", CLASSES, 224, 0.08, 0.6);
        assertThat(new OnnxPatternClassifier(null, props).ready()).isFalse();
        assertThat(new OnnxPatternClassifier(new ByteArrayResource(new byte[] {1, 2, 3}), props).ready()).isFalse();
    }

    @Test
    @DisplayName("입력: 짧은 변을 224로 줄이고 가운데를 자름, 0~1 RGB CHW")
    void input() {
        // 왼쪽 절반 빨강, 오른쪽 절반 파랑인 가로로 긴 사진 → 가운데 정사각형도 왼쪽 빨강·오른쪽 파랑
        BufferedImage img = new BufferedImage(800, 400, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        g.setColor(Color.RED);
        g.fillRect(0, 0, 400, 400);
        g.setColor(Color.BLUE);
        g.fillRect(400, 0, 400, 400);
        g.dispose();
        float[] chw = PatternInput.prepare(img, 0, 0, 1, 1, 0.08, 224);
        int plane = 224 * 224;
        assertThat(chw).hasSize(3 * plane);
        int mid = 112 * 224;
        assertThat(chw[mid + 5]).isEqualTo(1f);                // R, 왼쪽
        assertThat(chw[2 * plane + mid + 5]).isEqualTo(0f);    // B, 왼쪽
        assertThat(chw[2 * plane + mid + 218]).isEqualTo(1f);  // B, 오른쪽
        assertThat(chw[mid + 218]).isEqualTo(0f);              // R, 오른쪽
    }

    @Test
    @DisplayName("크기 그대로면 픽셀 그대로(PIL BILINEAR)")
    void identityResize() {
        int[] src = {0, 10, 200, 255, 30, 40};
        var w = PatternInput.weights(3, 3);
        var h = PatternInput.weights(2, 2);
        assertThat(PatternInput.resize(src, 3, 2, 3, 2, w, h)).containsExactly(src);
    }
}
