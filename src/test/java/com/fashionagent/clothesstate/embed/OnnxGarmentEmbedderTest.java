package com.fashionagent.clothesstate.embed;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.nio.file.Path;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * test/resources/models/tiny-embed.onnx: 입력 pixel_values [batch,3,224,224] → 채널 평균 → 3×768 곱 + 더하기 →
 * [batch,768]. 실제 FashionSigLIP과 입출력 모양만 같은 작은 모델입니다.
 */
class OnnxGarmentEmbedderTest {

    private static EmbedProperties props(String model) {
        return new EmbedProperties(model, "marqo-fashionSigLIP", 224, 0.03, 0.5f, 0.5f, true, 5);
    }

    private static String tiny() throws Exception {
        return Path.of(OnnxGarmentEmbedderTest.class.getResource("/models/tiny-embed.onnx").toURI()).toString();
    }

    @Test
    @DisplayName("모델 경로가 비었거나 파일이 없으면 준비 안 됨")
    void missing() {
        assertThat(new OnnxGarmentEmbedder(props("")).ready()).isFalse();
        assertThat(new OnnxGarmentEmbedder(props("/no/such/model.onnx")).ready()).isFalse();
    }

    @Test
    @DisplayName("768개, 길이 1, 소수점 5자리. 옷 색이 다르면 벡터도 다름")
    void embeds() throws Exception {
        try (var e = new OnnxGarmentEmbedder(props(tiny()))) {
            assertThat(e.ready()).isTrue();
            assertThat(e.version()).isEqualTo("marqo-fashionSigLIP");
            var img = new BufferedImage(400, 400, BufferedImage.TYPE_INT_RGB);
            Graphics2D g = img.createGraphics();
            g.setColor(new Color(220, 20, 20));
            g.fillRect(0, 0, 200, 400);
            g.setColor(new Color(20, 20, 220));
            g.fillRect(200, 0, 200, 400);
            g.dispose();
            float[] red = e.embed(img, 0.02, 0.1, 0.4, 0.8);
            float[] blue = e.embed(img, 0.55, 0.1, 0.4, 0.8);
            assertThat(red).hasSize(768);
            double norm = 0;
            double dot = 0;
            for (int i = 0; i < 768; i++) {
                norm += red[i] * red[i];
                dot += red[i] * blue[i];
                assertThat(Math.abs(red[i] * 1e5 - Math.round(red[i] * 1e5))).isLessThan(0.05);
            }
            assertThat(Math.sqrt(norm)).isCloseTo(1.0, within(1e-3));
            assertThat(dot).isLessThan(0.999);
            // 같은 옷을 다시 넣으면 같은 벡터
            assertThat(e.embed(img, 0.02, 0.1, 0.4, 0.8)).containsExactly(red);
        }
    }

    @Test
    @DisplayName("원본·뒤집기를 각각 길이 1로 맞춰 평균한 뒤 다시 길이 1")
    void average() {
        float[] v = OnnxGarmentEmbedder.average(new float[][] {{3, 0}, {0, 10}}, 5);
        assertThat(v[0]).isCloseTo(0.70711f, within(1e-5f));
        assertThat(v[1]).isCloseTo(0.70711f, within(1e-5f));
    }
}
