package com.fashionagent.clothesstate.detect;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import com.fashionagent.clothesstate.image.Images;
import java.awt.image.BufferedImage;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

class YoloOutputTest {

    // 가로 1024 × 세로 512 사진 → 512 입력이면 0.5배, 위아래로 128px 회색 패딩
    private final Images.Letterbox lb = Images.letterbox(new BufferedImage(1024, 512, BufferedImage.TYPE_INT_RGB), 512);

    @Test
    @DisplayName("letterbox: 비율 유지, 가운데 정렬 패딩")
    void letterbox() {
        assertThat(lb.scale()).isEqualTo(0.5);
        assertThat(lb.padX()).isZero();
        assertThat(lb.padY()).isEqualTo(128);
        assertThat(lb.chw()).hasSize(3 * 512 * 512);
        assertThat(lb.chw()[0]).isEqualTo(114 / 255f, within(1e-6f));
    }

    @Test
    @DisplayName("rect letterbox: 학습 때처럼 짧은 변은 32의 배수까지만 채움 (1024×700 → 512×352)")
    void rectLetterbox() {
        var r = Images.letterbox(new BufferedImage(1024, 700, BufferedImage.TYPE_INT_RGB), 512, true);
        assertThat(r.inputW()).isEqualTo(512);
        assertThat(r.inputH()).isEqualTo(352);
        assertThat(r.padY()).isEqualTo(1);
        assertThat(r.chw()).hasSize(3 * 512 * 352);
        assertThat(r.toRatioY(1 + 350 / 2.0)).isEqualTo(0.5, within(1e-6));
    }

    @Test
    @DisplayName("[1, 6, N] 출력: 패딩을 빼고 원본 비율로 되돌림, 기준 미만은 버림, 찢어짐은 tear")
    void channelsFirst() {
        float[][] out = new float[6][3];
        set(out, 0, 256, 256, 64, 32, 0.8f, 0.1f);   // 가운데 얼룩
        set(out, 1, 100, 200, 20, 20, 0.1f, 0.45f);  // 찢어짐
        set(out, 2, 400, 300, 10, 10, 0.2f, 0.1f);   // 기준 미만
        var found = YoloOutput.parse(out, lb, 0.3, 0.5);
        assertThat(found).hasSize(2);
        Detection stain = found.get(0);
        assertThat(stain.kind()).isEqualTo("stain");
        assertThat(stain.confidence()).isEqualTo(0.8, within(1e-6));
        assertThat(stain.centerX()).isEqualTo(0.5, within(1e-6));
        assertThat(stain.centerY()).isEqualTo(0.5, within(1e-6));
        assertThat(stain.w()).isEqualTo(0.125, within(1e-6));
        assertThat(stain.h()).isEqualTo(0.125, within(1e-6));
        assertThat(found.get(1).kind()).isEqualTo("tear");
    }

    @Test
    @DisplayName("[1, N, 6] 출력도 읽음")
    void rowsFirst() {
        float[][] out = {{256, 256, 64, 32, 0.9f, 0.0f}, {256, 256, 64, 32, 0.0f, 0.2f}};
        var found = YoloOutput.parse(out, lb, 0.3, 0.5);
        assertThat(found).singleElement().extracting(Detection::kind).isEqualTo("stain");
    }

    @Test
    @DisplayName("같은 종류가 많이 겹치면 확신도 높은 것 하나만, 다른 종류는 둘 다 남김")
    void nms() {
        float[][] out = new float[6][3];
        set(out, 0, 256, 256, 64, 64, 0.6f, 0f);
        set(out, 1, 260, 258, 64, 64, 0.9f, 0f);
        set(out, 2, 258, 256, 64, 64, 0f, 0.7f);
        var found = YoloOutput.parse(out, lb, 0.3, 0.5);
        assertThat(found).extracting(Detection::kind, Detection::confidence)
                .containsExactly(org.assertj.core.groups.Tuple.tuple("stain", (double) 0.9f),
                        org.assertj.core.groups.Tuple.tuple("tear", (double) 0.7f));
    }

    @Test
    @DisplayName("코트·재킷: 다른 쪽 점수가 1등의 0.1배 이상이면 겉옷(outer_unsure), 아니면 그대로")
    void outerUnsure() {
        String[] kinds = OnnxGarmentDetector.KINDS12;
        float[][] out = new float[16][3];
        int jacket = 4 + 4, coat = 4 + 5;
        for (int i = 0; i < 3; i++) {
            out[0][i] = 100 + 150 * i;
            out[1][i] = 256;
            out[2][i] = 60;
            out[3][i] = 60;
        }
        out[jacket][0] = 0.8f;
        out[coat][0] = 0.12f;  // 0.12 >= 0.08 → 겉옷
        out[jacket][1] = 0.8f;
        out[coat][1] = 0.05f;  // 0.05 < 0.08 → 재킷
        out[4 + 0][2] = 0.7f;  // 셔츠는 상관없음
        out[coat][2] = 0.6f;
        var rule = new YoloOutput.UnsurePair("jacket", "coat", 0.1, OnnxGarmentDetector.OUTER_UNSURE);
        var found = YoloOutput.parse(out, lb, 0.3, 0.5, kinds, rule);
        assertThat(found).extracting(Detection::kind)
                .containsExactlyInAnyOrder("outer_unsure", "jacket", "shirt");
        assertThat(YoloOutput.parse(out, lb, 0.3, 0.5, kinds)).extracting(Detection::kind)
                .containsExactlyInAnyOrder("jacket", "jacket", "shirt");
    }

    private static void set(float[][] out, int i, float cx, float cy, float w, float h, float stain, float torn) {
        out[0][i] = cx;
        out[1][i] = cy;
        out[2][i] = w;
        out[3][i] = h;
        out[4][i] = stain;
        out[5][i] = torn;
    }
}
