package com.fashionagent.clothesstate.wear;

import static org.assertj.core.api.Assertions.assertThat;

import java.awt.image.BufferedImage;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.core.io.DefaultResourceLoader;

class ModelWearCheckerTest {

    /** 정해 둔 확률을 돌려주는 시험용 분류기. 받은 자르기 박스를 기억. */
    static final class Fixed extends OnnxBinaryClassifier {
        final double p;
        double[] last;

        Fixed(double p) {
            super(null, 224, 1, "fixed");
            this.p = p;
        }

        @Override
        public boolean ready() {
            return true;
        }

        @Override
        public double probability(BufferedImage img, double x, double y, double w, double h, double margin) {
            last = new double[] {x, y, w, h, margin};
            return p;
        }
    }

    private static final WearModelProperties PROPS = new WearModelProperties(true, "a", "b", "c", "wear1", 0.8, 0.8, 0.8);
    private final BufferedImage img = new BufferedImage(10, 10, BufferedImage.TYPE_INT_RGB);
    private final double[] box = {0.2, 0.2, 0.5, 0.4};

    @Test
    @DisplayName("택: 뒷모습이면 tag_out, 앞모습이면 backwards. 목 부분(가운데 60%, 위 -10%~+30%)만 잘라 봄")
    void tag() {
        var tag = new Fixed(0.9);
        var c = new ModelWearChecker(PROPS, tag, null, null);
        assertThat(c.check(img, box, "top", "티셔츠", "back")).extracting(WearChecker.Issue::kind).containsExactly("tag_out");
        assertThat(c.check(img, box, "top", "티셔츠", "front")).extracting(WearChecker.Issue::kind)
                .containsExactly("backwards");
        assertThat(tag.last[0]).isEqualTo(0.3, org.assertj.core.data.Offset.offset(1e-9));
        assertThat(tag.last[1]).isEqualTo(0.16, org.assertj.core.data.Offset.offset(1e-9));
        assertThat(tag.last[2]).isEqualTo(0.3, org.assertj.core.data.Offset.offset(1e-9));
        assertThat(tag.last[3]).isEqualTo(0.16, org.assertj.core.data.Offset.offset(1e-9));
        assertThat(c.check(img, box, "bottom", "바지", "back")).isEmpty();
    }

    @Test
    @DisplayName("단추는 셔츠·가디건 앞모습만, 뒤집힘은 셔츠·티셔츠·니트·원피스, 기준 0.8 미만은 버림")
    void buttonsAndInside() {
        var c = new ModelWearChecker(PROPS, null, new Fixed(0.85), new Fixed(0.95));
        assertThat(c.check(img, box, "top", "셔츠", "front")).extracting(WearChecker.Issue::kind)
                .containsExactly("buttons_misaligned", "inside_out");
        assertThat(c.check(img, box, "top", "셔츠", "back")).extracting(WearChecker.Issue::kind)
                .containsExactly("inside_out");
        assertThat(c.check(img, box, "top", "티셔츠", "front")).extracting(WearChecker.Issue::kind)
                .containsExactly("inside_out");
        assertThat(c.check(img, box, "outer", "자켓", "front")).isEmpty();
        var low = new ModelWearChecker(PROPS, new Fixed(0.79), new Fixed(0.5), new Fixed(0.1));
        assertThat(low.check(img, box, "top", "셔츠", "front")).isEmpty();
    }

    @Test
    @DisplayName("모델 파일이 없거나 꺼져 있으면 준비 안 됨")
    void notReady() {
        var none = new WearModelProperties(true, "classpath:nope1.onnx", "classpath:nope2.onnx", "classpath:nope3.onnx",
                "v", 0.8, 0.8, 0.8);
        assertThat(new ModelWearChecker(none, new DefaultResourceLoader()).ready()).isFalse();
        var off = new WearModelProperties(false, "a", "b", "c", "v", 0.8, 0.8, 0.8);
        assertThat(new ModelWearChecker(off, new DefaultResourceLoader()).ready()).isFalse();
    }

    @Test
    @DisplayName("레포에 넣은 모델 파일을 읽음")
    void realModels() {
        var props = new WearModelProperties(true, "classpath:models/yolo26n_wear_tag.onnx",
                "classpath:models/yolo26n_wear_button.onnx", "classpath:models/yolo26n_wear_inside.onnx", "v", 0.8, 0.8, 0.8);
        var c = new ModelWearChecker(props, new DefaultResourceLoader());
        assertThat(c.ready()).isTrue();
        var white = new BufferedImage(400, 600, BufferedImage.TYPE_INT_RGB);
        assertThat(c.check(white, new double[] {0.1, 0.1, 0.8, 0.6}, "top", "셔츠", "front")).isNotNull();
    }
}
