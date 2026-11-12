package com.fashionagent.clothesstate.color;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.core.io.ClassPathResource;

/** 레포에 든 색 인식 모델(models/color20_cnn.onnx)을 실제로 돌려 봄. */
class ColorModelTest {

    private static ColorModel model;

    @BeforeAll
    static void load() {
        model = new ColorModel(new ClassPathResource("models/color20_cnn.onnx"));
    }

    @AfterAll
    static void close() throws Exception {
        model.close();
    }

    /** 옅은 회색 배경 가운데 옷 색 사각형(옷 박스 조각 흉내). */
    private static BufferedImage crop(Color garment) {
        BufferedImage img = new BufferedImage(300, 400, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        g.setColor(new Color(235, 235, 235));
        g.fillRect(0, 0, 300, 400);
        g.setColor(garment);
        g.fillRect(40, 30, 220, 360);
        g.dispose();
        return img;
    }

    private static GarmentColor.Result color(Color garment) {
        BufferedImage c = crop(garment);
        return GarmentColor.fromModel(model.probabilities(c), GarmentColor.boxRgb(c, "worn"));
    }

    @Test
    @DisplayName("모델을 읽고 20가지 확률(합 1)을 냄")
    void probabilities() {
        assertThat(model.ready()).isTrue();
        float[] p = model.probabilities(crop(Color.WHITE));
        assertThat(p).hasSize(ColorModel.CLASSES.length);
        float sum = 0;
        for (float v : p) {
            sum += v;
        }
        assertThat(sum).isCloseTo(1f, within(1e-3f));
    }

    @Test
    @DisplayName("흰색·검정·네이비·하늘색·초록·주황")
    void basics() {
        assertThat(color(new Color(245, 245, 245))).isEqualTo(new GarmentColor.Result("white", "흰색", "light"));
        assertThat(color(new Color(20, 20, 22))).isEqualTo(new GarmentColor.Result("black", "검정", "dark"));
        assertThat(color(new Color(25, 35, 80)).colorFamily()).isEqualTo("navy");
        assertThat(color(new Color(150, 195, 235))).isEqualTo(new GarmentColor.Result("blue", "하늘색", "light"));
        assertThat(color(new Color(40, 120, 60)).colorFamily()).isEqualTo("green");
        assertThat(color(new Color(230, 120, 40)).colorFamily()).isEqualTo("orange");
    }

    @Test
    @DisplayName("쨍한 빨강은 '밝은 빨강'이 아니라 빨강(mid), 연분홍은 연한 색(light)")
    void vividAndPale() {
        assertThat(color(new Color(200, 30, 35))).isEqualTo(new GarmentColor.Result("red", "빨강", "mid"));
        assertThat(color(new Color(240, 200, 210))).isEqualTo(new GarmentColor.Result("pink", "연분홍", "light"));
    }

    @Test
    @DisplayName("모델 파일이 없으면 ready=false (픽셀 규칙으로 돌아감)")
    void missing() {
        assertThat(new ColorModel(new ClassPathResource("models/none.onnx")).ready()).isFalse();
    }
}
