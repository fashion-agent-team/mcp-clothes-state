package com.fashionagent.clothesstate.color;

import static org.assertj.core.api.Assertions.assertThat;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

class GarmentColorTest {

    private static BufferedImage laid(Color floor, Color shirt) {
        BufferedImage img = new BufferedImage(400, 300, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        g.setColor(floor);
        g.fillRect(0, 0, 400, 300);
        g.setColor(shirt);
        g.fillRect(120, 70, 160, 160);
        g.dispose();
        return img;
    }

    @Test
    @DisplayName("바닥을 빼고 옷 색만 본다: 흰 바닥 위 네이비 셔츠")
    void navyOnWhite() {
        var r = GarmentColor.of(laid(Color.WHITE, new Color(25, 35, 80)), "laid");
        assertThat(r.colorFamily()).isEqualTo("navy");
        assertThat(r.colorName()).isEqualTo("네이비");
        assertThat(r.tone()).isEqualTo("dark");
    }

    @Test
    @DisplayName("어두운 바닥 위 베이지 니트, 회색 바닥 위 빨간 티")
    void others() {
        assertThat(GarmentColor.of(laid(new Color(40, 40, 40), new Color(215, 195, 160)), "laid").colorFamily())
                .isEqualTo("beige");
        assertThat(GarmentColor.of(laid(new Color(150, 150, 150), new Color(200, 30, 35)), "laid").colorFamily())
                .isEqualTo("red");
    }

    @Test
    @DisplayName("이름 규칙: 짙은 회색은 차콜 그레이, 검정·흰색")
    void names() {
        assertThat(GarmentColor.name(0x3C3C3E)).isEqualTo(new GarmentColor.Result("gray", "차콜 그레이", "dark"));
        assertThat(GarmentColor.name(0x111111).colorFamily()).isEqualTo("black");
        assertThat(GarmentColor.name(0xF4F4F2).colorFamily()).isEqualTo("white");
        assertThat(GarmentColor.name(0x2E7D32).colorFamily()).isEqualTo("green");
        assertThat(GarmentColor.name(0x1E88E5).colorFamily()).isEqualTo("blue");
        assertThat(GarmentColor.name(0x5D4037).colorFamily()).isEqualTo("brown");
    }

    @Test
    @DisplayName("worn: 화면 가운데 몸통 쪽 색")
    void worn() {
        BufferedImage img = laid(new Color(230, 230, 225), new Color(30, 120, 50));
        assertThat(GarmentColor.of(img, "worn").colorFamily()).isEqualTo("green");
    }

    @Test
    @DisplayName("화면에 띄운 사진을 비춘 것처럼 테두리가 검은 모니터여도 흰 배경을 옷으로 보지 않는다 (2026-11-09 리허설)")
    void photoOnScreen() {
        BufferedImage img = new BufferedImage(600, 800, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        g.setColor(new Color(15, 15, 18)); // 모니터 테두리
        g.fillRect(0, 0, 600, 800);
        g.setColor(new Color(235, 238, 245)); // 사진 속 흰 배경
        g.fillRect(60, 60, 480, 680);
        g.setColor(new Color(70, 52, 40)); // 갈색 자켓
        g.fillRect(200, 250, 200, 300);
        g.dispose();
        assertThat(GarmentColor.of(img, "laid").colorFamily()).isEqualTo("brown");
    }

    @Test
    @DisplayName("아주 어두운 청록 코트는 검정, 어두운 갈색은 회색이 아니라 갈색")
    void darkColors() {
        assertThat(GarmentColor.name(0x182828).colorFamily()).isEqualTo("black");
        assertThat(GarmentColor.name(0x3B3029).colorFamily()).isEqualTo("brown");
    }
}
