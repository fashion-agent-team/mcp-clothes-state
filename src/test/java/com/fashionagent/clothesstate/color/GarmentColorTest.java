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

    private static float[] onehot(String label) {
        float[] p = new float[ColorModel.CLASSES.length];
        p[java.util.Arrays.asList(ColorModel.CLASSES).indexOf(label)] = 1;
        return p;
    }

    @Test
    @DisplayName("모델 답 → 색 이름: 와인은 어두운 빨강(차분한 색), 라벤더·민트는 연한 색 이름, 회색 밝기는 픽셀로")
    void fromModel() {
        int mid = 0x808080;
        assertThat(GarmentColor.fromModel(onehot("와인"), mid)).isEqualTo(new GarmentColor.Result("red", "와인색", "dark"));
        assertThat(GarmentColor.fromModel(onehot("라벤더"), mid)).isEqualTo(new GarmentColor.Result("purple", "연보라", "light"));
        assertThat(GarmentColor.fromModel(onehot("민트"), mid)).isEqualTo(new GarmentColor.Result("green", "민트", "light"));
        assertThat(GarmentColor.fromModel(onehot("그레이"), 0x303030).colorName()).isEqualTo("차콜 그레이");
        assertThat(GarmentColor.fromModel(onehot("실버"), mid)).isEqualTo(new GarmentColor.Result("gray", "은색", "light"));
        // 픽셀이 연한 분홍(채도 낮고 밝음)이면 연분홍, 쨍하면 분홍
        assertThat(GarmentColor.fromModel(onehot("핑크"), 0xF0C8D2).colorName()).isEqualTo("연분홍");
        assertThat(GarmentColor.fromModel(onehot("핑크"), 0xE0307A)).isEqualTo(new GarmentColor.Result("pink", "분홍", "mid"));
        // 픽셀이 없으면(-1) 중간 밝기로
        assertThat(GarmentColor.fromModel(onehot("블루"), -1)).isEqualTo(new GarmentColor.Result("blue", "파랑", "mid"));
    }

    /** 흰 바닥 위 체크 옷 박스: [base] 바탕에 [line] 색 굵은 선을 가로세로로. 선 사이에 섞인 중간색도 조금. */
    private static BufferedImage check(Color base, Color line) {
        BufferedImage img = new BufferedImage(200, 200, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, 200, 200);
        g.setColor(base);
        g.fillRect(20, 20, 160, 160);
        Color mix = new Color((base.getRed() + line.getRed()) / 2, (base.getGreen() + line.getGreen()) / 2,
                (base.getBlue() + line.getBlue()) / 2);
        for (int i = 20; i < 180; i += 32) {
            g.setColor(line);
            g.fillRect(i, 20, 10, 160);
            g.fillRect(20, i, 160, 10);
            g.setColor(mix);
            g.fillRect(i + 10, 20, 3, 160);
        }
        g.dispose();
        return img;
    }

    @Test
    @DisplayName("무늬 옷 두 번째 색: 네이비 바탕 베이지 체크 → 베이지, 회색 바탕 검정 호피 → 회색")
    void secondColor() {
        var navy = new GarmentColor.Result("black", "검정", "dark");
        var sub = GarmentColor.secondColor(check(new Color(30, 35, 60), new Color(205, 185, 150)), "laid", navy);
        assertThat(sub).isNotNull();
        assertThat(sub.colorFamily()).isEqualTo("beige");
        var leopard = GarmentColor.secondColor(check(new Color(160, 165, 160), new Color(20, 20, 20)), "laid", navy);
        assertThat(leopard).isNotNull();
        assertThat(leopard.colorFamily()).isEqualTo("gray");
    }

    @Test
    @DisplayName("두 번째 색이 귀로 같은 색(검정 바탕 네이비 줄)이면 말하지 않음")
    void secondColorSameToHear() {
        var black = new GarmentColor.Result("black", "검정", "dark");
        assertThat(GarmentColor.secondColor(check(new Color(15, 15, 15), new Color(25, 35, 80)), "laid", black)).isNull();
        // 무지 옷(색 하나)도 null
        assertThat(GarmentColor.secondColor(laid(Color.WHITE, new Color(200, 30, 35)), "laid",
                new GarmentColor.Result("red", "빨강", "mid"))).isNull();
    }
}
