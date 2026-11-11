package com.fashionagent.clothesstate.image;

import static org.assertj.core.api.Assertions.assertThat;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

class WarmCastTest {

    private static BufferedImage photo(Color floor, Color garment) {
        BufferedImage img = new BufferedImage(200, 200, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        g.setColor(floor);
        g.fillRect(0, 0, 200, 200);
        g.setColor(garment);
        g.fillRect(50, 50, 100, 100);
        g.dispose();
        return img;
    }

    @Test
    @DisplayName("흰 불: 회색 바닥에 노란 옷이어도 노란 기운 낮음")
    void neutral() {
        assertThat(Images.warmCast(photo(new Color(200, 200, 200), new Color(240, 200, 60)))).isLessThan(12);
        assertThat(Images.warmCast(photo(new Color(240, 240, 240), new Color(25, 35, 80)))).isLessThan(12);
    }

    @Test
    @DisplayName("노란 불: 흰 바닥·흰 옷까지 노랗게 뜨면 높음")
    void warm() {
        // 흰 바닥(255,235,200)과 회색 옷(140,125,100)이 전구색 불빛에 노랗게 찍힌 모습
        assertThat(Images.warmCast(photo(new Color(255, 235, 200), new Color(140, 125, 100)))).isGreaterThan(12);
    }
}
