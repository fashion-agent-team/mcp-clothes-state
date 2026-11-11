package com.fashionagent.clothesstate.color;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

class ColorDoubtTest {

    private static String alt(int r, int g, int b, boolean warm) {
        int rgb = (r << 16) | (g << 8) | b;
        return ColorDoubt.alternative(rgb, GarmentColor.name(rgb), warm);
    }

    @Test
    @DisplayName("흰 불: 분명한 검정·네이비·흰색·베이지는 후보 없음")
    void clearColors() {
        assertThat(alt(15, 15, 15, false)).isNull();      // 검정
        assertThat(alt(30, 45, 100, false)).isNull();     // 네이비(v≈0.39)
        assertThat(alt(250, 250, 250, false)).isNull();   // 흰색
        assertThat(alt(215, 195, 160, false)).isNull();   // 베이지
    }

    @Test
    @DisplayName("흰 불: 경계 근처만 짝을 붙임")
    void nearBoundary() {
        assertThat(GarmentColor.name(rgb(20, 24, 45)).colorFamily()).isEqualTo("black");
        assertThat(alt(20, 24, 45, false)).isEqualTo("네이비");   // 파란 기가 도는 검정
        assertThat(GarmentColor.name(rgb(25, 32, 58)).colorFamily()).isEqualTo("navy");
        assertThat(alt(25, 32, 58, false)).isEqualTo("검정");     // 아주 어두운 네이비
        assertThat(GarmentColor.name(rgb(250, 244, 230)).colorFamily()).isEqualTo("white");
        assertThat(alt(250, 244, 230, false)).isEqualTo("베이지"); // 노란 기가 도는 흰색
        assertThat(GarmentColor.name(rgb(230, 215, 190)).colorFamily()).isEqualTo("beige");
        assertThat(alt(230, 215, 190, false)).isEqualTo("흰색");   // 아주 연한 베이지
    }

    @Test
    @DisplayName("노란 불: 검정·네이비·흰색·베이지는 모두 짝, 다른 색은 없음")
    void warmLight() {
        assertThat(alt(15, 15, 15, true)).isEqualTo("네이비");
        assertThat(alt(30, 45, 100, true)).isEqualTo("검정");
        assertThat(alt(250, 250, 250, true)).isEqualTo("베이지");
        assertThat(alt(215, 195, 160, true)).isEqualTo("흰색");
        assertThat(alt(200, 30, 35, true)).isNull();
        assertThat(ColorDoubt.alternative(-1, null, true)).isNull();
    }

    private static int rgb(int r, int g, int b) {
        return (r << 16) | (g << 8) | b;
    }
}
