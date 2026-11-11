package com.fashionagent.clothesstate.embed;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

class EmbedInputTest {

    private static final int S = 224;
    private static final int PLANE = S * S;

    /** 0~255 값을 정규화(mean 0.5, std 0.5)한 값. */
    private static float n(int v) {
        return (v / 255f - 0.5f) / 0.5f;
    }

    @Test
    @DisplayName("가로로 긴 옷 박스는 위아래를 회색(124,116,104)으로 채우고 224×224, 가운데는 옷 색")
    void padsToSquare() {
        var img = new BufferedImage(400, 300, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, 400, 300);
        g.setColor(new Color(200, 30, 40));
        g.fillRect(100, 100, 200, 100); // 옷 박스 = 정확히 이 빨간 칸
        g.dispose();
        var p = EmbedInput.prepare(img, 0.25, 1 / 3.0, 0.5, 1 / 3.0, 0.03, S, 0.5f, 0.5f);
        assertThat(p.chw()).hasSize(3 * PLANE);
        // 맨 위 줄은 회색
        assertThat(p.chw()[0]).isCloseTo(n(124), within(1e-6f));
        assertThat(p.chw()[PLANE]).isCloseTo(n(116), within(1e-6f));
        assertThat(p.chw()[2 * PLANE]).isCloseTo(n(104), within(1e-6f));
        // 가운데는 빨강
        int mid = 112 * S + 112;
        assertThat(p.chw()[mid]).isCloseTo(n(200), within(1e-6f));
        assertThat(p.chw()[PLANE + mid]).isCloseTo(n(30), within(1e-6f));
    }

    @Test
    @DisplayName("뒤집은 입력은 원본을 좌우로 바꾼 것")
    void flipped() {
        var img = new BufferedImage(300, 300, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        g.setColor(Color.BLACK);
        g.fillRect(0, 0, 150, 300);
        g.setColor(Color.WHITE);
        g.fillRect(150, 0, 150, 300);
        g.dispose();
        var p = EmbedInput.prepare(img, 0, 0, 1, 1, 0.03, S, 0.5f, 0.5f);
        for (int c = 0; c < 3; c++) {
            for (int r = 0; r < S; r += 37) {
                for (int x = 0; x < S; x++) {
                    assertThat(p.flipped()[c * PLANE + r * S + x]).isEqualTo(p.chw()[c * PLANE + r * S + (S - 1 - x)]);
                }
            }
        }
        assertThat(p.chw()[10]).isCloseTo(-1f, within(1e-6f)); // 왼쪽 검정
        assertThat(p.flipped()[10]).isCloseTo(1f, within(1e-6f)); // 뒤집으면 흰색
    }

    @Test
    @DisplayName("옷 박스를 사방 3% 넓혀 자르되 사진 밖으로는 안 나감")
    void marginInsideImage() {
        var img = new BufferedImage(1000, 1000, BufferedImage.TYPE_INT_RGB);
        int[][] crop = EmbedInput.squareCrop(img, 0.1, 0.1, 0.5, 0.5, 0.03);
        // 500px 박스 + 사방 15px = 530
        assertThat(crop[0]).hasSize(530 * 530);
        int[][] edge = EmbedInput.squareCrop(img, 0, 0, 1, 1, 0.03);
        assertThat(edge[0]).hasSize(1000 * 1000);
    }

    @Test
    @DisplayName("같은 색 사진을 줄이면 그 색 그대로(필터 합이 1)")
    void resizeKeepsFlatColor() {
        int side = 1000;
        int[][] rgb = new int[3][side * side];
        java.util.Arrays.fill(rgb[0], 77);
        java.util.Arrays.fill(rgb[1], 150);
        java.util.Arrays.fill(rgb[2], 233);
        float[][] out = EmbedInput.resize(rgb, S);
        assertThat(out[0]).containsOnly(77f);
        assertThat(out[1]).containsOnly(150f);
        assertThat(out[2]).containsOnly(233f);
    }
}
