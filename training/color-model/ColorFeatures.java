package com.fashionagent.clothesstate.color;

import java.awt.Color;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.util.Arrays;

/**
 * 비교 실험(색 히스토그램 + MLP, mlp.py)에 쓴 옷 조각 특징값. 서버 GarmentColor의 median·skin을 그대로 쓰려고 같은 패키지에 둡니다.
 * 서버는 지금 CNN(ColorModel)만 써서 이 파일은 서버 코드에 넣지 않았습니다.
 * 컴파일: javac -cp "<서버 classes>" -d out ColorFeatures.java FeatDump.java
 *
 * <p>옷 조각을 긴 변 {@value #SIZE}px로 줄인 뒤 두 영역(가운데, 조각 전체)마다 HSV 히스토그램을 셉니다.
 * 조각 전체(배경·피부 포함)를 같이 보면 조명이 어두운지·노란지 모델이 비교할 수 있어서, 그늘진 흰 옷을
 * 회색으로 보는 실수가 줄어듭니다.
 * <ul>
 * <li>색 있는 픽셀(채도 0.15 이상): 색상 24칸 × 채도 2칸 × 밝기 3칸 = 144</li>
 * <li>무채색 픽셀(채도 0.15 미만): 밝기 8칸</li>
 * <li>가운데 영역 통계: 중간값 R·G·B, 채도·밝기 10·50·90 백분위, 피부색 비율 = 10</li>
 * </ul>
 */
public final class ColorFeatures {

    /** 학습 조각(AI Hub K-Fashion 정답 상자)을 저장한 크기와 같게 맞춥니다. */
    public static final int SIZE = 96;
    static final int HUE_BINS = 24;
    static final int PER_REGION = HUE_BINS * 2 * 3 + 8;
    public static final int LENGTH = PER_REGION * 2 + 10;

    private ColorFeatures() {
    }

    public static float[] of(BufferedImage crop) {
        BufferedImage img = fit(crop);
        int w = img.getWidth();
        int h = img.getHeight();
        float[] out = new float[LENGTH];
        int[] inner = pixels(img, (int) (w * 0.2), (int) (h * 0.15), Math.max((int) (w * 0.8), (int) (w * 0.2) + 1),
                Math.max((int) (h * 0.85), (int) (h * 0.15) + 1));
        int[] all = pixels(img, 0, 0, w, h);
        histogram(inner, out, 0);
        histogram(all, out, PER_REGION);
        stats(inner, out, PER_REGION * 2);
        return out;
    }

    /** 긴 변이 {@value #SIZE}px보다 크면 면적 평균으로 줄임(학습 조각을 만든 방식과 비슷하게). */
    static BufferedImage fit(BufferedImage src) {
        int w = src.getWidth();
        int h = src.getHeight();
        int longest = Math.max(w, h);
        if (longest <= SIZE) {
            return src;
        }
        int nw = Math.max(1, Math.round((float) w * SIZE / longest));
        int nh = Math.max(1, Math.round((float) h * SIZE / longest));
        Image scaled = src.getScaledInstance(nw, nh, Image.SCALE_AREA_AVERAGING);
        BufferedImage out = new BufferedImage(nw, nh, BufferedImage.TYPE_INT_RGB);
        out.getGraphics().drawImage(scaled, 0, 0, null);
        return out;
    }

    private static int[] pixels(BufferedImage img, int x0, int y0, int x1, int y1) {
        int[] out = new int[(x1 - x0) * (y1 - y0)];
        int i = 0;
        for (int y = y0; y < y1; y++) {
            for (int x = x0; x < x1; x++) {
                out[i++] = img.getRGB(x, y) & 0xFFFFFF;
            }
        }
        return out;
    }

    private static void histogram(int[] rgb, float[] out, int off) {
        float[] hsv = new float[3];
        for (int p : rgb) {
            Color.RGBtoHSB((p >> 16) & 0xFF, (p >> 8) & 0xFF, p & 0xFF, hsv);
            float s = hsv[1];
            float v = hsv[2];
            if (s < 0.15f) {
                out[off + HUE_BINS * 6 + Math.min(7, (int) (v * 8))] += 1;
            } else {
                int hb = Math.min(HUE_BINS - 1, (int) (hsv[0] * HUE_BINS));
                int sb = s < 0.45f ? 0 : 1;
                int vb = v < 0.35f ? 0 : v < 0.7f ? 1 : 2;
                out[off + (hb * 2 + sb) * 3 + vb] += 1;
            }
        }
        float n = Math.max(1, rgb.length);
        for (int i = 0; i < PER_REGION; i++) {
            out[off + i] /= n;
        }
    }

    private static void stats(int[] rgb, float[] out, int off) {
        int n = rgb.length;
        if (n == 0) {
            return;
        }
        int med = GarmentColor.median(rgb);
        out[off] = ((med >> 16) & 0xFF) / 255f;
        out[off + 1] = ((med >> 8) & 0xFF) / 255f;
        out[off + 2] = (med & 0xFF) / 255f;
        float[] s = new float[n];
        float[] v = new float[n];
        float[] hsv = new float[3];
        int skin = 0;
        for (int i = 0; i < n; i++) {
            int p = rgb[i];
            Color.RGBtoHSB((p >> 16) & 0xFF, (p >> 8) & 0xFF, p & 0xFF, hsv);
            s[i] = hsv[1];
            v[i] = hsv[2];
            if (GarmentColor.skin(p)) {
                skin++;
            }
        }
        Arrays.sort(s);
        Arrays.sort(v);
        out[off + 3] = s[n / 10];
        out[off + 4] = s[n / 2];
        out[off + 5] = s[n * 9 / 10];
        out[off + 6] = v[n / 10];
        out[off + 7] = v[n / 2];
        out[off + 8] = v[n * 9 / 10];
        out[off + 9] = (float) skin / n;
    }
}
