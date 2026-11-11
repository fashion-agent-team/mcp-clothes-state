package com.fashionagent.clothesstate.embed;

import java.awt.image.BufferedImage;

/**
 * 이미지 벡터 모델 입력 만들기(이슈 #15 측정과 같은 방식): 옷 박스를 사방 margin만큼 넓혀 자르고(사진 안), 긴 변 기준
 * 정사각형으로 회색(124,116,104)을 가운데 정렬로 채운 뒤 size×size로 줄이고, (x/255 - mean)/std CHW로 만듭니다.
 * 줄이기는 PIL Image.resize(BICUBIC)와 같게(줄이는 비율만큼 필터를 넓힘) 직접 계산합니다. Java2D 쌍삼차는 크게
 * 줄일 때 계단이 생겨 측정 때와 벡터가 달라집니다.
 */
public final class EmbedInput {

    /** 회색 채우기 색(측정 스크립트와 같음). */
    static final int PAD_R = 124;
    static final int PAD_G = 116;
    static final int PAD_B = 104;

    private EmbedInput() {
    }

    /** 원본과 좌우 뒤집은 입력. */
    public record Pair(float[] chw, float[] flipped) {
    }

    public static Pair prepare(BufferedImage img, double x, double y, double w, double h, double margin, int size,
            float mean, float std) {
        int[][] rgb = squareCrop(img, x, y, w, h, margin);
        float[][] planes = resize(rgb, size);
        int plane = size * size;
        float[] chw = new float[3 * plane];
        float[] flip = new float[3 * plane];
        for (int c = 0; c < 3; c++) {
            float[] p = planes[c];
            for (int r = 0; r < size; r++) {
                for (int col = 0; col < size; col++) {
                    // PIL처럼 0~255로 반올림·자른 뒤 정규화
                    float v = Math.max(0, Math.min(255, Math.round(p[r * size + col])));
                    float n = (v / 255f - mean) / std;
                    chw[c * plane + r * size + col] = n;
                    flip[c * plane + r * size + (size - 1 - col)] = n;
                }
            }
        }
        return new Pair(chw, flip);
    }

    /** 넓혀 자른 뒤 긴 변 기준 정사각형(회색 채움). 결과는 [채널][y*side+x] 0~255 정수. */
    static int[][] squareCrop(BufferedImage img, double x, double y, double w, double h, double margin) {
        int iw = img.getWidth();
        int ih = img.getHeight();
        double bw = w * iw;
        double bh = h * ih;
        int x0 = clamp((int) Math.floor(x * iw - bw * margin), 0, iw - 1);
        int y0 = clamp((int) Math.floor(y * ih - bh * margin), 0, ih - 1);
        int x1 = clamp((int) Math.ceil((x + w) * iw + bw * margin), x0 + 1, iw);
        int y1 = clamp((int) Math.ceil((y + h) * ih + bh * margin), y0 + 1, ih);
        int cw = x1 - x0;
        int ch = y1 - y0;
        int side = Math.max(cw, ch);
        int offX = (side - cw) / 2;
        int offY = (side - ch) / 2;
        int[] px = img.getRGB(x0, y0, cw, ch, null, 0, cw);
        int[][] out = new int[3][side * side];
        java.util.Arrays.fill(out[0], PAD_R);
        java.util.Arrays.fill(out[1], PAD_G);
        java.util.Arrays.fill(out[2], PAD_B);
        for (int r = 0; r < ch; r++) {
            for (int c = 0; c < cw; c++) {
                int v = px[r * cw + c];
                int i = (r + offY) * side + c + offX;
                out[0][i] = (v >> 16) & 0xFF;
                out[1][i] = (v >> 8) & 0xFF;
                out[2][i] = v & 0xFF;
            }
        }
        return out;
    }

    /** 정사각형 side×side → size×size, PIL BICUBIC(a=-0.5, 줄일 때 필터 폭을 비율만큼 넓힘) 가로·세로 따로. */
    static float[][] resize(int[][] rgb, int size) {
        int side = (int) Math.round(Math.sqrt(rgb[0].length));
        Weights wt = weights(side, size);
        float[][] out = new float[3][];
        for (int c = 0; c < 3; c++) {
            int[] src = rgb[c];
            // 가로
            float[] tmp = new float[side * size];
            for (int r = 0; r < side; r++) {
                int row = r * side;
                for (int o = 0; o < size; o++) {
                    double s = 0;
                    int start = wt.start[o];
                    double[] k = wt.k[o];
                    for (int j = 0; j < k.length; j++) {
                        s += src[row + start + j] * k[j];
                    }
                    tmp[r * size + o] = clip8(s); // PIL도 가로 결과를 8비트로 저장
                }
            }
            // 세로
            float[] dst = new float[size * size];
            for (int o = 0; o < size; o++) {
                int start = wt.start[o];
                double[] k = wt.k[o];
                for (int col = 0; col < size; col++) {
                    double s = 0;
                    for (int j = 0; j < k.length; j++) {
                        s += tmp[(start + j) * size + col] * k[j];
                    }
                    dst[o * size + col] = (float) s;
                }
            }
            out[c] = dst;
        }
        return out;
    }

    private record Weights(int[] start, double[][] k) {
    }

    /** PIL _imaging precompute_coeffs와 같은 계산. */
    private static Weights weights(int in, int out) {
        double scale = (double) in / out;
        double filterScale = Math.max(1.0, scale);
        double support = 2.0 * filterScale;
        int[] start = new int[out];
        double[][] k = new double[out][];
        for (int o = 0; o < out; o++) {
            double center = (o + 0.5) * scale;
            int xmin = Math.max(0, (int) (center - support + 0.5));
            int xmax = Math.min(in, (int) (center + support + 0.5));
            double[] w = new double[xmax - xmin];
            double sum = 0;
            for (int i = 0; i < w.length; i++) {
                w[i] = cubic((i + xmin - center + 0.5) / filterScale);
                sum += w[i];
            }
            if (sum != 0) {
                for (int i = 0; i < w.length; i++) {
                    w[i] /= sum;
                }
            }
            start[o] = xmin;
            k[o] = w;
        }
        return new Weights(start, k);
    }

    private static double cubic(double x) {
        double a = -0.5;
        x = Math.abs(x);
        if (x < 1) {
            return ((a + 2) * x - (a + 3)) * x * x + 1;
        }
        if (x < 2) {
            return (((x - 5) * x + 8) * x - 4) * a;
        }
        return 0;
    }

    private static float clip8(double v) {
        return (float) Math.max(0, Math.min(255, Math.round(v)));
    }

    private static int clamp(int v, int lo, int hi) {
        return Math.max(lo, Math.min(hi, v));
    }
}
