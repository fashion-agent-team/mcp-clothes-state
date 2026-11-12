package com.fashionagent.clothesstate.pattern;

import java.awt.image.BufferedImage;

/**
 * 무늬 모델 입력 만들기. 학습·평가(Ultralytics classify)와 같게 합니다: 옷 박스를 사방 margin만큼 넓혀 자르고(사진 안),
 * 짧은 변을 size로 줄인 뒤(PIL BILINEAR, 줄이는 비율만큼 필터를 넓힘) 가운데를 size×size로 자르고, x/255 RGB CHW.
 */
public final class PatternInput {

    private PatternInput() {
    }

    public static float[] prepare(BufferedImage img, double x, double y, double w, double h, double margin, int size) {
        int iw = img.getWidth();
        int ih = img.getHeight();
        double bw = w * iw;
        double bh = h * ih;
        int x0 = clamp((int) Math.round(x * iw - bw * margin), 0, iw - 1);
        int y0 = clamp((int) Math.round(y * ih - bh * margin), 0, ih - 1);
        int x1 = clamp((int) Math.round((x + w) * iw + bw * margin), x0 + 1, iw);
        int y1 = clamp((int) Math.round((y + h) * ih + bh * margin), y0 + 1, ih);
        int cw = x1 - x0;
        int ch = y1 - y0;
        int[] px = img.getRGB(x0, y0, cw, ch, null, 0, cw);
        // torchvision Resize(int): 짧은 변 = size, 긴 변 = int(size * 긴/짧)
        int nw;
        int nh;
        if (cw <= ch) {
            nw = size;
            nh = (int) ((long) size * ch / cw);
        } else {
            nh = size;
            nw = (int) ((long) size * cw / ch);
        }
        int[][] planes = new int[3][cw * ch];
        for (int i = 0; i < px.length; i++) {
            planes[0][i] = (px[i] >> 16) & 0xFF;
            planes[1][i] = (px[i] >> 8) & 0xFF;
            planes[2][i] = px[i] & 0xFF;
        }
        // torchvision CenterCrop: 위·왼쪽 = round((크기 - size) / 2)
        int top = (int) Math.round((nh - size) / 2.0);
        int left = (int) Math.round((nw - size) / 2.0);
        int plane = size * size;
        float[] chw = new float[3 * plane];
        Weights wx = weights(cw, nw);
        Weights wy = weights(ch, nh);
        for (int c = 0; c < 3; c++) {
            int[] resized = resize(planes[c], cw, ch, nw, nh, wx, wy);
            for (int r = 0; r < size; r++) {
                for (int col = 0; col < size; col++) {
                    chw[c * plane + r * size + col] = resized[(r + top) * nw + col + left] / 255f;
                }
            }
        }
        return chw;
    }

    /** 가로 → 세로, 각 단계 결과를 8비트로 반올림(PIL과 같음). */
    static int[] resize(int[] src, int w, int h, int nw, int nh, Weights wx, Weights wy) {
        int[] tmp = new int[nw * h];
        for (int r = 0; r < h; r++) {
            for (int o = 0; o < nw; o++) {
                double s = 0;
                double[] k = wx.k[o];
                int start = wx.start[o];
                for (int j = 0; j < k.length; j++) {
                    s += src[r * w + start + j] * k[j];
                }
                tmp[r * nw + o] = clip8(s);
            }
        }
        int[] dst = new int[nw * nh];
        for (int o = 0; o < nh; o++) {
            double[] k = wy.k[o];
            int start = wy.start[o];
            for (int col = 0; col < nw; col++) {
                double s = 0;
                for (int j = 0; j < k.length; j++) {
                    s += tmp[(start + j) * nw + col] * k[j];
                }
                dst[o * nw + col] = clip8(s);
            }
        }
        return dst;
    }

    record Weights(int[] start, double[][] k) {
    }

    /** PIL precompute_coeffs, BILINEAR(삼각형 필터, support 1). */
    static Weights weights(int in, int out) {
        double scale = (double) in / out;
        double filterScale = Math.max(1.0, scale);
        double support = filterScale;
        int[] start = new int[out];
        double[][] k = new double[out][];
        for (int o = 0; o < out; o++) {
            double center = (o + 0.5) * scale;
            int xmin = Math.max(0, (int) (center - support + 0.5));
            int xmax = Math.min(in, (int) (center + support + 0.5));
            double[] wt = new double[Math.max(1, xmax - xmin)];
            double sum = 0;
            for (int i = 0; i < wt.length; i++) {
                double t = Math.abs((i + xmin - center + 0.5) / filterScale);
                wt[i] = t < 1 ? 1 - t : 0;
                sum += wt[i];
            }
            if (sum != 0) {
                for (int i = 0; i < wt.length; i++) {
                    wt[i] /= sum;
                }
            }
            start[o] = Math.min(xmin, in - wt.length);
            k[o] = wt;
        }
        return new Weights(start, k);
    }

    private static int clip8(double v) {
        return (int) Math.max(0, Math.min(255, Math.round(v)));
    }

    private static int clamp(int v, int lo, int hi) {
        return Math.max(lo, Math.min(hi, v));
    }
}
