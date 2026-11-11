package com.fashionagent.clothesstate.color;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * 옷 색 이름 정하기(학습 없이 픽셀로). 약속 문서의 colorFamily·tone·colorName을 채웁니다.
 * <ul>
 * <li>laid: 사진 테두리 색을 바닥(배경)으로 보고, 가운데 80% 안에서 바닥과 색이 다른 픽셀만 옷으로 봅니다.</li>
 * <li>worn: 사람이 화면 가운데 서 있다고 보고 몸통 쪽(가로 35~65%, 세로 25~55%)을 봅니다. 자세 인식은 다음 단계.</li>
 * </ul>
 * 옷 종류 모델이 옷 박스를 찾으면 {@link #inBox}로 박스 안만 봅니다.
 * 옷 픽셀의 중간값 색을 HSV로 바꿔 이름을 붙입니다. 조명이 노랗거나 무늬가 많은 옷은 틀릴 수 있습니다.
 */
public final class GarmentColor {

    public record Result(String colorFamily, String colorName, String tone) {
    }

    private GarmentColor() {
    }

    /**
     * 옷 박스 없이 사진 전체에서 옷 색 하나. 옷 종류 모델이 없거나 옷을 못 찾았을 때 씁니다.
     *
     * <p>2026-11-09 리허설: 화면에 띄운 사진을 폰으로 비추면 사진 테두리가 모니터 테두리·책상이라 바닥(배경)을
     * 잘못 잡았고, 흰 배경까지 옷으로 섞여 색이 틀렸습니다. 그래서 앱이 옷을 화면 가운데로 맞춘다는 점을 써서
     * 가운데 영역만 보고(테두리 색은 믿지 않음), 색을 몇 무리로 나눠 가장 큰 무리의 색을 씁니다.
     * 입은 사진은 피부색도 뺍니다. 가운데 영역 안의 흰 배경 같은 큰 무채색 판은 옷보다 밝고 고르면 뺍니다.
     */
    public static Result of(BufferedImage img, String mode) {
        int rgb = rgbOf(img, mode);
        return rgb < 0 ? new Result("unknown", null, null) : name(rgb);
    }

    /** {@link #of}가 고른 옷 색(RGB). 옷 픽셀이 없으면 -1. 헷갈리는 색 판단({@link ColorDoubt})에 씁니다. */
    public static int rgbOf(BufferedImage img, String mode) {
        boolean worn = "worn".equals(mode);
        int[] center = worn ? region(img, 0.3, 0.22, 0.7, 0.62) : region(img, 0.28, 0.28, 0.72, 0.72);
        if (center.length == 0) {
            return -1;
        }
        int[] pixels = center;
        if (worn) {
            pixels = keep(pixels, p -> !skin(p), 0.3);
        }
        // 가운데 영역을 둘러싼 띠. 여기를 대부분 채운 색은 배경(흰 배경판, 스튜디오 회색 벽 등)으로 봅니다.
        int[] ring = ring(img, 0.12, 0.88, worn ? 0.3 : 0.28, worn ? 0.7 : 0.72, worn ? 0.22 : 0.28,
                worn ? 0.62 : 0.72);
        return dominant(pixels, ring);
    }

    /** 조건에 맞는 픽셀만. 남는 게 [minShare]보다 적으면 그대로 둠(옷이 바닥과 비슷한 색이거나 아주 작을 때). */
    static int[] keep(int[] rgb, java.util.function.IntPredicate test, double minShare) {
        int[] kept = Arrays.stream(rgb).filter(test).toArray();
        return kept.length < rgb.length * minShare ? rgb : kept;
    }

    /** 피부색 같은 픽셀(주황 쪽 색, 중간 채도, 밝음). */
    static boolean skin(int rgb) {
        float[] hsv = java.awt.Color.RGBtoHSB((rgb >> 16) & 0xFF, (rgb >> 8) & 0xFF, rgb & 0xFF, null);
        double h = hsv[0] * 360;
        return h >= 5 && h <= 45 && hsv[1] >= 0.18 && hsv[1] <= 0.6 && hsv[2] >= 0.45;
    }

    /**
     * 픽셀을 k-평균으로 세 무리로 나눠 가장 큰 무리의 중간값 색. 가장 큰 무리가 아주 밝은 무채색(흰 배경판)이고
     * 다른 무리가 충분히 크면 그다음 무리를 씁니다.
     */
    static int dominant(int[] rgb) {
        return dominant(rgb, new int[0]);
    }

    /** [ring]: 가운데를 둘러싼 띠의 픽셀. 띠의 절반 넘게 비슷한 색인 무리는 배경으로 보고 건너뜁니다. */
    static int dominant(int[] rgb, int[] ring) {
        if (rgb.length < 12) {
            return median(rgb);
        }
        int k = 3;
        double[][] c = new double[k][3];
        int[] byLight = luminanceOrder(rgb);
        for (int i = 0; i < k; i++) {
            int p = byLight[(int) ((i + 0.5) * rgb.length / k)];
            c[i] = new double[] {(p >> 16) & 0xFF, (p >> 8) & 0xFF, p & 0xFF};
        }
        int[] label = new int[rgb.length];
        for (int iter = 0; iter < 10; iter++) {
            double[][] sum = new double[k][3];
            int[] count = new int[k];
            for (int i = 0; i < rgb.length; i++) {
                int r = (rgb[i] >> 16) & 0xFF;
                int g = (rgb[i] >> 8) & 0xFF;
                int b = rgb[i] & 0xFF;
                int best = 0;
                double bestD = Double.MAX_VALUE;
                for (int j = 0; j < k; j++) {
                    double d = sq(r - c[j][0]) + sq(g - c[j][1]) + sq(b - c[j][2]);
                    if (d < bestD) {
                        bestD = d;
                        best = j;
                    }
                }
                label[i] = best;
                sum[best][0] += r;
                sum[best][1] += g;
                sum[best][2] += b;
                count[best]++;
            }
            for (int j = 0; j < k; j++) {
                if (count[j] > 0) {
                    c[j] = new double[] {sum[j][0] / count[j], sum[j][1] / count[j], sum[j][2] / count[j]};
                }
            }
        }
        List<List<Integer>> groups = new ArrayList<>();
        for (int j = 0; j < k; j++) {
            groups.add(new ArrayList<>());
        }
        for (int i = 0; i < rgb.length; i++) {
            groups.get(label[i]).add(rgb[i]);
        }
        groups.sort((a, b) -> b.size() - a.size());
        for (List<Integer> g : groups) {
            if (g.size() < rgb.length * 0.15) {
                break;
            }
            int color = median(g.stream().mapToInt(Integer::intValue).toArray());
            if (!backdrop(color, ring)) {
                return color;
            }
        }
        return median(groups.get(0).stream().mapToInt(Integer::intValue).toArray());
    }

    /** 배경으로 볼 색: 둘레 띠의 절반 넘게 이 색과 비슷함. */
    private static boolean backdrop(int color, int[] ring) {
        if (ring.length == 0) {
            return false;
        }
        long near = Arrays.stream(ring).filter(p -> distance(p, color) < 40).count();
        return near > ring.length * 0.5;
    }

    /** (x0,y0)-(x1,y1) 바깥 사각형에서 가운데 사각형(cx0,cy0)-(cx1,cy1)을 뺀 띠의 픽셀. */
    static int[] ring(BufferedImage img, double o0, double o1, double cx0, double cx1, double cy0, double cy1) {
        int w = img.getWidth();
        int h = img.getHeight();
        int step = Math.max(1, Math.max(w, h) / 100);
        List<Integer> out = new ArrayList<>();
        for (int y = (int) (h * o0); y < (int) (h * o1); y += step) {
            for (int x = (int) (w * o0); x < (int) (w * o1); x += step) {
                double fx = (double) x / w;
                double fy = (double) y / h;
                if (fx >= cx0 && fx < cx1 && fy >= cy0 && fy < cy1) {
                    continue;
                }
                out.add(img.getRGB(x, y) & 0xFFFFFF);
            }
        }
        return out.stream().mapToInt(Integer::intValue).toArray();
    }

    private static int[] luminanceOrder(int[] rgb) {
        return Arrays.stream(rgb).boxed()
                .sorted(java.util.Comparator.comparingInt(p -> ((p >> 16) & 0xFF) * 3 + ((p >> 8) & 0xFF) * 6 + (p & 0xFF)))
                .mapToInt(Integer::intValue).toArray();
    }

    private static double sq(double v) {
        return v * v;
    }

    /**
     * 옷 박스 하나를 잘라 낸 사진의 색. laid는 박스 가장자리(바닥이 조금 보임)를 배경으로 빼고,
     * worn은 박스 가운데(가로 25~75%, 세로 20~70%)만 봐서 팔 옆 배경·피부가 덜 섞이게 합니다.
     */
    public static Result inBox(BufferedImage crop, String mode) {
        int rgb = rgbInBox(crop, mode);
        return rgb < 0 ? new Result("unknown", null, null) : name(rgb);
    }

    /** {@link #inBox}가 고른 옷 색(RGB). 옷 픽셀이 없으면 -1. */
    public static int rgbInBox(BufferedImage crop, String mode) {
        int[] rgb = "worn".equals(mode) ? region(crop, 0.25, 0.2, 0.75, 0.7) : garmentPixels(crop);
        if (rgb.length == 0) {
            return -1;
        }
        return median(rgb);
    }

    /** 바닥과 다른 색 픽셀. 너무 적으면(옷이 바닥과 비슷한 색) 가운데 50%를 그대로 씀. */
    static int[] garmentPixels(BufferedImage img) {
        int[] border = border(img);
        int bg = median(border);
        int[] center = region(img, 0.1, 0.1, 0.9, 0.9);
        List<Integer> keep = new ArrayList<>();
        for (int p : center) {
            if (distance(p, bg) > 40) {
                keep.add(p);
            }
        }
        if (keep.size() < center.length * 0.05) {
            return region(img, 0.25, 0.25, 0.75, 0.75);
        }
        return keep.stream().mapToInt(Integer::intValue).toArray();
    }

    static int[] region(BufferedImage img, double x0, double y0, double x1, double y1) {
        int w = img.getWidth();
        int h = img.getHeight();
        int sx = (int) (w * x0);
        int ex = (int) (w * x1);
        int sy = (int) (h * y0);
        int ey = (int) (h * y1);
        int step = Math.max(1, Math.max(ex - sx, ey - sy) / 80);
        List<Integer> out = new ArrayList<>();
        for (int y = sy; y < ey; y += step) {
            for (int x = sx; x < ex; x += step) {
                out.add(img.getRGB(x, y) & 0xFFFFFF);
            }
        }
        return out.stream().mapToInt(Integer::intValue).toArray();
    }

    static int[] border(BufferedImage img) {
        int w = img.getWidth();
        int h = img.getHeight();
        int band = Math.max(1, Math.min(w, h) / 20);
        int step = Math.max(1, Math.max(w, h) / 100);
        List<Integer> out = new ArrayList<>();
        for (int x = 0; x < w; x += step) {
            for (int b = 0; b < band; b += Math.max(1, band / 3)) {
                out.add(img.getRGB(x, b) & 0xFFFFFF);
                out.add(img.getRGB(x, h - 1 - b) & 0xFFFFFF);
            }
        }
        for (int y = 0; y < h; y += step) {
            for (int b = 0; b < band; b += Math.max(1, band / 3)) {
                out.add(img.getRGB(b, y) & 0xFFFFFF);
                out.add(img.getRGB(w - 1 - b, y) & 0xFFFFFF);
            }
        }
        return out.stream().mapToInt(Integer::intValue).toArray();
    }

    /** 채널별 중간값. */
    static int median(int[] rgb) {
        int n = rgb.length;
        int[] r = new int[n];
        int[] g = new int[n];
        int[] b = new int[n];
        for (int i = 0; i < n; i++) {
            r[i] = (rgb[i] >> 16) & 0xFF;
            g[i] = (rgb[i] >> 8) & 0xFF;
            b[i] = rgb[i] & 0xFF;
        }
        Arrays.sort(r);
        Arrays.sort(g);
        Arrays.sort(b);
        return (r[n / 2] << 16) | (g[n / 2] << 8) | b[n / 2];
    }

    private static double distance(int a, int b) {
        int dr = ((a >> 16) & 0xFF) - ((b >> 16) & 0xFF);
        int dg = ((a >> 8) & 0xFF) - ((b >> 8) & 0xFF);
        int db = (a & 0xFF) - (b & 0xFF);
        return Math.sqrt(dr * dr + dg * dg + db * db);
    }

    /** RGB 하나 → colorFamily(약속 문서 값), 한국어 이름, tone. */
    public static Result name(int rgb) {
        float[] hsv = java.awt.Color.RGBtoHSB((rgb >> 16) & 0xFF, (rgb >> 8) & 0xFF, rgb & 0xFF, null);
        double h = hsv[0] * 360;
        double s = hsv[1];
        double v = hsv[2];
        String tone = v < 0.35 ? "dark" : v > 0.75 ? "light" : "mid";
        String family;
        // 아주 어두운 색은 화면·조명 탓에 파랗거나 초록빛이 끼어도 검정으로 봅니다(2026-11-09: 검정 코트를 파랑이라 함).
        if (v < 0.16 || (s < 0.25 && v < 0.22 && !(h >= 10 && h < 45 && s >= 0.15)) || (v < 0.2 && s < 0.6)) {
            family = "black";
        } else if (h >= 10 && h < 45 && s >= 0.15 && v < 0.45) {
            // 어두운 갈색은 채도가 낮아 회색으로 잡히기 쉬워서 회색보다 먼저 봅니다.
            family = "brown";
        } else if (s < 0.12 || (s < 0.2 && v < 0.5)) {
            family = v > 0.88 ? "white" : "gray";
        } else if (h >= 200 && h < 255 && v < 0.45) {
            family = "navy";
        } else if (h < 50 && s < 0.45 && v > 0.6) {
            family = "beige";
        } else if (h >= 10 && h < 45 && v <= 0.6) {
            family = "brown";
        } else if (h >= 45 && h < 75 && s < 0.6 && v < 0.7) {
            family = "khaki";
        } else if (h < 12 || h >= 340) {
            family = s < 0.5 && v > 0.7 ? "pink" : "red";
        } else if (h < 40) {
            family = "orange";
        } else if (h < 70) {
            family = "yellow";
        } else if (h < 170) {
            family = "green";
        } else if (h < 255) {
            family = "blue";
        } else if (h < 290) {
            family = "purple";
        } else {
            family = "pink";
        }
        return new Result(family, korean(family, tone), tone);
    }

    private static String korean(String family, String tone) {
        String base = switch (family) {
            case "black" -> "검정";
            case "white" -> "흰색";
            case "gray" -> "회색";
            case "navy" -> "네이비";
            case "beige" -> "베이지";
            case "brown" -> "갈색";
            case "khaki" -> "카키";
            case "red" -> "빨강";
            case "pink" -> "분홍";
            case "orange" -> "주황";
            case "yellow" -> "노랑";
            case "green" -> "초록";
            case "blue" -> "파랑";
            case "purple" -> "보라";
            default -> null;
        };
        if (base == null || family.equals("black") || family.equals("white") || family.equals("navy")) {
            return base;
        }
        if (family.equals("gray")) {
            return tone.equals("dark") ? "차콜 그레이" : tone.equals("light") ? "밝은 회색" : "회색";
        }
        return switch (tone) {
            case "dark" -> "어두운 " + base;
            case "light" -> "밝은 " + base;
            default -> base;
        };
    }
}
