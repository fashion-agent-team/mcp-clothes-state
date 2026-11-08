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

    public static Result of(BufferedImage img, String mode) {
        int[] rgb = "worn".equals(mode) ? region(img, 0.35, 0.25, 0.65, 0.55) : garmentPixels(img);
        if (rgb.length == 0) {
            return new Result("unknown", null, null);
        }
        return name(median(rgb));
    }

    /**
     * 옷 박스 하나를 잘라 낸 사진의 색. laid는 박스 가장자리(바닥이 조금 보임)를 배경으로 빼고,
     * worn은 박스 가운데(가로 25~75%, 세로 20~70%)만 봐서 팔 옆 배경·피부가 덜 섞이게 합니다.
     */
    public static Result inBox(BufferedImage crop, String mode) {
        int[] rgb = "worn".equals(mode) ? region(crop, 0.25, 0.2, 0.75, 0.7) : garmentPixels(crop);
        if (rgb.length == 0) {
            return new Result("unknown", null, null);
        }
        return name(median(rgb));
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
    static Result name(int rgb) {
        float[] hsv = java.awt.Color.RGBtoHSB((rgb >> 16) & 0xFF, (rgb >> 8) & 0xFF, rgb & 0xFF, null);
        double h = hsv[0] * 360;
        double s = hsv[1];
        double v = hsv[2];
        String tone = v < 0.35 ? "dark" : v > 0.75 ? "light" : "mid";
        String family;
        if (v < 0.16 || (s < 0.25 && v < 0.22)) {
            family = "black";
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
