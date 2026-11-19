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
        boolean worn = "worn".equals(mode);
        int[] center = worn ? region(img, 0.3, 0.22, 0.7, 0.62) : region(img, 0.28, 0.28, 0.72, 0.72);
        if (center.length == 0) {
            return new Result("unknown", null, null);
        }
        int[] pixels = center;
        if (worn) {
            pixels = keep(pixels, p -> !skin(p), 0.3);
        }
        // 가운데 영역을 둘러싼 띠. 여기를 대부분 채운 색은 배경(흰 배경판, 스튜디오 회색 벽 등)으로 봅니다.
        int[] ring = ring(img, 0.12, 0.88, worn ? 0.3 : 0.28, worn ? 0.7 : 0.72, worn ? 0.22 : 0.28,
                worn ? 0.62 : 0.72);
        return name(dominant(pixels, ring));
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

    /**
     * 무늬 옷의 두 번째 색. 옷 픽셀(laid는 박스 가장자리 바닥색을 뺀 픽셀, worn은 박스 가운데)을 k-평균으로 네 무리로 나눠,
     * 옷 픽셀의 12% 이상인 무리 가운데 [main]과 다른 계열이고 대표색에서 가장 먼 무리의 색을 돌려줍니다. 없으면 null.
     *
     * <p>2026-11-12: 네이비·베이지 체크 셔츠를 "검정 셔츠", 회색 바탕 검정 호피 셔츠를 "검정 셔츠"라고 해서 넣었습니다.
     * 색 인식 모델은 옷의 대표색 하나만 배워서(K-Fashion 색상 라벨), 섞인 색은 픽셀로 봅니다.
     * 검정·네이비·차콜처럼 둘 다 어두운 색은 듣는 사람에게 같은 색처럼 들려서 다른 색으로 치지 않습니다.
     */
    public static Result secondColor(BufferedImage crop, String mode, Result main) {
        // 박스 가운데(가로 25~75%, 세로 20~80%)만 봅니다. 무늬는 옷 전체에 퍼져 있지만 팔·목 피부, 손, 배경은 박스 가장자리에 있어서요.
        // 2026-11-12: 흰 티를 입은 모델 사진에서 피부를 "갈색"으로 봄. 피부색 필터는 베이지 체크 선까지 지워서 쓰지 않습니다.
        int[] rgb = "worn".equals(mode) ? region(crop, 0.25, 0.2, 0.75, 0.7) : centerGarmentPixels(crop);
        if (rgb.length < 40 || main == null) {
            return null;
        }
        List<int[]> groups = clusters(rgb, 4);
        // 대표색에 해당하는 무리(없으면 가장 큰 무리)에서 가장 멀리 떨어진 색을 고릅니다. 모니터·그늘에서 찍으면 무늬 경계가
        // 섞인 중간색 무리(회갈색 등)가 생기는데, 그보다 실제 무늬 색(체크의 베이지 선)이 대표색과 더 멀어서요.
        int anchor = groups.get(0)[0];
        for (int[] g : groups) {
            if (sameToHear(main, name(g[0]))) {
                anchor = g[0];
                break;
            }
        }
        Result best = null;
        double bestD = -1;
        for (int[] g : groups) {
            if (g[1] < rgb.length * 0.12) {
                continue;
            }
            Result r = name(g[0]);
            double d = distance(g[0], anchor);
            if (r.colorName() != null && !sameToHear(main, r) && d > bestD) {
                best = r;
                bestD = d;
            }
        }
        return best;
    }

    /**
     * 박스 가운데의 옷 픽셀. 박스 테두리의 60% 이상이 한 색(바닥)일 때만 그 색과 먼 픽셀을 남깁니다(바지 다리 사이 바닥 등).
     * 테두리가 옷 무늬(줄무늬 두 색 반반)면 바닥이 아니라서 빼지 않습니다. 너무 적게 남으면 가운데 그대로.
     */
    static int[] centerGarmentPixels(BufferedImage crop) {
        int[] ring = border(crop);
        int bg = median(ring);
        int[] center = region(crop, 0.25, 0.2, 0.75, 0.8);
        long floorLike = Arrays.stream(ring).filter(p -> distance(p, bg) <= 40).count();
        if (floorLike < ring.length * 0.6) {
            return center;
        }
        int[] kept = Arrays.stream(center).filter(p -> distance(p, bg) > 40).toArray();
        return kept.length < center.length * 0.05 ? center : kept;
    }

    /** 귀로 들었을 때 같은 색: 같은 계열이거나, 둘 다 어두운 무채색·네이비(검정·네이비·차콜·어두운 갈색)이거나, 흰색과 밝은 회색. */
    static boolean sameToHear(Result a, Result b) {
        if (a.colorFamily().equals(b.colorFamily())) {
            return true;
        }
        if (darkNeutral(a) && darkNeutral(b)) {
            return true;
        }
        java.util.Set<String> pale = java.util.Set.of("white", "gray");
        return pale.contains(a.colorFamily()) && pale.contains(b.colorFamily())
                && "light".equals(a.tone()) && "light".equals(b.tone());
    }

    private static boolean darkNeutral(Result r) {
        return switch (r.colorFamily()) {
            case "black", "navy" -> true;
            case "gray", "brown" -> "dark".equals(r.tone());
            default -> false;
        };
    }

    /** 픽셀을 k-평균으로 [k]무리로 나눠 {무리 중간값 색, 픽셀 수}를 큰 무리부터. */
    static List<int[]> clusters(int[] rgb, int k) {
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
        List<int[]> out = new ArrayList<>();
        for (int j = 0; j < k; j++) {
            final int jj = j;
            int[] members = java.util.stream.IntStream.range(0, rgb.length).filter(i -> label[i] == jj).map(i -> rgb[i]).toArray();
            if (members.length > 0) {
                out.add(new int[] {median(members), members.length});
            }
        }
        out.sort((a, b) -> b[1] - a[1]);
        return out;
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
        int rgb = boxRgb(crop, mode);
        return rgb < 0 ? new Result("unknown", null, null) : name(rgb);
    }

    /** {@link #inBox}가 보는 옷 픽셀의 중간값 색. 픽셀이 없으면 -1. */
    public static int boxRgb(BufferedImage crop, String mode) {
        int[] rgb = "worn".equals(mode) ? region(crop, 0.25, 0.2, 0.75, 0.7) : garmentPixels(crop);
        return rgb.length == 0 ? -1 : median(rgb);
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
        // 연한 색(채도 낮고 밝음): 하늘색 셔츠·연분홍처럼 격식 자리에도 무난한 색. tone은 약속 값(light|mid|dark)을 지키려고
        // light로 두고, 이름으로 구분합니다("연한 ~", 하늘색, 연분홍). 색 조합 규칙(ColorHarmony)이 이 이름을 봅니다.
        boolean pale = CHROMATIC.contains(family) && s < 0.4 && v > 0.7;
        if (pale) {
            return new Result(family, paleName(family), "light");
        }
        return new Result(family, korean(family, tone), tone);
    }

    /**
     * 색 인식 모델({@link ColorModel})의 답으로 색 이름을 정합니다. 밝기(tone)가 정해진 색(흰색·검정·네이비·하늘색·와인 등)은
     * 그 값을 쓰고, 나머지는 옷 픽셀 중간값 [pixelRgb]({@link #boxRgb})의 밝기·채도로 정합니다.
     *
     * <p>2026-11-11 AI Hub K-Fashion 시험 옷 25,626벌: 픽셀 규칙 45% → 모델(docs/color-rules.md "색 인식 모델").
     */
    public static Result fromModel(float[] probabilities, int pixelRgb) {
        int best = 0;
        for (int i = 1; i < probabilities.length; i++) {
            if (probabilities[i] > probabilities[best]) {
                best = i;
            }
        }
        float[] hsv = pixelRgb < 0 ? new float[] {0, 0, 0.5f}
                : java.awt.Color.RGBtoHSB((pixelRgb >> 16) & 0xFF, (pixelRgb >> 8) & 0xFF, pixelRgb & 0xFF, null);
        double s = hsv[1];
        double v = hsv[2];
        String tone = v < 0.35 ? "dark" : v > 0.75 ? "light" : "mid";
        // 강한 색은 연한 색(채도 낮고 밝음, 픽셀 규칙의 연한 색 기준과 같음)일 때만 light, 어두우면 dark, 나머지는 mid.
        // 쨍한 빨강을 "밝은 빨강"이라 하지 않고, 연분홍 블라우스를 강한 분홍으로 보지 않게.
        String vivid = v < 0.35 ? "dark" : s < 0.4 && v > 0.7 ? "light" : "mid";
        return switch (ColorModel.CLASSES[best]) {
            case "화이트" -> new Result("white", "흰색", "light");
            case "블랙" -> new Result("black", "검정", "dark");
            case "베이지" -> new Result("beige", "베이지", "light");
            case "핑크" -> vivid("pink", vivid);
            case "그레이" -> new Result("gray", korean("gray", tone), tone);
            case "스카이블루" -> new Result("blue", "하늘색", "light");
            case "네이비" -> new Result("navy", "네이비", "dark");
            case "브라운" -> new Result("brown", korean("brown", tone), tone);
            case "블루" -> vivid("blue", vivid);
            case "카키" -> new Result("khaki", korean("khaki", tone), tone);
            case "레드" -> vivid("red", vivid);
            case "그린" -> vivid("green", vivid);
            case "라벤더" -> new Result("purple", "연보라", "light");
            case "실버" -> new Result("gray", "은색", "light");
            case "옐로우" -> vivid("yellow", vivid);
            case "와인" -> new Result("red", "와인색", "dark");
            case "민트" -> new Result("green", "민트", "light");
            case "퍼플" -> vivid("purple", vivid);
            case "오렌지" -> vivid("orange", vivid);
            default -> new Result("yellow", "겨자색", vivid); // 골드(K-Fashion 골드 라벨은 대부분 겨자색 옷)
        };
    }

    /** 강한 색 이름. 연한 색(light)이면 픽셀 규칙과 같은 연한 색 이름(하늘색·연분홍·연한 ~)이라 색 조합에서 차분한 색으로 봅니다. */
    private static Result vivid(String family, String tone) {
        return "light".equals(tone) ? new Result(family, paleName(family), "light")
                : new Result(family, korean(family, tone), tone);
    }

    private static final java.util.Set<String> CHROMATIC =
            java.util.Set.of("red", "pink", "orange", "yellow", "green", "blue", "purple");

    private static String paleName(String family) {
        return switch (family) {
            case "blue" -> "하늘색";
            case "pink", "red" -> "연분홍";
            case "green" -> "연한 초록";
            case "yellow" -> "연한 노랑";
            case "orange" -> "연한 주황";
            default -> "연한 보라";
        };
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
