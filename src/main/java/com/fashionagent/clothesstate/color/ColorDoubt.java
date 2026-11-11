package com.fashionagent.clothesstate.color;

/**
 * 사진만으로는 헷갈리기 쉬운 색 짝(검정↔네이비, 흰색↔베이지)일 때 "다른 후보" 이름을 줍니다. agent는 이 값이 있으면
 * "검정이나 네이비 같아요"처럼 단정하지 않고 말합니다. 색을 고쳐 쓰지는 않습니다(colorName은 그대로).
 *
 * <p>2026-11-11 결정: 노란 불 아래 사진은 색을 강제로 보정하지 않고(베이지 옷까지 흰색으로 바뀔 수 있어서),
 * lighting=warm으로 알리고 헷갈리는 짝만 이렇게 표시합니다.
 * <ul>
 * <li>흰 불(warm=false): 경계 근처만. 파란 기가 도는 검정 → 네이비, 아주 어두운 네이비 → 검정, 노란 기가 도는 흰색 → 베이지,
 * 채도가 아주 낮은 베이지 → 흰색.</li>
 * <li>노란 불(warm=true): 노란 불은 파랑을 어둡게, 흰색을 노랗게 만드므로 검정·네이비·흰색·베이지는 모두 짝을 붙입니다.</li>
 * </ul>
 */
public final class ColorDoubt {

    private ColorDoubt() {
    }

    /** 다른 후보 색 이름. 헷갈리지 않으면 null. rgb가 음수(색을 못 고름)면 null. */
    public static String alternative(int rgb, GarmentColor.Result color, boolean warm) {
        if (rgb < 0 || color == null || color.colorFamily() == null) {
            return null;
        }
        float[] hsv = java.awt.Color.RGBtoHSB((rgb >> 16) & 0xFF, (rgb >> 8) & 0xFF, rgb & 0xFF, null);
        double h = hsv[0] * 360;
        double s = hsv[1];
        double v = hsv[2];
        return switch (color.colorFamily()) {
            case "black" -> warm || (h >= 190 && h < 260 && s >= 0.2) ? "네이비" : null;
            case "navy" -> warm || v < 0.25 ? "검정" : null;
            case "white" -> warm || (h >= 20 && h < 60 && s >= 0.06) ? "베이지" : null;
            case "beige" -> warm || s < 0.2 ? "흰색" : null;
            default -> null;
        };
    }
}
