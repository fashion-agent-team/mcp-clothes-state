package com.fashionagent.clothesstate.inspect;

import com.fashionagent.clothesstate.detect.Detection;

/**
 * 결함 위치 말(where). 사진을 가로·세로 셋으로 나눠 박스 중심이 어디에 있는지로 정합니다.
 * <ul>
 * <li>laid(내려놓은 옷): 사진 기준. 사진 왼쪽 = 사용자 왼쪽(약속 문서).</li>
 * <li>worn(입은 사진, 앱이 뒤집지 않고 보냄): 입은 사람 몸 기준. 앞모습이면 사진 오른쪽이 그 사람의 왼쪽이고,
 * 뒷모습이면 사진 왼쪽이 왼쪽입니다. 옆모습은 왼쪽·오른쪽을 말하지 않습니다.
 * 지금은 사람이 화면 가운데 바로 서 있다고 보고 정하며, 자세 인식(pose)으로 팔·몸통을 찾는 건 다음 단계입니다.</li>
 * </ul>
 */
public final class Where {

    private Where() {
    }

    public static String of(Detection d, String mode, String view) {
        String h = horizontal(d.centerX());
        String v = vertical(d.centerY());
        if (!"worn".equals(mode)) {
            return join("옷", h, v);
        }
        String body = switch (view == null ? "front" : view) {
            case "back" -> h;
            case "side" -> "가운데";
            default -> mirror(h);
        };
        String prefix = "back".equals(view) ? "등" : "side".equals(view) ? "옆" : "몸";
        return join(prefix, body, v);
    }

    static String horizontal(double cx) {
        return cx < 1.0 / 3 ? "왼쪽" : cx > 2.0 / 3 ? "오른쪽" : "가운데";
    }

    static String vertical(double cy) {
        return cy < 1.0 / 3 ? "위쪽" : cy > 2.0 / 3 ? "아래쪽" : "가운데";
    }

    private static String mirror(String h) {
        return switch (h) {
            case "왼쪽" -> "오른쪽";
            case "오른쪽" -> "왼쪽";
            default -> h;
        };
    }

    private static String join(String prefix, String h, String v) {
        if (h.equals("가운데") && v.equals("가운데")) {
            return prefix + " 가운데";
        }
        if (h.equals("가운데")) {
            return prefix + " " + v;
        }
        if (v.equals("가운데")) {
            return prefix + " " + h;
        }
        return prefix + " " + h + " " + v;
    }
}
