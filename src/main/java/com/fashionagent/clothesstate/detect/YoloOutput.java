package com.fashionagent.clothesstate.detect;

import com.fashionagent.clothesstate.image.Images.Letterbox;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * YOLO26n ONNX 출력 풀기. 출력은 [1, 4+클래스 수, N] 또는 [1, N, 4+클래스 수]이고 한 줄이
 * [중심x, 중심y, 너비, 높이, 클래스별 확률...](모델 입력 픽셀)입니다. 512 입력이면 N = 5,376.
 * 얼룩 모델은 [얼룩, 찢어짐], 옷 종류 모델은 [상의, 하의, 아우터, 원피스, 신발]입니다.
 */
public final class YoloOutput {

    /** 학습 클래스 순서: 0 stain, 1 torn. 약속 문서 이름으로 바꿈(torn → tear). */
    static final String[] KINDS = {"stain", "tear"};

    private YoloOutput() {
    }

    public static List<Detection> parse(float[][] out, Letterbox lb, double minConfidence, double nmsIou) {
        return parse(out, lb, minConfidence, nmsIou, KINDS);
    }

    public static List<Detection> parse(float[][] out, Letterbox lb, double minConfidence, double nmsIou,
            String[] kinds) {
        return parse(out, lb, minConfidence, nmsIou, kinds, null);
    }

    /**
     * 비슷한 두 종류를 1등으로 고른 박스에서 다른 쪽 점수가 1등의 ratio배 이상이면 merged 종류로 내보낸다.
     * 옷 종류 모델의 코트·재킷이 이런 경우다(기장이 안 보이면 둘을 자주 바꿔 말함).
     */
    public record UnsurePair(String a, String b, double ratio, String merged) {
    }

    public static List<Detection> parse(float[][] out, Letterbox lb, double minConfidence, double nmsIou,
            String[] kinds, UnsurePair unsure) {
        int ua = unsure == null ? -1 : java.util.Arrays.asList(kinds).indexOf(unsure.a());
        int ub = unsure == null ? -1 : java.util.Arrays.asList(kinds).indexOf(unsure.b());
        boolean channelsFirst = out.length == 4 + kinds.length && out[0].length != 4 + kinds.length;
        int n = channelsFirst ? out[0].length : out.length;
        List<Detection> found = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            int best = -1;
            double score = 0;
            for (int c = 0; c < kinds.length; c++) {
                double s = channelsFirst ? out[4 + c][i] : out[i][4 + c];
                if (s > score) {
                    score = s;
                    best = c;
                }
            }
            if (best < 0 || score < minConfidence) {
                continue;
            }
            double cx = channelsFirst ? out[0][i] : out[i][0];
            double cy = channelsFirst ? out[1][i] : out[i][1];
            double bw = channelsFirst ? out[2][i] : out[i][2];
            double bh = channelsFirst ? out[3][i] : out[i][3];
            double x1 = lb.toRatioX(cx - bw / 2);
            double y1 = lb.toRatioY(cy - bh / 2);
            double x2 = lb.toRatioX(cx + bw / 2);
            double y2 = lb.toRatioY(cy + bh / 2);
            if (x2 <= x1 || y2 <= y1) {
                continue;
            }
            String kind = kinds[best];
            if (ua >= 0 && ub >= 0 && (best == ua || best == ub)) {
                int other = best == ua ? ub : ua;
                double so = channelsFirst ? out[4 + other][i] : out[i][4 + other];
                if (so >= unsure.ratio() * score) {
                    kind = unsure.merged();
                }
            }
            found.add(new Detection(kind, score, x1, y1, x2 - x1, y2 - y1));
        }
        return nms(found, nmsIou);
    }

    /** 같은 종류끼리 많이 겹치면 확신도 높은 것 하나만 남긴다. */
    static List<Detection> nms(List<Detection> in, double iou) {
        List<Detection> sorted = new ArrayList<>(in);
        sorted.sort(Comparator.comparingDouble(Detection::confidence).reversed());
        List<Detection> kept = new ArrayList<>();
        for (Detection d : sorted) {
            boolean overlaps = kept.stream().anyMatch(k -> k.kind().equals(d.kind()) && k.iou(d) > iou);
            if (!overlaps) {
                kept.add(d);
            }
        }
        return kept;
    }
}
