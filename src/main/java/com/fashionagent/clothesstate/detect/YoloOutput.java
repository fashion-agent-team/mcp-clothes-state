package com.fashionagent.clothesstate.detect;

import com.fashionagent.clothesstate.image.Images.Letterbox;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * YOLO26n ONNX 출력 풀기. 출력은 [1, 6, N] 또는 [1, N, 6]이고 한 줄이
 * [중심x, 중심y, 너비, 높이, 얼룩 확률, 찢어짐 확률](모델 입력 픽셀)입니다. 512 입력이면 N = 5,376.
 */
public final class YoloOutput {

    /** 학습 클래스 순서: 0 stain, 1 torn. 약속 문서 이름으로 바꿈(torn → tear). */
    static final String[] KINDS = {"stain", "tear"};

    private YoloOutput() {
    }

    public static List<Detection> parse(float[][] out, Letterbox lb, double minConfidence, double nmsIou) {
        boolean channelsFirst = out.length == 4 + KINDS.length && out[0].length != 4 + KINDS.length;
        int n = channelsFirst ? out[0].length : out.length;
        List<Detection> found = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            int best = -1;
            double score = 0;
            for (int c = 0; c < KINDS.length; c++) {
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
            found.add(new Detection(KINDS[best], score, x1, y1, x2 - x1, y2 - y1));
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
