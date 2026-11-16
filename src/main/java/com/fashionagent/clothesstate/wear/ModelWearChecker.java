package com.fashionagent.clothesstate.wear;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.springframework.core.io.ResourceLoader;
import org.springframework.stereotype.Component;

/**
 * 우리 모델로 입은 옷 한 벌의 실수를 봅니다(사진은 서버 밖으로 안 나감). 학습 때 자른 방식과 같게 자릅니다.
 * <ul>
 * <li>택: 상의·겉옷·원피스 박스의 가로 가운데 60%, 세로 위 -10%~+30%(목 부분). 뒷모습이면 tag_out, 앞모습이면 backwards
 * (앞뒤 바꿔 입으면 뒷목 택이 앞 목에 보임).</li>
 * <li>단추: 셔츠·가디건 박스(사방 5%), 앞모습만.</li>
 * <li>뒤집어 입음: 셔츠·티셔츠·니트·원피스 박스(사방 5%).</li>
 * </ul>
 */
@Component
public class ModelWearChecker implements AutoCloseable {

    /** 학습 폴더 순서: tag=[none, tag], button=[ok, shifted], inside=[inside_out, ok]. */
    private final OnnxBinaryClassifier tag;
    private final OnnxBinaryClassifier button;
    private final OnnxBinaryClassifier inside;
    private final WearModelProperties props;

    private static final Set<String> NECK = Set.of("top", "outer", "dress");
    /** 상의(종류를 모르는 윗옷): 2026-11-16 가까이 찍은 셔츠를 옷 찾기가 신발로 봐서 상의로 바꿔 줄 때도 단추를 봄. */
    private static final Set<String> BUTTONED = Set.of("셔츠", "가디건", "상의");
    private static final Set<String> INSIDE = Set.of("셔츠", "티셔츠", "니트", "원피스", "상의");

    @org.springframework.beans.factory.annotation.Autowired
    public ModelWearChecker(WearModelProperties props, ResourceLoader loader) {
        this(props, props.enabled() ? new OnnxBinaryClassifier(loader.getResource(props.tagModel()), 224, 1, "택") : null,
                props.enabled() ? new OnnxBinaryClassifier(loader.getResource(props.buttonModel()), 320, 1, "단추") : null,
                props.enabled() ? new OnnxBinaryClassifier(loader.getResource(props.insideModel()), 320, 0, "뒤집힘") : null);
    }

    /** 시험용(분류기를 직접 넣음). */
    public ModelWearChecker(WearModelProperties props, OnnxBinaryClassifier tag, OnnxBinaryClassifier button,
            OnnxBinaryClassifier inside) {
        this.props = props;
        this.tag = tag;
        this.button = button;
        this.inside = inside;
    }

    public boolean ready() {
        return props.enabled() && (ok(tag) || ok(button) || ok(inside));
    }

    public String version() {
        return props.version();
    }

    private static boolean ok(OnnxBinaryClassifier c) {
        return c != null && c.ready();
    }

    /**
     * @param box 옷 박스 {x, y, w, h}(사진 비율)
     * @param category top·outer·dress·bottom…
     * @param name 우리 옷 이름(셔츠·티셔츠…)
     * @return 기준을 넘은 실수(확신도는 모델 확률)
     */
    public List<WearChecker.Issue> check(BufferedImage img, double[] box, String category, String name, String view) {
        return issues(probabilities(img, box, category, name, view));
    }

    /** probabilities 결과 중 기준을 넘은 것만 실수로(택은 tagProbability가 이미 기준을 적용해 0보다 크면). */
    public List<WearChecker.Issue> issues(Map<String, Double> probs) {
        List<WearChecker.Issue> out = new ArrayList<>();
        probs.forEach((kind, p) -> {
            double t = switch (kind) {
                case "buttons_misaligned" -> props.buttonThreshold();
                case "inside_out" -> props.insideThreshold();
                default -> Double.MIN_VALUE; // tag_out·backwards
            };
            if (p >= t && p > 0) {
                out.add(new WearChecker.Issue(kind, p, props.version()));
            }
        });
        return out;
    }

    /**
     * 이 옷에 볼 수 있는 실수마다 모델 확률(기준 적용 전, 택만 창 찾기 기준 적용). 볼 수 없는 실수는 넣지 않음.
     * GPT 답과 견줘 "모델을 믿어도 되는지"(WearTrust)를 재는 데도 씀.
     */
    public Map<String, Double> probabilities(BufferedImage img, double[] box, String category, String name, String view) {
        Map<String, Double> out = new LinkedHashMap<>();
        double x = box[0], y = box[1], w = box[2], h = box[3];
        boolean back = "back".equals(view);
        if (ok(tag) && NECK.contains(category)) {
            out.put(back ? "tag_out" : "backwards", tagProbability(img, x, y, w, h));
        }
        if (ok(button) && !back && BUTTONED.contains(name)) {
            out.put("buttons_misaligned", button.probabilitySquare(img, x, y, w, h, 0.05));
        }
        if (ok(inside) && INSIDE.contains(name)) {
            out.put("inside_out", inside.probabilitySquare(img, x, y, w, h, 0.05));
        }
        return out;
    }

    /**
     * 택 확률. 가운데 창(옷 박스 가로 가운데 60%, 세로 -10%~+30%, 학습과 같음)이 기준(0.8)을 넘으면 그 값. 아니면 옆으로 옮긴 창 4개와
     * 작은 창 3개를 더 보고 더 높은 기준(0.95)을 넘은 가장 큰 값. 둘 다 아니면 0.
     */
    double tagProbability(BufferedImage img, double x, double y, double w, double h) {
        double p = tagWindow(img, x + w * 0.2, y - h * 0.1, w * 0.6, h * 0.4);
        if (p >= props.tagThreshold()) {
            return p;
        }
        double best = 0;
        for (double dx : new double[] {-0.2, -0.1, 0.1, 0.2}) {
            best = Math.max(best, tagWindow(img, x + w * (0.2 + dx), y - h * 0.1, w * 0.6, h * 0.4));
        }
        for (double cx : new double[] {0.3, 0.5, 0.7}) {
            best = Math.max(best, tagWindow(img, x + w * (cx - 0.15), y - h * 0.05, w * 0.3, h * 0.25));
        }
        return best >= props.tagScanThreshold() ? best : 0;
    }

    /** 사진 밖으로 나간 부분은 잘라 냄(학습 스크립트 neck_crop과 같음). 너무 작으면 0. */
    private double tagWindow(BufferedImage img, double x, double y, double w, double h) {
        double x0 = Math.max(0, x), y0 = Math.max(0, y);
        double x1 = Math.min(1, x + w), y1 = Math.min(1, y + h);
        if ((x1 - x0) * img.getWidth() < 16 || (y1 - y0) * img.getHeight() < 16) {
            return 0;
        }
        return tag.probability(img, x0, y0, x1 - x0, y1 - y0, 0);
    }

    @Override
    public void close() throws Exception {
        for (OnnxBinaryClassifier c : new OnnxBinaryClassifier[] {tag, button, inside}) {
            if (c != null) {
                c.close();
            }
        }
    }
}
