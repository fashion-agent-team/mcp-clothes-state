package com.fashionagent.clothesstate.wear;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;
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
    private static final Set<String> BUTTONED = Set.of("셔츠", "가디건");
    private static final Set<String> INSIDE = Set.of("셔츠", "티셔츠", "니트", "원피스", "상의");

    @org.springframework.beans.factory.annotation.Autowired
    public ModelWearChecker(WearModelProperties props, ResourceLoader loader) {
        this(props, props.enabled() ? new OnnxBinaryClassifier(loader.getResource(props.tagModel()), 224, 1, "택") : null,
                props.enabled() ? new OnnxBinaryClassifier(loader.getResource(props.buttonModel()), 320, 1, "단추") : null,
                props.enabled() ? new OnnxBinaryClassifier(loader.getResource(props.insideModel()), 320, 0, "뒤집힘") : null);
    }

    ModelWearChecker(WearModelProperties props, OnnxBinaryClassifier tag, OnnxBinaryClassifier button,
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
        List<WearChecker.Issue> out = new ArrayList<>();
        double x = box[0], y = box[1], w = box[2], h = box[3];
        boolean back = "back".equals(view);
        if (ok(tag) && NECK.contains(category)) {
            double ny = Math.max(0, y - h * 0.1);
            double p = tag.probability(img, x + w * 0.2, ny, w * 0.6, Math.min(1 - ny, y + h * 0.3 - ny), 0);
            if (p >= props.tagThreshold()) {
                out.add(new WearChecker.Issue(back ? "tag_out" : "backwards", p, props.version()));
            }
        }
        if (ok(button) && !back && BUTTONED.contains(name)) {
            double p = button.probability(img, x, y, w, h, 0.05);
            if (p >= props.buttonThreshold()) {
                out.add(new WearChecker.Issue("buttons_misaligned", p, props.version()));
            }
        }
        if (ok(inside) && INSIDE.contains(name)) {
            double p = inside.probability(img, x, y, w, h, 0.05);
            if (p >= props.insideThreshold()) {
                out.add(new WearChecker.Issue("inside_out", p, props.version()));
            }
        }
        return out;
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
