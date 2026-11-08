package com.fashionagent.clothesstate.inspect;

import com.fashionagent.clothesstate.ClothesStateException;
import com.fashionagent.clothesstate.ClothesStateProperties;
import com.fashionagent.clothesstate.color.GarmentColor;
import com.fashionagent.clothesstate.detect.DefectDetector;
import com.fashionagent.clothesstate.detect.Detection;
import com.fashionagent.clothesstate.image.Images;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import org.springframework.stereotype.Service;

/**
 * clothes_inspect: 옷 사진 한 장 → 얼룩·찢어짐. 우리가 학습한 모델(ONNX)을 이 서버에서 돌리고, 사진은 외부 AI로 보내지 않고
 * 저장하지도 않습니다(2026-11-06 결정). 확신도 tell 이상은 defects(바로 알림), ask~tell은 pendingHelper(지인 확인 후보),
 * 그 아래는 버립니다(기준 0.5 / 0.3, 정지원 2026-11-08).
 */
@Service
public class ClothesInspectService {

    static final String GARMENT_ID = "g1";
    private static final Set<String> MODES = Set.of("laid", "worn");
    private static final Set<String> VIEWS = Set.of("front", "side", "back");

    private final ClothesStateProperties props;
    private final DefectDetector detector;

    public ClothesInspectService(ClothesStateProperties props, DefectDetector detector) {
        this.props = props;
        this.detector = detector;
    }

    public InspectResult inspect(String imageBase64, String mode, String view) {
        String m = mode == null || mode.isBlank() ? "laid" : mode.strip();
        if (!MODES.contains(m)) {
            throw new ClothesStateException("bad_request", "mode는 laid 또는 worn이어야 해요.");
        }
        String v = view == null || view.isBlank() || m.equals("laid") ? null : view.strip();
        if (v != null && !VIEWS.contains(v)) {
            throw new ClothesStateException("bad_request", "view는 front, side, back 중 하나여야 해요.");
        }
        if (!detector.ready()) {
            throw new ClothesStateException("not_ready", "옷 상태 확인은 아직 준비 중이에요.");
        }
        BufferedImage img = Images.decode(imageBase64, props.maxImageBytes().toBytes());
        if (Images.meanLightness(img) < props.darkLightness()) {
            return new InspectResult(List.of(), null, "dark", "model", props.modelVersion(), null);
        }

        List<InspectResult.Defect> defects = new ArrayList<>();
        List<InspectResult.Candidate> candidates = new ArrayList<>();
        for (Detection d : detector.detect(img, props.askThreshold())) {
            String where = Where.of(d, m, v);
            double conf = round(d.confidence());
            if (d.confidence() >= props.tellThreshold()) {
                defects.add(new InspectResult.Defect(d.kind(), where, conf));
            } else if (candidates.size() < props.maxCandidates()) {
                candidates.add(new InspectResult.Candidate(d.kind(), where, conf,
                        new InspectResult.Box(round(d.x()), round(d.y()), round(d.w()), round(d.h()))));
            }
        }
        GarmentColor.Result color = GarmentColor.of(img, m);
        var garment = new InspectResult.Garment(GARMENT_ID, "other", "옷", color.colorName(), color.colorFamily(),
                color.tone(), null, null, null,
                "unknown", defects, new InspectResult.Box(0, 0, 1, 1), m.equals("laid") ? "center" : null, null);
        var pending = candidates.isEmpty() ? null
                : new InspectResult.PendingHelper(GARMENT_ID, candidates, "확실하지 않은 부분이 있어요.");
        return new InspectResult(List.of(garment), null, "ok", "model", props.modelVersion(), pending);
    }

    private static double round(double v) {
        return Math.round(v * 100) / 100.0;
    }
}
