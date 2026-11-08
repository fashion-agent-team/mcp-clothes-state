package com.fashionagent.clothesstate.inspect;

import com.fashionagent.clothesstate.ClothesStateException;
import com.fashionagent.clothesstate.ClothesStateProperties;
import com.fashionagent.clothesstate.color.GarmentColor;
import com.fashionagent.clothesstate.detect.DefectDetector;
import com.fashionagent.clothesstate.detect.Detection;
import com.fashionagent.clothesstate.detect.GarmentDetector;
import com.fashionagent.clothesstate.image.Images;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.springframework.stereotype.Service;

/**
 * clothes_inspect: 옷 사진 한 장 → 옷 한 벌마다 종류·색·얼룩·찢어짐. 우리가 학습한 모델(ONNX) 두 개를 이 서버에서 돌리고,
 * 사진은 외부 AI로 보내지 않고 저장하지도 않습니다(2026-11-06 결정).
 * <ul>
 * <li>옷 종류 모델이 옷 박스를 찾습니다. 못 찾거나 모델이 없으면 사진 전체를 옷 한 벌(category other)로 봅니다.</li>
 * <li>색은 옷 박스 안 픽셀로 정합니다(GarmentColor).</li>
 * <li>얼룩·찢어짐은 박스 중심이 들어가는 옷(여럿이면 가장 작은 옷)에 붙이고, where는 그 옷 박스 안 위치로 씁니다.</li>
 * <li>확신도 tell 이상은 defects(바로 알림), ask~tell은 pendingHelper(지인 확인 후보), 그 아래는 버립니다
 * (기준 0.5 / 0.3, 정지원 2026-11-08). pendingHelper는 한 벌에 한 번만 묻는 약속이라, 가장 확신도 높은 후보가
 * 있는 옷 한 벌의 후보만 담습니다.</li>
 * </ul>
 */
@Service
public class ClothesInspectService {

    private static final Set<String> MODES = Set.of("laid", "worn");
    private static final Set<String> VIEWS = Set.of("front", "side", "back");
    private static final InspectResult.Box WHOLE = new InspectResult.Box(0, 0, 1, 1);
    private static final Map<String, String> NAMES = Map.of(
            "top", "상의", "bottom", "하의", "outer", "아우터", "dress", "원피스", "shoes", "신발", "other", "옷");

    private final ClothesStateProperties props;
    private final DefectDetector detector;
    private final GarmentDetector garments;

    public ClothesInspectService(ClothesStateProperties props, DefectDetector detector, GarmentDetector garments) {
        this.props = props;
        this.detector = detector;
        this.garments = garments;
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
        String version = garments.ready()
                ? props.modelVersion() + "+" + props.garmentModelVersion()
                : props.modelVersion();
        if (Images.meanLightness(img) < props.darkLightness()) {
            return new InspectResult(List.of(), null, "dark", "model", version, null);
        }

        List<Slot> slots = findGarments(img, m);
        Map<Slot, List<InspectResult.Candidate>> candidates = new LinkedHashMap<>();
        Slot topCandidate = null;
        double topConf = -1;
        int candidateCount = 0;
        for (Detection d : detector.detect(img, props.askThreshold())) {
            Slot s = owner(slots, d);
            String where = Where.of(relative(d, s.box), m, v);
            double conf = round(d.confidence());
            if (d.confidence() >= props.tellThreshold()) {
                s.defects.add(new InspectResult.Defect(d.kind(), where, conf));
            } else if (candidateCount < props.maxCandidates()) {
                candidateCount++;
                candidates.computeIfAbsent(s, k -> new ArrayList<>()).add(new InspectResult.Candidate(d.kind(),
                        where, conf, box(d)));
                if (d.confidence() > topConf) {
                    topConf = d.confidence();
                    topCandidate = s;
                }
            }
        }

        List<InspectResult.Garment> out = new ArrayList<>();
        for (int i = 0; i < slots.size(); i++) {
            Slot s = slots.get(i);
            s.id = "g" + (i + 1);
            GarmentColor.Result color = s.whole ? GarmentColor.of(img, m) : GarmentColor.inBox(crop(img, s.box), m);
            out.add(new InspectResult.Garment(s.id, s.category, NAMES.get(s.category), color.colorName(),
                    color.colorFamily(), color.tone(), null, null, null, "unknown", List.copyOf(s.defects),
                    s.box, m.equals("laid") ? position(s.box) : null, s.confidence));
        }
        var pending = topCandidate == null ? null
                : new InspectResult.PendingHelper(topCandidate.id, candidates.get(topCandidate),
                        "확실하지 않은 부분이 있어요.");
        return new InspectResult(out, null, "ok", "model", version, pending);
    }

    /** 옷 박스들. laid는 왼쪽부터, worn은 위(상의)부터 차례로 g1, g2... */
    private List<Slot> findGarments(BufferedImage img, String mode) {
        List<Slot> slots = new ArrayList<>();
        if (garments.ready()) {
            double min = "laid".equals(mode) ? props.laidGarmentThreshold() : props.garmentThreshold();
            for (Detection d : garments.detect(img, min)) {
                if (slots.size() >= props.maxGarments()) {
                    break;
                }
                slots.add(new Slot(d.kind(), box(d), round(d.confidence()), false));
            }
        }
        if (slots.isEmpty()) {
            slots.add(new Slot("other", WHOLE, null, true));
            return slots;
        }
        Comparator<Slot> order = "worn".equals(mode)
                ? Comparator.comparingDouble(s -> s.box.y() + s.box.h() / 2)
                : Comparator.comparingDouble(s -> s.box.x() + s.box.w() / 2);
        slots.sort(order);
        return slots;
    }

    /** 결함 중심이 들어가는 옷 중 가장 작은 옷(상의 위 아우터처럼 겹칠 때 더 가까운 쪽). 없으면 중심이 가장 가까운 옷. */
    static Slot owner(List<Slot> slots, Detection d) {
        double cx = d.centerX();
        double cy = d.centerY();
        Slot best = null;
        for (Slot s : slots) {
            InspectResult.Box b = s.box;
            boolean inside = cx >= b.x() && cx <= b.x() + b.w() && cy >= b.y() && cy <= b.y() + b.h();
            if (inside && (best == null || b.w() * b.h() < best.box.w() * best.box.h())) {
                best = s;
            }
        }
        if (best != null) {
            return best;
        }
        return slots.stream().min(Comparator.comparingDouble(s -> {
            double dx = s.box.x() + s.box.w() / 2 - cx;
            double dy = s.box.y() + s.box.h() / 2 - cy;
            return dx * dx + dy * dy;
        })).orElseThrow();
    }

    /** 결함 박스를 옷 박스 안 비율로 바꿈(where를 옷 한 벌 기준으로 말하려고). */
    static Detection relative(Detection d, InspectResult.Box b) {
        double w = b.w() <= 0 ? 1 : b.w();
        double h = b.h() <= 0 ? 1 : b.h();
        return new Detection(d.kind(), d.confidence(), (d.x() - b.x()) / w, (d.y() - b.y()) / h, d.w() / w, d.h() / h);
    }

    /** 약속 문서: box 중심 x가 0.4 미만이면 left, 0.6 초과면 right, 그 사이는 center. */
    static String position(InspectResult.Box b) {
        double cx = b.x() + b.w() / 2;
        return cx < 0.4 ? "left" : cx > 0.6 ? "right" : "center";
    }

    private static BufferedImage crop(BufferedImage img, InspectResult.Box b) {
        int x = clamp((int) Math.round(b.x() * img.getWidth()), 0, img.getWidth() - 1);
        int y = clamp((int) Math.round(b.y() * img.getHeight()), 0, img.getHeight() - 1);
        int w = clamp((int) Math.round(b.w() * img.getWidth()), 1, img.getWidth() - x);
        int h = clamp((int) Math.round(b.h() * img.getHeight()), 1, img.getHeight() - y);
        return img.getSubimage(x, y, w, h);
    }

    private static int clamp(int v, int lo, int hi) {
        return Math.max(lo, Math.min(hi, v));
    }

    private static InspectResult.Box box(Detection d) {
        return new InspectResult.Box(round(d.x()), round(d.y()), round(d.w()), round(d.h()));
    }

    private static double round(double v) {
        return Math.round(v * 100) / 100.0;
    }

    /** 응답을 만드는 동안 옷 한 벌. */
    static final class Slot {
        final String category;
        final InspectResult.Box box;
        final Double confidence;
        final boolean whole;
        final List<InspectResult.Defect> defects = new ArrayList<>();
        String id;

        Slot(String category, InspectResult.Box box, Double confidence, boolean whole) {
            this.category = category;
            this.box = box;
            this.confidence = confidence;
            this.whole = whole;
        }
    }
}
