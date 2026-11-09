package com.fashionagent.clothesstate.inspect;

import com.fashionagent.clothesstate.ClothesStateException;
import com.fashionagent.clothesstate.ClothesStateProperties;
import com.fashionagent.clothesstate.color.GarmentColor;
import com.fashionagent.clothesstate.detect.DefectDetector;
import com.fashionagent.clothesstate.detect.Detection;
import com.fashionagent.clothesstate.detect.GarmentDetector;
import com.fashionagent.clothesstate.image.Images;
import com.fashionagent.clothesstate.verify.CandidateStore;
import com.fashionagent.clothesstate.verify.CropVerifier;
import com.fashionagent.clothesstate.verify.GptCheckProperties;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.Semaphore;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

/**
 * clothes_inspect: 옷 사진 한 장 → 옷 한 벌마다 종류·색·얼룩·찢어짐. 우리가 학습한 모델(ONNX) 두 개를 이 서버에서 돌립니다.
 * 옷 전체 사진은 외부로 보내지 않고 저장하지 않습니다. 애매한 결함만 그 둘레 조각을 GPT에게 묻고, GPT가 결함이라고 한
 * 조각만 다시 학습할 후보로 저장합니다(정지원 2026-11-09, 지인 확인 대신).
 * <ul>
 * <li>옷 종류 모델이 옷 박스를 찾습니다. 못 찾거나 모델이 없으면 사진 전체를 옷 한 벌(category other)로 봅니다.</li>
 * <li>색은 옷 박스 안 픽셀로 정합니다(GarmentColor).</li>
 * <li>얼룩·찢어짐은 박스 중심이 들어가는 옷(여럿이면 가장 작은 옷)에 붙이고, where는 그 옷 박스 안 위치로 씁니다.</li>
 * <li>확신도 tell 이상은 defects(바로 알림, checkedBy=model). ask~tell은 결함 조각을 GPT에게 물어 얼룩·찢어짐이면
 * defects(checkedBy=gpt), 아니라면 버립니다. 그 아래는 버립니다(기준 0.5 / 0.2, 정지원 2026-11-09).</li>
 * <li>키가 없거나 동의하지 않았거나 GPT가 3초 안에 답하지 못한 후보, 입은 사진에서 얼굴이 들어갈 수 있는 위쪽 후보는
 * 예전처럼 0.3 이상만 pendingHelper로 돌려줍니다. 한 벌에 한 번만 묻는 약속이라 가장 확신도 높은 후보가 있는 옷 한 벌의 후보만 담습니다.</li>
 * </ul>
 */
@Service
public class ClothesInspectService {

    private static final Logger log = LoggerFactory.getLogger(ClothesInspectService.class);
    private static final Set<String> MODES = Set.of("laid", "worn");
    private static final Set<String> VIEWS = Set.of("front", "side", "back");
    private static final InspectResult.Box WHOLE = new InspectResult.Box(0, 0, 1, 1);

    private final ClothesStateProperties props;
    private final DefectDetector detector;
    private final GarmentDetector garments;
    private final GptCheckProperties gpt;
    private final CropVerifier verifier;
    private final CandidateStore store;
    /** GPT 동시 요청 수(시험과 같게 4). */
    private final Semaphore gptSlots = new Semaphore(4);

    public ClothesInspectService(ClothesStateProperties props, DefectDetector detector, GarmentDetector garments,
            GptCheckProperties gpt, CropVerifier verifier, CandidateStore store) {
        this.props = props;
        this.detector = detector;
        this.garments = garments;
        this.gpt = gpt;
        this.verifier = verifier;
        this.store = store;
    }

    public InspectResult inspect(String imageBase64, String mode, String view) {
        return inspect(imageBase64, mode, view, null);
    }

    /**
     * allowGpt=true일 때만 GPT에게 묻습니다. 사용자가 앱 처음 쓸 때 한 번 동의한 경우에만 agent가 true로 넣습니다
     * (정지원 2026-11-09 "처음 한 번만 동의"). null·false면 묻지 않고 pendingHelper로 돌려줍니다.
     */
    public InspectResult inspect(String imageBase64, String mode, String view, Boolean allowGpt) {
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
        long t0 = System.nanoTime();
        BufferedImage img = Images.decode(imageBase64, props.maxImageBytes().toBytes());
        long tDecode = System.nanoTime();
        String version = garments.ready()
                ? props.modelVersion() + "+" + props.garmentModelVersion()
                : props.modelVersion();
        if (Images.meanLightness(img) < props.darkLightness()) {
            return new InspectResult(List.of(), null, "dark", "model", version, null);
        }

        List<Slot> slots = findGarments(img, m);
        long tGarments = System.nanoTime();
        // 1) tell 이상은 바로 결함, ask~tell은 애매한 것으로 모음
        List<Detection> unsure = new ArrayList<>();
        for (Detection d : detector.detect(img, props.askThreshold())) {
            if (d.confidence() >= props.tellThreshold()) {
                Slot s = owner(slots, d);
                s.defects.add(new InspectResult.Defect(d.kind(), Where.of(relative(d, s.box), m, v), round(d.confidence()),
                        "model"));
            } else if (unsure.size() < props.maxCandidates()) {
                unsure.add(d);
            }
        }
        long tModel = System.nanoTime();
        // 2) 애매한 것은 결함 둘레 조각만 GPT에게 물음(동시에). 답이 없으면 null.
        List<CropVerifier.Verdict> verdicts = askGpt(img, m, Boolean.TRUE.equals(allowGpt) ? unsure : List.of(),
                unsure.size());
        Map<Slot, List<InspectResult.Candidate>> candidates = new LinkedHashMap<>();
        Slot topCandidate = null;
        double topConf = -1;
        int candidateCount = 0;
        int gptAsked = 0;
        for (int i = 0; i < unsure.size(); i++) {
            Detection d = unsure.get(i);
            CropVerifier.Verdict verdict = verdicts.get(i);
            Slot s = owner(slots, d);
            String where = Where.of(relative(d, s.box), m, v);
            double conf = round(d.confidence());
            if (verdict != null) {
                gptAsked++;
                if (verdict.defect(gpt.acceptConfidence())) {
                    s.defects.add(new InspectResult.Defect(verdict.label(), where, conf, "gpt"));
                }
                continue;
            }
            if (d.confidence() < gpt.fallbackAskThreshold()) {
                continue; // GPT로 확인하지 못한 0.2~0.3은 예전처럼 버림("확실하지 않아요"가 너무 잦지 않게)
            }
            candidateCount++;
            candidates.computeIfAbsent(s, k -> new ArrayList<>()).add(new InspectResult.Candidate(d.kind(),
                    where, conf, box(d)));
            if (d.confidence() > topConf) {
                topConf = d.confidence();
                topCandidate = s;
            }
        }
        long tDefects = System.nanoTime();
        List<InspectResult.Garment> out = new ArrayList<>();
        for (int i = 0; i < slots.size(); i++) {
            Slot s = slots.get(i);
            s.id = "g" + (i + 1);
            GarmentColor.Result color = s.whole ? GarmentColor.of(img, m) : GarmentColor.inBox(crop(img, s.box), m);
            GarmentKind k = s.kind;
            out.add(new InspectResult.Garment(s.id, k.category(), k.name(), color.colorName(),
                    color.colorFamily(), color.tone(), null, k.material(), null, k.formality(), List.copyOf(s.defects),
                    s.box, m.equals("laid") ? position(s.box) : null, s.confidence));
        }
        long tEnd = System.nanoTime();
        // 단계별 시간(ms). 사진 내용은 남기지 않고 크기와 개수만.
        log.info("clothes_inspect {}x{} {} garments={} defects={} gpt={} candidates={} | 풀기 {}ms, 옷 종류 {}ms, 얼룩 {}ms, GPT {}ms, 색·정리 {}ms, 전체 {}ms",
                img.getWidth(), img.getHeight(), m, out.size(),
                out.stream().mapToInt(g -> g.defects().size()).sum(), gptAsked, candidateCount,
                ms(t0, tDecode), ms(tDecode, tGarments), ms(tGarments, tModel), ms(tModel, tDefects), ms(tDefects, tEnd),
                ms(t0, tEnd));
        var pending = topCandidate == null ? null
                : new InspectResult.PendingHelper(topCandidate.id, candidates.get(topCandidate),
                        "확실하지 않은 부분이 있어요.");
        return new InspectResult(out, null, "ok", "model", version, pending);
    }

    /**
     * 애매한 결함마다 둘레 조각을 GPT에게 묻습니다(동시에 최대 4개). 결과는 unsure와 같은 순서이고, 묻지 않았거나 답이
     * 없으면 null. GPT가 결함이라고 한 조각만 학습 후보로 저장합니다.
     */
    private List<CropVerifier.Verdict> askGpt(BufferedImage img, String mode, List<Detection> unsure, int size) {
        List<CropVerifier.Verdict> out = new ArrayList<>(java.util.Collections.nCopies(size, null));
        if (unsure.isEmpty() || !verifier.ready()) {
            return out;
        }
        try (ExecutorService pool = Executors.newVirtualThreadPerTaskExecutor()) {
            List<Future<CropVerifier.Verdict>> futures = new ArrayList<>();
            for (Detection d : unsure) {
                if ("worn".equals(mode) && d.y() < gpt.wornSkipTop()) {
                    futures.add(null); // 얼굴이 들어갈 수 있어 보내지 않음
                    continue;
                }
                BufferedImage piece = defectCrop(img, d, gpt.cropMargin(), gpt.cropMinSide(), gpt.cropMaxSide());
                futures.add(pool.submit(() -> {
                    gptSlots.acquire();
                    try {
                        CropVerifier.Verdict verdict = verifier.verify(piece, d.kind());
                        if (verdict != null && verdict.defect(gpt.acceptConfidence())) {
                            store.save(piece, d.kind(), d.confidence(), verdict);
                        }
                        return verdict;
                    } finally {
                        gptSlots.release();
                    }
                }));
            }
            // 전체를 timeout 한 번(+조금) 안에 끝냄. 늦은 것은 확인하지 못한 후보로 둡니다.
            long deadline = System.nanoTime() + gpt.timeout().toNanos() + 500_000_000L;
            for (int i = 0; i < futures.size(); i++) {
                Future<CropVerifier.Verdict> f = futures.get(i);
                if (f == null) {
                    continue;
                }
                try {
                    out.set(i, f.get(Math.max(0, deadline - System.nanoTime()), java.util.concurrent.TimeUnit.NANOSECONDS));
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    break;
                } catch (java.util.concurrent.TimeoutException e) {
                    f.cancel(true);
                    log.warn("GPT 확인 시간 넘김");
                } catch (java.util.concurrent.ExecutionException e) {
                    log.warn("GPT 확인 실패: {}", e.getCause().getClass().getSimpleName());
                }
            }
            pool.shutdownNow();
        }
        return out;
    }

    /**
     * 결함 박스를 사방으로 박스 크기의 margin만큼 넓혀 자르고(사진 경계 안), 그래도 minSide보다 작으면 가운데 기준으로
     * 늘리고, 긴 변이 maxSide보다 크면 줄입니다(11-09 시험과 같은 방식).
     */
    static BufferedImage defectCrop(BufferedImage img, Detection d, double margin, int minSide, int maxSide) {
        int iw = img.getWidth();
        int ih = img.getHeight();
        double bw = d.w() * iw;
        double bh = d.h() * ih;
        double x0 = d.x() * iw - bw * margin;
        double y0 = d.y() * ih - bh * margin;
        double x1 = (d.x() + d.w()) * iw + bw * margin;
        double y1 = (d.y() + d.h()) * ih + bh * margin;
        int[] xs = span(x0, x1, minSide, iw);
        int[] ys = span(y0, y1, minSide, ih);
        BufferedImage sub = img.getSubimage(xs[0], ys[0], xs[1] - xs[0], ys[1] - ys[0]);
        int w = sub.getWidth();
        int h = sub.getHeight();
        double scale = Math.min(1.0, (double) maxSide / Math.max(w, h));
        int nw = Math.max(1, (int) Math.round(w * scale));
        int nh = Math.max(1, (int) Math.round(h * scale));
        BufferedImage out = new BufferedImage(nw, nh, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = out.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.drawImage(sub, 0, 0, nw, nh, null);
        g.dispose();
        return out;
    }

    /** [lo, hi)를 사진 안으로 자르고, min보다 짧으면 가운데 기준으로 min까지 늘림. */
    private static int[] span(double lo, double hi, int min, int limit) {
        int a = clamp((int) Math.floor(lo), 0, limit - 1);
        int b = clamp((int) Math.ceil(hi), a + 1, limit);
        if (b - a < min) {
            int want = Math.min(min, limit);
            int center = (a + b) / 2;
            a = clamp(center - want / 2, 0, limit - want);
            b = a + want;
        }
        return new int[] {a, b};
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
                slots.add(new Slot(GarmentKind.of(d.kind()), box(d), round(d.confidence()), false));
            }
        }
        if (slots.isEmpty()) {
            slots.add(new Slot(GarmentKind.OTHER, WHOLE, null, true));
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

    private static long ms(long from, long to) {
        return (to - from) / 1_000_000;
    }

    private static double round(double v) {
        return Math.round(v * 100) / 100.0;
    }

    /** 응답을 만드는 동안 옷 한 벌. */
    static final class Slot {
        final GarmentKind kind;
        final InspectResult.Box box;
        final Double confidence;
        final boolean whole;
        final List<InspectResult.Defect> defects = new ArrayList<>();
        String id;

        Slot(GarmentKind kind, InspectResult.Box box, Double confidence, boolean whole) {
            this.kind = kind;
            this.box = box;
            this.confidence = confidence;
            this.whole = whole;
        }
    }
}
