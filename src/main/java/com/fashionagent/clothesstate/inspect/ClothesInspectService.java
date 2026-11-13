package com.fashionagent.clothesstate.inspect;

import com.fashionagent.clothesstate.ClothesStateException;
import com.fashionagent.clothesstate.ClothesStateProperties;
import com.fashionagent.clothesstate.color.ColorModel;
import com.fashionagent.clothesstate.color.GarmentColor;
import com.fashionagent.clothesstate.detect.DefectDetector;
import com.fashionagent.clothesstate.detect.Detection;
import com.fashionagent.clothesstate.detect.GarmentDetector;
import com.fashionagent.clothesstate.embed.GarmentEmbedder;
import com.fashionagent.clothesstate.embed.GarmentTextCheck;
import com.fashionagent.clothesstate.image.Images;
import com.fashionagent.clothesstate.pattern.PatternClassifier;
import com.fashionagent.clothesstate.pattern.OpenAiPatternVerifier;
import com.fashionagent.clothesstate.pattern.PatternVerifier;
import com.fashionagent.clothesstate.verify.CandidateStore;
import com.fashionagent.clothesstate.verify.CropVerifier;
import com.fashionagent.clothesstate.verify.GptCheckProperties;
import com.fashionagent.clothesstate.wear.WearCheckProperties;
import com.fashionagent.clothesstate.wear.WearChecker;
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
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

/**
 * clothes_inspect: 옷 사진 한 장 → 옷 한 벌마다 종류·색·얼룩·찢어짐. 우리가 학습한 모델(ONNX) 두 개를 이 서버에서 돌립니다.
 * 찍은 사진 전체는 외부로 보내지 않고 저장하지 않습니다. 애매한 결함만 그 둘레 조각을 GPT에게 묻고, GPT가 결함이라고 한
 * 조각만 다시 학습할 후보로 저장합니다(정지원 2026-11-09, 지인 확인 대신).
 * <ul>
 * <li>옷 종류 모델이 옷 박스를 찾습니다. 못 찾거나 모델이 없으면 사진 전체를 옷 한 벌(category other)로 봅니다.</li>
 * <li>색은 옷 박스 조각을 색 인식 모델(ColorModel, K-Fashion 20색)로 정합니다. 모델 파일이 없으면 픽셀 규칙(GarmentColor).</li>
 * <li>얼룩·찢어짐은 박스 중심이 들어가는 옷(여럿이면 가장 작은 옷)에 붙이고, where는 그 옷 박스 안 위치로 씁니다.</li>
 * <li>확신도 tell 이상은 defects(바로 알림, checkedBy=model). ask~tell은 결함 조각을 GPT에게 물어 얼룩·찢어짐이면
 * defects(checkedBy=gpt), 아니라면 버립니다. 그 아래는 버립니다(기준 0.5 / 0.2, 정지원 2026-11-09).</li>
 * <li>키가 없거나 동의하지 않았거나 GPT가 3초 안에 답하지 못한 후보, 입은 사진에서 얼굴이 들어갈 수 있는 위쪽 후보는
 * 예전처럼 0.3 이상만 pendingHelper로 돌려줍니다. 한 벌에 한 번만 묻는 약속이라 가장 확신도 높은 후보가 있는 옷 한 벌의 후보만 담습니다.</li>
 * <li>무늬 모델이 있으면 옷마다 pattern(plain·stripe·check·floral·dot·graphic·animal·other)을 붙입니다. 무늬라고 봤지만
 * 확신이 낮으면 null(모름)이고, 신발은 학습하지 않아 null입니다. 모름이면 내려놓은 옷 사진이고 GPT를 쓸 수 있을 때만
 * 그 옷 박스를 GPT에게 물어 patternCheckedBy=gpt로 채우고, 그 그림과 GPT 답을 무늬 재학습 후보로 저장합니다(정지원 2026-11-11).
 * 입은 사진은 몸·얼굴이 들어가 보내지 않습니다.</li>
 * <li>입은 사진이고 GPT를 쓸 수 있으면 옷마다(신발 빼고 위에서부터 3벌) 옷 박스를 GPT에게 보내 옷 실수(뒤집어 입음·앞뒤 바뀜·
 * 단추 어긋남·택 튀어나옴)를 묻고 wearIssues로 돌려줍니다(정지원 2026-11-13). 얼굴이 들어갈 수 있는 사진 위쪽(앞 25%, 뒤 12%)은
 * 회색으로 가린 뒤 보내고, 조각은 저장하지 않습니다. 얼룩 확인과 동시에 돌립니다.</li>
 * <li>벡터 모델이 있으면 옷마다 embedding(옷 박스 이미지 벡터, 옷장 같은 옷 판단용, 이슈 #15)을 붙입니다.</li>
 * </ul>
 */
@Service
public class ClothesInspectService {

    private static final Logger log = LoggerFactory.getLogger(ClothesInspectService.class);
    private static final Set<String> MODES = Set.of("laid", "worn");
    // agent는 앞·뒤만 보냅니다(2026-11-12 옆모습 뺌). side는 예전 호출이 깨지지 않게 받아만 둡니다.
    private static final Set<String> VIEWS = Set.of("front", "side", "back");
    private static final InspectResult.Box WHOLE = new InspectResult.Box(0, 0, 1, 1);

    private final ClothesStateProperties props;
    private final DefectDetector detector;
    private final GarmentDetector garments;
    private final GptCheckProperties gpt;
    private final CropVerifier verifier;
    private final CandidateStore store;
    private final GarmentEmbedder embedder;
    private final PatternClassifier patterns;
    private final PatternVerifier patternVerifier;
    private final WearChecker wear;
    private final WearCheckProperties wearProps;
    /** GPT 동시 요청 수(시험과 같게 4). */
    private final Semaphore gptSlots = new Semaphore(4);

    /** 색 인식 모델. 없거나 파일이 없으면 픽셀 규칙(GarmentColor)으로 색을 정합니다. */
    private final ColorModel colors;

    @Autowired
    public ClothesInspectService(ClothesStateProperties props, DefectDetector detector, GarmentDetector garments,
            GptCheckProperties gpt, CropVerifier verifier, CandidateStore store, GarmentEmbedder embedder,
            ColorModel colors, PatternClassifier patterns, PatternVerifier patternVerifier, WearChecker wear,
            WearCheckProperties wearProps) {
        this.props = props;
        this.detector = detector;
        this.garments = garments;
        this.gpt = gpt;
        this.verifier = verifier;
        this.store = store;
        this.embedder = embedder;
        this.colors = colors;
        this.patterns = patterns;
        this.patternVerifier = patternVerifier;
        this.wear = wear;
        this.wearProps = wearProps;
    }

    /** 색 인식 모델 없이(픽셀 규칙만). 시험용. */
    public ClothesInspectService(ClothesStateProperties props, DefectDetector detector, GarmentDetector garments,
            GptCheckProperties gpt, CropVerifier verifier, CandidateStore store, GarmentEmbedder embedder,
            PatternClassifier patterns, PatternVerifier patternVerifier) {
        this(props, detector, garments, gpt, verifier, store, embedder, null, patterns, patternVerifier, WearChecker.NONE,
                DEFAULT_WEAR);
    }

    /** 옷 실수 확인 시험용(색 인식 모델 없이). */
    public ClothesInspectService(ClothesStateProperties props, DefectDetector detector, GarmentDetector garments,
            GptCheckProperties gpt, CropVerifier verifier, CandidateStore store, GarmentEmbedder embedder,
            PatternClassifier patterns, PatternVerifier patternVerifier, WearChecker wear, WearCheckProperties wearProps) {
        this(props, detector, garments, gpt, verifier, store, embedder, null, patterns, patternVerifier, wear, wearProps);
    }

    static final WearCheckProperties DEFAULT_WEAR = new WearCheckProperties(true, "", java.time.Duration.ofSeconds(6),
            768, 0.7, 0.25, 0.12, 3);

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
            throw new ClothesStateException("bad_request", "view는 front 또는 back이어야 해요.");
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
        if (patterns.ready()) {
            version += "+" + patterns.version();
        }
        if (Images.meanLightness(img) < props.darkLightness()) {
            return new InspectResult(List.of(), null, "dark", "model", version, null, null);
        }

        List<Slot> slots = findGarments(img, m);
        int textChanged = textCheck(img, slots, m);
        long tGarments = System.nanoTime();
        if (slots.isEmpty()) {
            // 빈 벽·바닥처럼 옷이 아닌 것만 보였음(2026-11-12 시뮬: 빈 벽을 "흰색 티셔츠"·"치마"로 봄). agent가 다시 비춰 달라고 함.
            log.info("clothes_inspect {}x{} {} garments=0 (글 벡터로 옷이 아니라고 봄)", img.getWidth(), img.getHeight(), m);
            return new InspectResult(List.of(), null, "ok", "model", version, null, null);
        }
        // 0) 입은 사진이면 옷 실수 확인을 먼저 보내 두고(GPT, 동시에) 맨 끝에 모음
        WearJob wearJob = startWear(img, slots, m, v, Boolean.TRUE.equals(allowGpt));
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
        String[][] pats = patterns(img, slots, m.equals("laid") && Boolean.TRUE.equals(allowGpt));
        long tPatterns = System.nanoTime();
        boolean embed = embedder.ready();
        List<InspectResult.Garment> out = new ArrayList<>();
        for (int i = 0; i < slots.size(); i++) {
            Slot s = slots.get(i);
            s.id = "g" + (i + 1);
            BufferedImage piece = s.whole ? null : crop(img, s.box);
            GarmentColor.Result color = s.whole ? GarmentColor.of(img, m) : color(piece, m);
            String pattern = pats[i][0];
            String guess = patternGuess(pats[i]);
            GarmentColor.Result sub = null;
            if (piece != null && (patternedByModel(pattern, pats[i][1]) || guess != null || "gpt".equals(pats[i][1]))) {
                sub = GarmentColor.secondColor(piece, m, color);
            }
            // 색이 하나뿐인데 모델이 무늬(줄·체크·도트·호피·꽃·기타)를 0.9 미만으로 말하면 믿지 않습니다. 2026-11-12: 모니터를 찍은
            // 흰 티(화면 줄무늬)를 "흰색·갈색 줄무늬", 청바지를 줄무늬·도트로 봄. 그래픽(작은 프린트)과 GPT가 본 무늬는 그대로.
            // 기준 0.9: 무늬 시험 사진(test_sh·test_fp, 무늬별 최대 300장)에서 무지를 무늬라 한 비율 0.7→0.0%·4.3→3.3%,
            // 맞힌 무늬는 체크 -1.7·줄 -2.5·꽃 -4.5%p만 "모름"으로. 늘 버리면(1.0) 체크 -14·줄 -11%p라 쓰지 않음.
            if (sub == null && patternedByModel(pattern, pats[i][1]) && !"graphic".equals(pattern)
                    && Double.parseDouble(pats[i][3]) < SINGLE_COLOR_PATTERN_SURE) {
                log.info("무늬 {}({})를 버림: 옷 색이 한 가지만 보임 ({})", pattern, pats[i][3], color.colorName());
                pattern = null;
                guess = null;
            }
            if (sub == null && guess != null && !"graphic".equals(guess)) {
                guess = null;
            }
            GarmentKind k = s.kind;
            out.add(new InspectResult.Garment(s.id, k.category(), k.name(), color.colorName(),
                    color.colorFamily(), color.tone(), sub == null ? null : sub.colorName(),
                    sub == null ? null : sub.colorFamily(), pattern, pattern == null ? null : pats[i][1], guess, k.material(), null, k.formality(), List.copyOf(s.defects),
                    s.box, m.equals("laid") ? position(s.box) : null, s.confidence,
                    !embed ? null : s.embedding != null ? s.embedding : embed(img, s.box)));
        }
        List<InspectResult.WearIssue> wearIssues = wearJob == null ? null : wearJob.collect(slots);
        long tEnd = System.nanoTime();
        if (wearIssues != null) {
            log.info("clothes_inspect wear {} {}", v, wearIssues.stream()
                    .map(w -> w.garmentId() + "=" + w.kind() + "/" + String.format("%.2f", w.confidence())).toList());
        }
        // 단계별 시간(ms). 사진 내용은 남기지 않고 크기와 개수만.
        log.info("clothes_inspect {}x{} {} garments={} textChanged={} defects={} gpt={} candidates={} | 풀기 {}ms, 옷 종류 {}ms, 얼룩 {}ms, GPT {}ms, 무늬 {}ms, 색·벡터·정리 {}ms, 전체 {}ms",
                img.getWidth(), img.getHeight(), m, out.size(), textChanged,
                out.stream().mapToInt(g -> g.defects().size()).sum(), gptAsked, candidateCount,
                ms(t0, tDecode), ms(tDecode, tGarments), ms(tGarments, tModel), ms(tModel, tDefects), ms(tDefects, tPatterns), ms(tPatterns, tEnd),
                ms(t0, tEnd));
        // 옷마다 무엇으로 봤는지(종류·색·무늬·확신도). 말이 틀렸을 때 서버가 틀렸는지 agent가 틀렸는지 가리려고(2026-11-12).
        log.info("clothes_inspect garments {}", out.stream()
                .map(g -> g.id() + "=" + g.category() + "/" + g.name() + "/" + g.colorName()
                        + (g.subColorName() == null ? "" : "+" + g.subColorName()) + "/" + g.pattern()
                        + (g.patternGuess() == null ? "" : "(" + g.patternGuess() + "?)")
                        + "/" + g.position() + "/" + String.format("%.2f", g.confidence() == null ? 0.0 : g.confidence()))
                .toList());
        var pending = topCandidate == null ? null
                : new InspectResult.PendingHelper(topCandidate.id, candidates.get(topCandidate),
                        "확실하지 않은 부분이 있어요.");
        return new InspectResult(out, null, "ok", "model", version, pending, embed ? embedder.version() : null,
                wearIssues);
    }

    /**
     * 입은 사진의 옷 박스마다 옷 실수 확인을 보냅니다(동시에). 입은 사진이 아니거나 동의·키가 없으면 null(확인 안 함).
     * 옷을 하나도 못 찾아 사진 전체를 한 벌로 본 경우도 보내지 않습니다(얼굴이 다 들어감).
     */
    private WearJob startWear(BufferedImage img, List<Slot> slots, String mode, String view, boolean allowGpt) {
        if (!"worn".equals(mode) || !allowGpt || !wearProps.enabled() || !wear.ready()) {
            return null;
        }
        String side = "back".equals(view) ? "back" : "front";
        double hideTop = "back".equals(side) ? wearProps.backHideTop() : wearProps.frontHideTop();
        ExecutorService pool = Executors.newVirtualThreadPerTaskExecutor();
        List<Slot> asked = new ArrayList<>();
        List<Future<List<WearChecker.Issue>>> futures = new ArrayList<>();
        for (Slot s : slots) {
            if (asked.size() >= wearProps.maxGarments()) {
                break;
            }
            if (s.whole || "shoes".equals(s.kind.category())) {
                continue;
            }
            BufferedImage piece = wearCrop(img, s.box, hideTop, wearProps.maxSide());
            if (piece == null) {
                continue; // 옷 박스가 다 가려지는 위쪽에 있음
            }
            String name = s.kind.english();
            asked.add(s);
            futures.add(pool.submit(() -> {
                gptSlots.acquire();
                try {
                    return wear.check(piece, name, side);
                } finally {
                    gptSlots.release();
                }
            }));
        }
        if (asked.isEmpty()) {
            pool.shutdownNow();
            pool.close();
            return null;
        }
        return new WearJob(pool, asked, futures, System.nanoTime() + wearProps.timeout().toNanos() + 500_000_000L);
    }

    /** 보내 둔 옷 실수 확인. collect는 한 번만 부릅니다. */
    private final class WearJob {
        final ExecutorService pool;
        final List<Slot> asked;
        final List<Future<List<WearChecker.Issue>>> futures;
        final long deadline;

        WearJob(ExecutorService pool, List<Slot> asked, List<Future<List<WearChecker.Issue>>> futures, long deadline) {
            this.pool = pool;
            this.asked = asked;
            this.futures = futures;
            this.deadline = deadline;
        }

        /** 기준(acceptConfidence) 이상만. 한 벌도 답을 못 받았으면 null(확인 못 함). */
        List<InspectResult.WearIssue> collect(List<Slot> slots) {
            List<InspectResult.WearIssue> out = new ArrayList<>();
            boolean answered = false;
            try {
                for (int i = 0; i < futures.size(); i++) {
                    Slot s = asked.get(i);
                    try {
                        List<WearChecker.Issue> issues = futures.get(i).get(Math.max(0, deadline - System.nanoTime()),
                                java.util.concurrent.TimeUnit.NANOSECONDS);
                        if (issues == null) {
                            continue;
                        }
                        answered = true;
                        for (WearChecker.Issue issue : issues) {
                            if (issue.confidence() >= wearProps.acceptConfidence() && fits(issue.kind(), s.kind)) {
                                out.add(new InspectResult.WearIssue(issue.kind(), s.id, s.kind.category(),
                                        round(issue.confidence()), "gpt"));
                            }
                        }
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        break;
                    } catch (java.util.concurrent.TimeoutException e) {
                        futures.get(i).cancel(true);
                        log.warn("GPT 옷 실수 확인 시간 넘김");
                    } catch (java.util.concurrent.ExecutionException e) {
                        log.warn("GPT 옷 실수 확인 실패: {}", e.getCause().getClass().getSimpleName());
                    }
                }
            } finally {
                pool.shutdownNow();
                pool.close();
            }
            return answered ? out : null;
        }
    }

    /** 단추 어긋남은 단추가 있을 수 있는 옷(셔츠·가디건·자켓·코트·조끼·원피스·겉옷)에만. 티셔츠·니트·바지·치마는 버림. */
    static boolean fits(String kind, GarmentKind k) {
        if (!"buttons_misaligned".equals(kind)) {
            return true;
        }
        return Set.of("셔츠", "가디건", "자켓", "코트", "조끼", "원피스", "겉옷", "상의", "아우터", "옷").contains(k.name());
    }

    /**
     * 옷 박스를 조금(사방 5%) 넓혀 자르고, 사진 높이 hideTop보다 위쪽은 회색으로 칠한 뒤 긴 변을 maxSide 이하로 줄입니다.
     * 박스가 전부 가려지면 null.
     */
    static BufferedImage wearCrop(BufferedImage img, InspectResult.Box b, double hideTop, int maxSide) {
        int iw = img.getWidth();
        int ih = img.getHeight();
        double mx = b.w() * 0.05;
        double my = b.h() * 0.05;
        int x0 = clamp((int) Math.floor((b.x() - mx) * iw), 0, iw - 1);
        int y0 = clamp((int) Math.floor((b.y() - my) * ih), 0, ih - 1);
        int x1 = clamp((int) Math.ceil((b.x() + b.w() + mx) * iw), x0 + 1, iw);
        int y1 = clamp((int) Math.ceil((b.y() + b.h() + my) * ih), y0 + 1, ih);
        int hideY = (int) Math.ceil(hideTop * ih);
        if (hideY >= y1 - 8) {
            return null;
        }
        int w = x1 - x0;
        int h = y1 - y0;
        double scale = Math.min(1.0, (double) maxSide / Math.max(w, h));
        int nw = Math.max(1, (int) Math.round(w * scale));
        int nh = Math.max(1, (int) Math.round(h * scale));
        BufferedImage out = new BufferedImage(nw, nh, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = out.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.drawImage(img.getSubimage(x0, y0, w, h), 0, 0, nw, nh, null);
        if (hideY > y0) {
            g.setColor(new java.awt.Color(128, 128, 128));
            g.fillRect(0, 0, nw, (int) Math.ceil((hideY - y0) * scale));
        }
        g.dispose();
        return out;
    }

    /**
     * 옷마다 {무늬, checkedBy}. 모델이 없거나 실패하면 {null, null}. 모델이 "모름"이고 askGpt면 그 옷 박스만
     * GPT에게 동시에 묻고, timeout 안에 답한 것만 채웁니다.
     */
    private String[][] patterns(BufferedImage img, List<Slot> slots, boolean askGpt) {
        String[][] out = new String[slots.size()][4]; // {무늬, checkedBy, 모델 1등, 1등 확률}
        if (!patterns.ready()) {
            return out;
        }
        List<Integer> unsure = new ArrayList<>();
        List<PatternClassifier.Result> unsureResults = new ArrayList<>();
        for (int i = 0; i < slots.size(); i++) {
            Slot s = slots.get(i);
            if ("shoes".equals(s.kind.category())) {
                continue;
            }
            try {
                PatternClassifier.Result r = patterns.classify(img, s.box.x(), s.box.y(), s.box.w(), s.box.h());
                out[i][2] = r.top();
                out[i][3] = String.format(java.util.Locale.ROOT, "%.2f", r.confidence());
                if (r.pattern() != null) {
                    out[i] = new String[] {r.pattern(), "model", r.top(), out[i][3]};
                } else {
                    unsure.add(i);
                    unsureResults.add(r);
                }
            } catch (RuntimeException e) {
                log.warn("무늬 판단 실패: {}", e.toString());
            }
        }
        if (!askGpt || unsure.isEmpty() || !patternVerifier.ready()) {
            return out;
        }
        try (ExecutorService pool = Executors.newVirtualThreadPerTaskExecutor()) {
            List<Future<PatternVerifier.Answer>> futures = new ArrayList<>();
            for (int j = 0; j < unsure.size(); j++) {
                BufferedImage piece = OpenAiPatternVerifier.shrink(crop(img, slots.get(unsure.get(j)).box),
                        OpenAiPatternVerifier.MAX_SIDE);
                PatternClassifier.Result mine = unsureResults.get(j);
                futures.add(pool.submit(() -> {
                    gptSlots.acquire();
                    try {
                        PatternVerifier.Answer a = patternVerifier.ask(piece);
                        if (a != null) {
                            store.savePattern(piece, a.pattern(), a.confidence(), a.model(), mine.top(), mine.confidence());
                        }
                        return a;
                    } finally {
                        gptSlots.release();
                    }
                }));
            }
            long deadline = System.nanoTime() + gpt.timeout().toNanos() + 500_000_000L;
            for (int j = 0; j < futures.size(); j++) {
                try {
                    PatternVerifier.Answer a = futures.get(j).get(Math.max(0, deadline - System.nanoTime()),
                            java.util.concurrent.TimeUnit.NANOSECONDS);
                    if (a != null) {
                        out[unsure.get(j)][0] = a.pattern();
                        out[unsure.get(j)][1] = "gpt";
                    }
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    break;
                } catch (java.util.concurrent.TimeoutException e) {
                    futures.get(j).cancel(true);
                    log.warn("GPT 무늬 확인 시간 넘김");
                } catch (java.util.concurrent.ExecutionException e) {
                    log.warn("GPT 무늬 확인 실패: {}", e.getCause().getClass().getSimpleName());
                }
            }
            pool.shutdownNow();
        }
        return out;
    }

    /** 두 번째 색이 없을 때 무늬를 믿는 모델 확률(위 주석 참고). */
    static final double SINGLE_COLOR_PATTERN_SURE = 0.9;

    private static boolean patternedByModel(String pattern, String checkedBy) {
        return pattern != null && !"plain".equals(pattern) && "model".equals(checkedBy);
    }

    /**
     * 무늬를 확실히 모를 때(pattern=null) 모델 1등이 무늬면 그 무늬. agent가 "호피 무늬 같아요"처럼 약하게 말하게.
     * 1등이 무지이거나 확률이 0.35보다 낮으면 null.
     */
    private static String patternGuess(String[] p) {
        if (p[0] != null || p[2] == null || "plain".equals(p[2]) || p[3] == null) {
            return null;
        }
        return Double.parseDouble(p[3]) >= 0.35 ? p[2] : null;
    }

    /** 옷 박스 이미지 벡터. 실패해도 검사 결과는 돌려주도록 null(로그만). */
    private float[] embed(BufferedImage img, InspectResult.Box b) {
        try {
            return embedder.embed(img, b.x(), b.y(), b.w(), b.h());
        } catch (RuntimeException e) {
            log.warn("옷 벡터 만들기 실패: {}", e.toString());
            return null;
        }
    }

    /** 옷 박스 조각의 색. 색 인식 모델이 있으면 모델로 색을 정하고 밝기(tone)는 픽셀로 보탭니다. 없으면 픽셀 규칙. */
    private GarmentColor.Result color(BufferedImage crop, String mode) {
        if (colors == null || !colors.ready()) {
            return GarmentColor.inBox(crop, mode);
        }
        return GarmentColor.fromModel(colors.probabilities(crop), GarmentColor.boxRgb(crop, mode));
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
    /**
     * 옷 박스 이미지 벡터를 옷 이름 글 벡터와 견줍니다(옷 벡터 모델과 글 벡터 파일이 있을 때만).
     * <ul>
     * <li>"빈 벽·바닥"이 어떤 옷 이름보다 noneMargin 넘게 맞으면 그 박스를 뺍니다.</li>
     * <li>옷 종류 모델 확신도가 낮으면(내려놓은 사진 keepLaid, 입은 사진 keepWorn 미만) 글 벡터가 고른 이름으로 바꿉니다.
     * 사진 전체를 한 벌로 본 경우(종류 모름)도 바꿉니다. 신발은 모델이 강해서(mAP50 0.99) 그대로 둡니다.</li>
     * </ul>
     * 근거: docs/training-log.md "글 벡터로 종류 다시 보기"(2026-11-12). 바꾼 옷 수를 돌려줍니다.
     */
    int textCheck(BufferedImage img, List<Slot> slots, String mode) {
        GarmentTextCheck check = embedder.ready() ? embedder.textCheck() : null;
        if (check == null) {
            return 0;
        }
        var p = check.props();
        double keep = "worn".equals(mode) ? p.keepWorn() : p.keepLaid();
        int changed = 0;
        for (var it = slots.iterator(); it.hasNext();) {
            Slot s = it.next();
            s.embedding = embed(img, s.box);
            if (s.embedding == null) {
                continue;
            }
            GarmentTextCheck.Result r = check.check(s.embedding);
            if (r.notGarment(p.noneMargin())) {
                it.remove();
                continue;
            }
            boolean unsure = s.whole || s.confidence == null || s.confidence < keep;
            // 코트·재킷이 애매해 "겉옷"이라고 한 것은 글 벡터도 둘을 잘 못 가려서 그대로 둠
            boolean outerUnsure = "겉옷".equals(s.kind.name()) && ("jacket".equals(r.name()) || "coat".equals(r.name()));
            if (unsure && r.name() != null && !outerUnsure && !"shoes".equals(s.kind.category())) {
                GarmentKind k = GarmentKind.of(r.name());
                if (!k.name().equals(s.kind.name())) {
                    s.kind = k;
                    changed++;
                }
            }
        }
        return changed;
    }

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
        GarmentKind kind;
        /** textCheck에서 미리 만든 옷 벡터(응답에 다시 씀). */
        float[] embedding;
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
