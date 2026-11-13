package com.fashionagent.clothesstate.color;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * 같이 입은 옷들의 색 조합과 자리(occasion)에 맞는 색인지 판단합니다(학습 없이 규칙으로, docs/color-rules.md).
 * clothes_inspect가 붙인 colorFamily·tone만 씁니다. 색을 모르는 옷(unknown)은 판단에서 뺍니다.
 *
 * <p>왜 규칙인가: 시각장애인에게는 "점수 0.4"가 아니라 "빨간 바지는 면접에 튀어요. 검정·네이비·회색 바지가 무난해요"처럼
 * 까닭과 바꿀 옷을 말해야 합니다. 옷 조합 모델(Polyvore 등)은 까닭을 말하지 못하고 자리(면접·장례식)도 모릅니다.
 */
public final class ColorHarmony {

    /** 판단에 쓰는 옷 정보. clothes_inspect Garment에서 필요한 것만. */
    public record Item(String id, String category, String name, String colorFamily, String colorName, String tone,
            String pattern) {
    }

    /**
     * level: bad(바꾸는 게 좋음) | check(맞는지 확인 부탁) | note(조금 튀지만 괜찮음) | tell(판정에 넣지 않고 색만 알려 줌)
     * | info(말하지 않고 결과에만 남김).
     * swapGarmentId: 바꾸면 좋은 옷. 없으면 null.
     */
    public record Issue(String rule, String level, List<String> garmentIds, String swapGarmentId, String say) {
    }

    /**
     * verdict: good | ok | bad. speech: 사용자에게 읽어 줄 1~2문장.
     * score: 어울림 점수 0~100(ColorScore). 몸에 걸친 옷 종류가 2가지 미만이면 null.
     * 같은 종류 옷이 두 벌 이상이면(비교 중) outfits에 조합마다 점수가 있고, score·scheme·scoreReason은 가장 높은 조합 것입니다.
     */
    public record Result(String verdict, List<Issue> issues, String speech, Integer score, String scheme,
            String scoreReason, List<ColorScore.Outfit> outfits) {

        public Result(String verdict, List<Issue> issues, String speech) {
            this(verdict, issues, speech, null, null, null, List.of());
        }
    }

    static final Set<String> NEUTRAL = Set.of("black", "white", "gray", "navy", "beige", "brown", "khaki");

    /**
     * 강한 색의 색상환 위치(도). 패션·미술에서 쓰는 RYB 색상환(빨강↔초록, 주황↔파랑, 노랑↔보라가 반대 색)을 씁니다.
     * 화면용 HSV 색상환으로 재면 빨강과 초록이 120도라 반대 색으로 잡히지 않습니다.
     */
    static final Map<String, Integer> HUE = Map.of(
            "red", 0, "orange", 60, "yellow", 120, "green", 180, "blue", 240, "purple", 300, "pink", 345);

    /** 이만큼 가까우면 비슷한 계열(빨강+주황, 파랑+보라)이라 괜찮음. */
    static final int NEAR = 60;
    /** 이만큼 멀면 반대 색. */
    static final int OPPOSITE = 150;

    /** 같이 입었을 때 모양을 이루는 옷. 신발·기타는 조합 규칙에서 뺍니다(자리 규칙에는 넣음). */
    private static final Set<String> BODY = Set.of("top", "bottom", "outer", "dress");

    /** 바꾸자고 할 때 먼저 고르는 순서: 하의 → 겉옷 → 상의 → 원피스. 하의가 가장 바꾸기 쉽고 기본색이 많아서. */
    private static final List<String> SWAP_ORDER = List.of("bottom", "outer", "top", "dress", "shoes", "other");

    private static final String SAFE = "검정·네이비·회색 같은 기본색";

    private ColorHarmony() {
    }

    /**
     * @param items    같이 입을 옷. 같은 종류가 두 벌 이상이면(비교 중) 종류가 다른 옷끼리만 조합을 봅니다.
     * @param occasion mourning | wedding | formal | business | active | casual (모르면 casual)
     * @param lighting dark면 어두운 색끼리 헷갈림(검정·네이비) 규칙은 믿을 수 없어 건너뜁니다.
     */
    public static Result check(List<Item> items, String occasion, String lighting) {
        List<Item> known = items.stream().filter(ColorHarmony::knownColor).toList();
        List<Issue> issues = new ArrayList<>();
        issues.addAll(occasionRules(known, occasion == null ? "casual" : occasion));
        List<Item> body = known.stream().filter(i -> BODY.contains(i.category())).toList();
        issues.addAll(pairRules(body, "dark".equals(lighting)));
        issues.addAll(countRules(body));

        issues.sort(Comparator.comparingInt((Issue i) -> rank(i.level())).thenComparingInt(i -> order(i.rule())));
        // info는 말하지 않고, tell은 판정에 넣지 않고 색만 알려 줍니다.
        List<Issue> judged = issues.stream().filter(i -> !"info".equals(i.level()) && !"tell".equals(i.level())).toList();
        List<Issue> told = issues.stream().filter(i -> "tell".equals(i.level())).toList();
        String verdict = judged.isEmpty() ? "good" : "bad".equals(judged.get(0).level()) ? "bad" : "ok";
        List<ColorScore.Outfit> outfits = ColorScore.score(known, issues);
        ColorScore.Outfit best = outfits.isEmpty() ? null : outfits.get(0);
        return new Result(verdict, List.copyOf(issues), speech(judged, told, known, items, lighting),
                best == null ? null : best.score(), best == null ? null : best.schemeName(),
                best == null ? null : best.reason(), outfits);
    }

    // ---------------------------------------------------------------- 자리 규칙

    private static List<Issue> occasionRules(List<Item> items, String occasion) {
        List<Issue> out = new ArrayList<>();
        switch (occasion) {
            case "mourning" -> {
                for (Item i : items) {
                    boolean whiteTop = "top".equals(i.category()) && "white".equals(i.colorFamily());
                    if (!mourningColor(i) && !whiteTop) {
                        out.add(new Issue("occasion_mourning", "bad", List.of(i.id()), i.id(),
                                "조문 자리에는 " + Korean.josa(label(i), "이", "가") + " 눈에 띌 수 있어요. "
                                        + Korean.josa("검정이나 짙은 회색 " + kind(i), "이", "가") + " 좋아요."));
                    }
                }
            }
            case "wedding" -> {
                boolean darkOuter = items.stream().anyMatch(i -> "outer".equals(i.category()) && !light(i));
                for (Item i : items) {
                    boolean white = "white".equals(i.colorFamily());
                    if (white && "dress".equals(i.category())) {
                        out.add(new Issue("occasion_wedding", "bad", List.of(i.id()), i.id(),
                                "결혼식에는 흰 원피스를 피하는 게 예의예요. 다른 색 원피스가 좋아요."));
                    } else if (white && "top".equals(i.category()) && !darkOuter) {
                        out.add(new Issue("occasion_wedding", "bad", List.of(i.id()), i.id(),
                                "결혼식에는 흰 상의만 입으면 신부와 겹쳐 보일 수 있어요. 위에 어두운 겉옷을 걸치거나 다른 색 상의가 좋아요."));
                    }
                }
            }
            case "formal" -> {
                // 2026-11-11 정지원: 면접·발표에 빨간 상의도 바꾸자고 해야 함. 연한 색(하늘색 셔츠)만 괜찮음.
                for (Item i : items) {
                    if (!chromatic(i) || pale(i) && "top".equals(i.category())) {
                        continue;
                    }
                    boolean dark = "dark".equals(i.tone());
                    String want = "top".equals(i.category()) ? "흰색·하늘색이나 네이비·회색 같은 차분한 색 " + kind(i)
                            : "검정·네이비·회색 " + kind(i);
                    if (dark) {
                        out.add(new Issue("occasion_formal", "note", List.of(i.id()), null,
                                label(i) + "도 괜찮지만, 격식 있는 자리에는 " + Korean.josa(want, "이", "가")
                                        + " 더 무난해요."));
                    } else if ("shoes".equals(i.category())) {
                        out.add(new Issue("occasion_formal", "note", List.of(i.id()), null,
                                Korean.josa(label(i), "은", "는") + " 격식 있는 자리에 조금 튈 수 있어요. 검정이나 갈색 구두가 무난해요."));
                    } else {
                        out.add(new Issue("occasion_formal", "bad", List.of(i.id()), i.id(),
                                Korean.josa(label(i), "은", "는") + " 격식 있는 자리에 튀어요. " + Korean.josa(want, "이", "가")
                                        + " 무난해요."));
                    }
                }
            }
            case "business" -> {
                for (Item i : items) {
                    if (strong(i) && !"top".equals(i.category()) && !"shoes".equals(i.category())) {
                        out.add(new Issue("occasion_business", "note", List.of(i.id()), null,
                                Korean.josa(label(i), "은", "는") + " 일하는 자리에 조금 튈 수 있어요."));
                    }
                }
            }
            default -> {
                // active·casual: 자리 색 규칙 없음
            }
        }
        return out;
    }

    private static boolean mourningColor(Item i) {
        return switch (i.colorFamily()) {
            case "black", "navy" -> true;
            case "gray" -> !"light".equals(i.tone());
            case "brown" -> "shoes".equals(i.category()) && "dark".equals(i.tone());
            default -> chromatic(i) && "dark".equals(i.tone()) && "shoes".equals(i.category());
        };
    }

    // ---------------------------------------------------------------- 옷끼리 조합

    private static List<Issue> pairRules(List<Item> body, boolean darkLight) {
        List<Issue> out = new ArrayList<>();
        for (int a = 0; a < body.size(); a++) {
            for (int b = a + 1; b < body.size(); b++) {
                Item x = body.get(a);
                Item y = body.get(b);
                if (x.category().equals(y.category()) || dressWith(x, y)) {
                    continue; // 같은 종류 두 벌은 비교 중인 것. 원피스와 상의·하의는 같이 입지 않음
                }
                Issue issue = pair(x, y, darkLight);
                if (issue != null) {
                    out.add(issue);
                }
            }
        }
        // 같은 두 옷 묶음에 대한 말은 하나만
        Map<Set<String>, Issue> unique = new LinkedHashMap<>();
        for (Issue i : out) {
            unique.putIfAbsent(new LinkedHashSet<>(i.garmentIds()), i);
        }
        return new ArrayList<>(unique.values());
    }

    private static boolean dressWith(Item x, Item y) {
        return ("dress".equals(x.category()) && Set.of("top", "bottom").contains(y.category()))
                || ("dress".equals(y.category()) && Set.of("top", "bottom").contains(x.category()));
    }

    private static Issue pair(Item x, Item y, boolean darkLight) {
        List<String> ids = List.of(x.id(), y.id());
        if (strong(x) && strong(y) && !x.colorFamily().equals(y.colorFamily())) {
            int d = hueDistance(x.colorFamily(), y.colorFamily());
            Item swap = swapOf(x, y);
            if (d >= OPPOSITE) {
                return new Issue("clash_complementary", "bad", ids, swap.id(),
                        both(x, y, false) + " 서로 반대 색이라 많이 튀어요. " + Korean.josa(label(swap), "을", "를") + " " + SAFE
                                + "으로 바꾸면 좋겠어요.");
            }
            if (d > NEAR) {
                return new Issue("clash_strong", "note", ids, swap.id(),
                        both(x, y, false) + " 둘 다 강한 색이라 조금 튈 수 있어요. 차분하게 입고 싶으면 "
                                + Korean.josa(label(swap), "을", "를") + " 기본색으로 바꿔 보세요.");
            }
            return null; // 비슷한 계열(빨강+주황, 빨강+분홍)은 괜찮음
        }
        if (!darkLight && confusingDarks(x, y)) {
            // 네이비+검정은 일부러 입는 멋진 조합이라(2026-11-11 정지원, "네이비에 블랙은 시크해" 코디 사진) 경고하지 않고
            // 판정은 그대로 둔 채 색만 정확히 알려 줍니다(tell). 둘 다 검정인 줄 알았던 사람은 여기서 알아챕니다.
            // 색 인식 모델로 이 조합이 실제로 맞은 비율: 코디 사진 18장 중 13장(픽셀 규칙 때는 12장 중 1~3장).
            Item first = SWAP_ORDER.indexOf(x.category()) > SWAP_ORDER.indexOf(y.category()) ? x : y;
            Item second = first == x ? y : x;
            return new Issue("dark_mix", "tell", ids, null,
                    Korean.josa(kind(first), "은", "는") + " " + darkName(first) + ", " + Korean.josa(kind(second), "은", "는")
                            + " " + Korean.josa(darkName(second), "이에요", "예요")
                            + ". 비슷해 보이지만 다른 색이라 함께 입으면 시크해요.");
        }
        return null;
    }

    /** 손으로 구분이 안 되고 눈으로 보면 다른 색인 어두운 색 짝(검정+네이비, 네이비+차콜, 검정+짙은 갈색). */
    static boolean confusingDarks(Item x, Item y) {
        String a = darkKey(x);
        String b = darkKey(y);
        if (a == null || b == null || a.equals(b)) {
            return false;
        }
        Set<String> pair = Set.of(a, b);
        return pair.equals(Set.of("black", "navy")) || pair.equals(Set.of("navy", "charcoal"))
                || pair.equals(Set.of("black", "darkbrown"));
    }

    private static String darkName(Item i) {
        if (i.colorName() != null && !i.colorName().isBlank()) {
            return i.colorName();
        }
        return switch (darkKey(i)) {
            case "black" -> "검정";
            case "navy" -> "네이비";
            case "charcoal" -> "차콜 그레이";
            default -> "어두운 갈색";
        };
    }

    private static String darkKey(Item i) {
        return switch (i.colorFamily()) {
            case "black" -> "black";
            case "navy" -> "navy";
            case "gray" -> "dark".equals(i.tone()) ? "charcoal" : null;
            case "brown" -> "dark".equals(i.tone()) ? "darkbrown" : null;
            default -> null;
        };
    }

    /** 강한 색 3가지 이상, 무늬 옷 2벌 이상. 종류마다 한 벌씩 고른 조합 중 가장 많은 경우로 셉니다. */
    private static List<Issue> countRules(List<Item> body) {
        List<Issue> out = new ArrayList<>();
        Map<String, List<Item>> byCategory = new LinkedHashMap<>();
        for (Item i : body) {
            byCategory.computeIfAbsent(i.category(), k -> new ArrayList<>()).add(i);
        }
        if (byCategory.containsKey("dress")) {
            byCategory.remove("top");
            byCategory.remove("bottom");
        }
        // 종류마다 강한 색이 있는 옷 하나씩(색이 서로 다르게) 골라 셉니다.
        List<Item> strongPicks = new ArrayList<>();
        Set<String> families = new LinkedHashSet<>();
        for (List<Item> group : byCategory.values()) {
            group.stream().filter(ColorHarmony::strong).filter(i -> !families.contains(i.colorFamily())).findFirst()
                    .ifPresent(i -> {
                        strongPicks.add(i);
                        families.add(i.colorFamily());
                    });
        }
        if (strongPicks.size() >= 3) {
            Item swap = strongPicks.stream().min(Comparator.comparingInt(i -> SWAP_ORDER.indexOf(i.category())))
                    .orElseThrow();
            out.add(new Issue("too_many_colors", "bad", strongPicks.stream().map(Item::id).toList(), swap.id(),
                    "강한 색이 " + strongPicks.size() + "가지라 복잡해 보여요. " + Korean.josa(label(swap), "을", "를") + " "
                            + SAFE + "으로 바꾸면 좋겠어요."));
        }
        List<Item> patterned = new ArrayList<>();
        for (List<Item> group : byCategory.values()) {
            group.stream().filter(i -> i.pattern() != null && !i.pattern().isBlank()).findFirst()
                    .ifPresent(patterned::add);
        }
        if (patterned.size() >= 2) {
            out.add(new Issue("two_patterns", "note", patterned.stream().map(Item::id).toList(), null,
                    "무늬 있는 옷이 두 벌이라 복잡해 보일 수 있어요. 하나는 무늬 없는 옷이 무난해요."));
        }
        return out;
    }

    // ---------------------------------------------------------------- 말하기

    private static String speech(List<Issue> issues, List<Issue> told, List<Item> known, List<Item> all,
            String lighting) {
        List<String> sentences = new ArrayList<>();
        Set<String> swaps = new java.util.HashSet<>();
        for (Issue i : issues) {
            if (sentences.size() == 2) {
                break;
            }
            if (i.swapGarmentId() != null && !swaps.add(i.swapGarmentId())) {
                continue; // 같은 옷을 바꾸자는 말은 한 번만
            }
            if (i.rule().startsWith("clash_") && issues.get(0).rule().equals("too_many_colors")) {
                continue; // 색이 많다고 이미 말했으면 둘씩 짝지은 말은 하지 않음
            }
            sentences.add(i.say());
        }
        if (sentences.isEmpty()) {
            long body = known.stream().filter(i -> BODY.contains(i.category())).map(Item::category).distinct().count();
            if (body >= 2) {
                sentences.add("색 조합이 잘 어울려요.");
            }
        }
        if (!told.isEmpty()) {
            sentences.add(told.get(0).say()); // 색 알려 주기는 한 번만
        }
        if ("dark".equals(lighting) && known.size() >= 2) {
            sentences.add("어두워서 색이 정확하지 않을 수 있어요.");
        }
        if (known.size() < all.size() && !known.isEmpty()) {
            sentences.add("색을 알아보지 못한 옷은 빼고 봤어요.");
        }
        return String.join(" ", sentences);
    }

    /** "빨강 니트와 초록 바지는"처럼 두 옷을 이어 부름. 상의·겉옷을 먼저, 하의를 뒤에. [object]면 을/를, 아니면 은/는. */
    private static String both(Item x, Item y, boolean object) {
        Item first = SWAP_ORDER.indexOf(x.category()) > SWAP_ORDER.indexOf(y.category()) ? x : y;
        Item second = first == x ? y : x;
        return Korean.josa(label(first), "과", "와") + " "
                + (object ? Korean.josa(label(second), "을", "를") : Korean.josa(label(second), "은", "는"));
    }

    /** 바꿀 옷: 원피스와 겉옷이면 겉옷, 아니면 하의 → 겉옷 → 상의 순. */
    private static Item swapOf(Item x, Item y) {
        return SWAP_ORDER.indexOf(x.category()) <= SWAP_ORDER.indexOf(y.category()) ? x : y;
    }

    static String label(Item i) {
        String color = i.colorName() == null || i.colorName().isBlank() ? "" : i.colorName() + " ";
        return color + kind(i);
    }

    private static String kind(Item i) {
        if (i.name() != null && !i.name().isBlank()) {
            return i.name();
        }
        return switch (i.category()) {
            case "top" -> "상의";
            case "bottom" -> "하의";
            case "outer" -> "겉옷";
            case "dress" -> "원피스";
            case "shoes" -> "신발";
            default -> "옷";
        };
    }

    // ---------------------------------------------------------------- 색 성질

    private static boolean knownColor(Item i) {
        return i.colorFamily() != null && (NEUTRAL.contains(i.colorFamily()) || HUE.containsKey(i.colorFamily()))
                && i.category() != null;
    }

    static boolean chromatic(Item i) {
        return HUE.containsKey(i.colorFamily());
    }

    /**
     * 눈에 띄는 강한 색. 어두운 진하기(버건디·짙은 초록)와 연한 색(하늘색)은 차분해서 빼고,
     * 하의 파랑(중간·어두움)은 청바지 색이라 뺍니다.
     */
    static boolean strong(Item i) {
        if (!chromatic(i) || "dark".equals(i.tone()) || pale(i)) {
            return false;
        }
        return !("bottom".equals(i.category()) && "blue".equals(i.colorFamily()) && !"light".equals(i.tone()));
    }

    /** 연한 색(하늘색·연분홍·연보라·민트·연한 ~). GarmentColor가 채도 낮고 밝은 색에 이 이름을 붙입니다(tone은 light). */
    static boolean pale(Item i) {
        String n = i.colorName();
        return chromatic(i) && n != null && (n.startsWith("연한 ") || PALE_NAMES.contains(n));
    }

    private static final Set<String> PALE_NAMES = Set.of("하늘색", "연분홍", "연보라", "민트");

    private static boolean light(Item i) {
        return "white".equals(i.colorFamily()) || "light".equals(i.tone());
    }

    static int hueDistance(String a, String b) {
        int d = Math.abs(HUE.get(a) - HUE.get(b));
        return Math.min(d, 360 - d);
    }

    /** 같은 단계면 자리 규칙 → 색이 많음 → 반대 색 → 어두운 색 섞임 → 강한 색 둘 → 무늬 순으로 말합니다. */
    private static int order(String rule) {
        if (rule.startsWith("occasion_")) {
            return 0;
        }
        return switch (rule) {
            case "too_many_colors" -> 1;
            case "clash_complementary" -> 2;
            case "dark_mix" -> 3;
            case "clash_strong" -> 4;
            default -> 5;
        };
    }

    private static int rank(String level) {
        return switch (level) {
            case "bad" -> 0;
            case "check" -> 1;
            case "note" -> 2;
            case "tell" -> 3;
            default -> 4;
        };
    }

    /** 받침에 따라 조사 고르기. */
    static final class Korean {
        private Korean() {
        }

        static String josa(String word, String withBatchim, String without) {
            return word + (batchim(word) ? withBatchim : without);
        }

        static boolean batchim(String word) {
            if (word == null || word.isEmpty()) {
                return false;
            }
            char c = word.charAt(word.length() - 1);
            if (c < 0xAC00 || c > 0xD7A3) {
                return false;
            }
            return (c - 0xAC00) % 28 != 0;
        }
    }
}
