package com.fashionagent.clothesstate.color;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.fashionagent.clothesstate.color.ColorHarmony.Issue;
import com.fashionagent.clothesstate.color.ColorHarmony.Item;

/**
 * 같이 입을 옷의 색 어울림 점수(0~100). 학습 없이 색상환 이론으로 계산합니다(docs/color-rules.md "4. 어울림 점수").
 *
 * <p>계산: ① 색 조합 종류(scheme)로 기본 점수를 정하고 ② 명도 대비·무늬·자리 규칙으로 더하거나 뺍니다.
 * 같은 옷이면 늘 같은 점수가 나오고, 점수마다 까닭(parts)이 붙어서 "왜 85점인지" 말할 수 있습니다.
 * 판단(verdict)은 {@link ColorHarmony}가 정하고, 점수는 그 판단을 숫자와 까닭으로 보여 주는 것입니다.
 */
public final class ColorScore {

    /** 점수 한 줄. points는 기본 점수면 그 값, 아니면 더하거나 뺀 값. */
    public record Part(String why, int points) {
    }

    /**
     * 한 벌 조합의 점수.
     * scheme: neutral | one_point | complementary_point | same_family | analogous | two_strong | complementary | many_strong
     * schemeName: 사용자에게 말할 이름(예: "기본색에 포인트 색 하나"). reason: 까닭 한 문장.
     */
    public record Outfit(List<String> garmentIds, int score, String scheme, String schemeName, String reason,
            List<Part> parts) {
    }

    /** 이 점수 이상이면 잘 어울림, 미만이면 무난, 60 미만이면 바꾸면 좋음. */
    public static final int GOOD = 80;
    public static final int OK = 60;

    /** 몸에 걸쳐 모양을 이루는 옷(ColorHarmony.BODY와 같음). */
    private static final Set<String> BODY = Set.of("top", "bottom", "outer", "dress");
    /** 비교 중에 만들 조합 수 상한. 옷 10벌이 최대라 넉넉합니다. */
    private static final int MAX_OUTFITS = 12;

    private ColorScore() {
    }

    /**
     * @param known  색을 아는 옷(ColorHarmony가 unknown을 뺀 목록)
     * @param issues ColorHarmony가 찾은 문제. 자리 규칙(occasion_*)만 점수에서 뺍니다(조합 규칙은 scheme에 이미 들어 있음).
     * @return 점수 높은 순. 몸에 걸친 옷 종류가 2가지 미만이면 빈 목록(조합을 볼 수 없음).
     */
    static List<Outfit> score(List<Item> known, List<Issue> issues) {
        List<Item> body = known.stream().filter(i -> BODY.contains(i.category())).toList();
        Set<String> others = new LinkedHashSet<>();
        known.stream().filter(i -> !BODY.contains(i.category())).forEach(i -> others.add(i.id()));
        List<Outfit> out = new ArrayList<>();
        for (List<Item> outfit : outfits(body)) {
            if (outfit.size() < 2) {
                continue;
            }
            Set<String> ids = new LinkedHashSet<>(others);
            outfit.forEach(i -> ids.add(i.id()));
            List<Issue> mine = issues.stream()
                    .filter(i -> i.rule().startsWith("occasion_") && ids.containsAll(i.garmentIds())).toList();
            out.add(one(outfit, mine));
        }
        out.sort(Comparator.comparingInt(Outfit::score).reversed());
        return out;
    }

    /** 종류마다 한 벌씩 고른 조합. 원피스는 상의·하의 대신 입습니다. 같은 종류가 두 벌이면 조합이 여러 개(비교 중). */
    static List<List<Item>> outfits(List<Item> body) {
        Map<String, List<Item>> by = new LinkedHashMap<>();
        for (Item i : body) {
            by.computeIfAbsent(i.category(), k -> new ArrayList<>()).add(i);
        }
        List<List<String>> shapes = new ArrayList<>();
        if (by.containsKey("top") || by.containsKey("bottom")) {
            shapes.add(List.of("top", "bottom", "outer"));
        }
        if (by.containsKey("dress")) {
            shapes.add(List.of("dress", "outer"));
        }
        if (shapes.isEmpty()) {
            shapes.add(List.of("outer"));
        }
        List<List<Item>> all = new ArrayList<>();
        for (List<String> shape : shapes) {
            List<List<Item>> partial = new ArrayList<>();
            partial.add(List.of());
            for (String category : shape) {
                List<Item> choices = by.get(category);
                if (choices == null) {
                    continue;
                }
                List<List<Item>> next = new ArrayList<>();
                for (List<Item> p : partial) {
                    for (Item c : choices) {
                        List<Item> grown = new ArrayList<>(p);
                        grown.add(c);
                        next.add(grown);
                    }
                }
                partial = next;
            }
            all.addAll(partial);
        }
        return all.size() > MAX_OUTFITS ? all.subList(0, MAX_OUTFITS) : all;
    }

    /** 한 벌 조합의 점수. */
    static Outfit one(List<Item> outfit, List<Issue> occasionIssues) {
        List<Part> parts = new ArrayList<>();
        Scheme s = scheme(outfit);
        parts.add(new Part(s.name, s.base));

        // 명도 대비: 밝은 옷과 어두운 옷이 같이 있으면 또렷해 보입니다. 강한 색끼리 부딪칠 때는 더하지 않습니다.
        boolean light = outfit.stream().anyMatch(i -> "light".equals(i.tone()) || "white".equals(i.colorFamily()));
        boolean dark = outfit.stream().anyMatch(i -> "dark".equals(i.tone()) || "black".equals(i.colorFamily()));
        if (light && dark && s.base >= 75) {
            parts.add(new Part("밝은 옷과 어두운 옷의 명도 대비", 5));
        }
        // 전부 같은 색(검정+검정)이면 단조로울 수 있음.
        if ("neutral".equals(s.code) && outfit.stream().map(ColorScore::colorKey).distinct().count() == 1) {
            parts.add(new Part("모두 같은 색이라 조금 단조로움", -5));
        }
        long patterned = outfit.stream().filter(i -> i.pattern() != null && !i.pattern().isBlank()
                && !"plain".equals(i.pattern())).count();
        if (patterned >= 2) {
            parts.add(new Part("무늬 있는 옷이 두 벌", -10));
        }
        for (Issue i : occasionIssues) {
            switch (i.level()) {
                case "bad" -> parts.add(new Part("자리에 맞지 않는 색", -35));
                case "note" -> parts.add(new Part("자리에 조금 튀는 색", -8));
                default -> {
                }
            }
        }
        int score = Math.max(0, Math.min(100, parts.stream().mapToInt(Part::points).sum()));
        return new Outfit(outfit.stream().map(Item::id).toList(), score, s.code, s.name, s.reason, List.copyOf(parts));
    }

    private record Scheme(String code, String name, int base, String reason) {
    }

    /**
     * 색 조합 종류. 강한 색(ColorHarmony.strong)의 개수와 색상환 거리(RYB, ColorHarmony.HUE)로 나눕니다.
     * 버건디·하늘색·청바지처럼 차분한 강한 색은 기본색처럼 보되, 강한 색과 반대 색이면 "보색 포인트"로 봅니다.
     */
    private static Scheme scheme(List<Item> outfit) {
        List<Item> strong = outfit.stream().filter(ColorHarmony::strong).toList();
        Set<String> strongFamilies = new LinkedHashSet<>();
        strong.forEach(i -> strongFamilies.add(i.colorFamily()));
        if (strongFamilies.size() >= 3) {
            return new Scheme("many_strong", "강한 색 세 가지", 35, "강한 색이 셋이라 복잡해 보여요.");
        }
        if (strongFamilies.size() == 2) {
            var it = strongFamilies.iterator();
            int d = ColorHarmony.hueDistance(it.next(), it.next());
            if (d >= ColorHarmony.OPPOSITE) {
                return new Scheme("complementary", "강한 보색끼리", 45, "강한 반대 색끼리라 서로 부딪혀요.");
            }
            if (d > ColorHarmony.NEAR) {
                return new Scheme("two_strong", "강한 색 둘", 65, "강한 색이 둘이라 조금 튈 수 있어요.");
            }
            return new Scheme("analogous", "유사색 조합", 85, "색상환에서 가까운 색끼리라 자연스럽게 이어져요.");
        }
        if (strongFamilies.size() == 1) {
            String family = strongFamilies.iterator().next();
            if (strong.size() >= 2) {
                return new Scheme("same_family", "같은 색 계열", 84, "같은 계열 색으로 맞춰서 통일감이 있어요.");
            }
            boolean opposite = outfit.stream()
                    .filter(i -> ColorHarmony.chromatic(i) && !ColorHarmony.strong(i))
                    .anyMatch(i -> ColorHarmony.hueDistance(family, i.colorFamily()) >= ColorHarmony.OPPOSITE);
            if (opposite) {
                return new Scheme("complementary_point", "차분한 보색 포인트", 90,
                        "반대 색이지만 한쪽이 차분한 색이라 포인트가 살아요.");
            }
            return new Scheme("one_point", "기본색에 포인트 색 하나", 92, "기본색에 강한 색 하나라 깔끔하면서 포인트가 있어요.");
        }
        return new Scheme("neutral", "기본색 조합", 85, "기본색끼리라 어디에나 무난하게 어울려요.");
    }

    private static String colorKey(Item i) {
        return i.colorFamily() + "/" + (i.tone() == null ? "" : i.tone());
    }
}
