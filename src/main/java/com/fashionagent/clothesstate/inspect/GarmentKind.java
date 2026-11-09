package com.fashionagent.clothesstate.inspect;

import java.util.Map;

/**
 * 옷 종류 모델의 클래스 이름 → 약속 문서 category·name, 그리고 이름으로 정하는 격식·소재.
 * 격식은 이름만으로 분명한 것만 정하고(셔츠·코트 smart, 티셔츠·반바지 casual) 나머지는 unknown으로 둬서
 * agent(GPT)가 고르게 합니다. 옷장(mcp-closet)은 unknown을 빼지 않습니다.
 * 5종 모델(top·bottom·outer·dress·shoes)과 세부 12종 모델 이름을 모두 받습니다.
 */
record GarmentKind(String category, String name, String formality, String material) {

    private static final Map<String, GarmentKind> KINDS = Map.ofEntries(
            // 세부 12종 (Fashionpedia 2026-11-09 재학습)
            Map.entry("shirt", new GarmentKind("top", "셔츠", "smart", null)),
            Map.entry("tshirt", new GarmentKind("top", "티셔츠", "casual", null)),
            Map.entry("sweater", new GarmentKind("top", "니트", "unknown", "니트")),
            Map.entry("cardigan", new GarmentKind("outer", "가디건", "unknown", "니트")),
            Map.entry("jacket", new GarmentKind("outer", "자켓", "unknown", null)),
            Map.entry("coat", new GarmentKind("outer", "코트", "smart", null)),
            Map.entry("vest", new GarmentKind("outer", "조끼", "unknown", null)),
            Map.entry("pants", new GarmentKind("bottom", "바지", "unknown", null)),
            Map.entry("shorts", new GarmentKind("bottom", "반바지", "casual", null)),
            Map.entry("skirt", new GarmentKind("bottom", "치마", "unknown", null)),
            Map.entry("dress", new GarmentKind("dress", "원피스", "unknown", null)),
            Map.entry("shoes", new GarmentKind("shoes", "신발", "unknown", null)),
            // 5종 모델
            Map.entry("top", new GarmentKind("top", "상의", "unknown", null)),
            Map.entry("bottom", new GarmentKind("bottom", "하의", "unknown", null)),
            Map.entry("outer", new GarmentKind("outer", "아우터", "unknown", null)));

    static final GarmentKind OTHER = new GarmentKind("other", "옷", "unknown", null);

    static GarmentKind of(String modelClass) {
        return KINDS.getOrDefault(modelClass, OTHER);
    }
}
