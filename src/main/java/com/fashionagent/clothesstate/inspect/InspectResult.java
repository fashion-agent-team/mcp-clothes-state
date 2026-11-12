package com.fashionagent.clothesstate.inspect;

import java.util.List;

/**
 * clothes_inspect 결과 (앱 레포 docs/contracts/mcp-tools.md). 옷 종류·박스는 옷 종류 모델로, 색은 박스 안 픽셀로
 * 정합니다. 무늬(pattern)는 무늬 모델로 정하고 모르면 null, 격식(formality)은 아직 unknown, 소재는 null입니다.
 * embedding은 옷 박스의 이미지 벡터이고 사진은 저장하지 않습니다(이슈 #15).
 */
public record InspectResult(
        List<Garment> garments,
        String framing,
        String lighting,
        String source,
        String modelVersion,
        PendingHelper pendingHelper,
        /** garments[].embedding을 만든 모델(예: marqo-fashionSigLIP). 벡터를 안 만들었으면 null. 옷장은 같은 값끼리만 비교. */
        String embeddingModel) {

    public record Box(double x, double y, double w, double h) {
    }

    /** checkedBy: model(우리 모델 확신도 tell 이상) 또는 gpt(애매해서 결함 조각만 GPT에게 물어 확인함). */
    public record Defect(String kind, String where, double confidence, String checkedBy) {
    }

    public record Garment(
            String id,
            String category,
            String name,
            String colorName,
            String colorFamily,
            String tone,
            /**
             * 무늬 옷(pattern이 plain·null이 아님)의 두 번째 색. 체크·줄무늬·호피처럼 색이 섞인 옷을 "네이비·베이지 체크 셔츠"로
             * 읽어 주려고 넣습니다. 옷 픽셀에서 colorName과 다른 계열의 큰 색 무리가 없으면 null. (2026-11-12 체크 셔츠를 "검정 셔츠"라 함)
             */
            String subColorName,
            String subColorFamily,
            /** plain·stripe·check·floral·dot·graphic·animal·other, 모르면 null. */
            String pattern,
            /** pattern을 정한 쪽: model 또는 gpt(모델이 몰라서 옷 박스를 GPT에게 물음). pattern이 null이면 null. */
            String patternCheckedBy,
            /** pattern이 null(확률 0.7 미만)인데 모델 1등이 무늬면 그 무늬(확률 0.35 이상). "~무늬 같아요"로만 말할 것. */
            String patternGuess,
            String material,
            String notes,
            String formality,
            List<Defect> defects,
            Box box,
            String position,
            Double confidence,
            /** 옷 박스 이미지 벡터(768개, 길이 1). 옷장이 같은 옷인지 판단할 때만 씀. 모델이 없으면 null. */
            float[] embedding) {
    }

    public record Candidate(String kind, String where, double confidence, Box box) {
    }

    public record PendingHelper(String garmentId, List<Candidate> candidates, String note) {
    }
}
