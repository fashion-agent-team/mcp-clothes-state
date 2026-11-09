package com.fashionagent.clothesstate.inspect;

import java.util.List;

/**
 * clothes_inspect 결과 (앱 레포 docs/contracts/mcp-tools.md). 옷 종류·박스는 옷 종류 모델로, 색은 박스 안 픽셀로
 * 정합니다. 격식(formality)은 아직 unknown, 무늬·소재는 null입니다.
 */
public record InspectResult(
        List<Garment> garments,
        String framing,
        String lighting,
        String source,
        String modelVersion,
        PendingHelper pendingHelper) {

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
            String pattern,
            String material,
            String notes,
            String formality,
            List<Defect> defects,
            Box box,
            String position,
            Double confidence) {
    }

    public record Candidate(String kind, String where, double confidence, Box box) {
    }

    public record PendingHelper(String garmentId, List<Candidate> candidates, String note) {
    }
}
