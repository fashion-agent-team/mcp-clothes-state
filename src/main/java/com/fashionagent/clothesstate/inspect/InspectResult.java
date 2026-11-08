package com.fashionagent.clothesstate.inspect;

import java.util.List;

/**
 * clothes_inspect 결과 (앱 레포 docs/contracts/mcp-tools.md). 색은 픽셀로 정하고(GarmentColor), 옷 종류·격식은
 * 아직 채우지 않습니다(category other, formality unknown). 사진 전체를 옷 한 벌(g1)로 봅니다.
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

    public record Defect(String kind, String where, double confidence) {
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
