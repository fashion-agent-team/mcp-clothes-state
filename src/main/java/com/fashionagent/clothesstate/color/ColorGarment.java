package com.fashionagent.clothesstate.color;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import java.util.List;
import java.util.Map;
import org.springframework.ai.tool.annotation.ToolParam;

/**
 * outfit_color_check에 넘기는 옷 한 벌. clothes_inspect의 Garment를 그대로 넘길 수 있게 필드를 모두 둡니다
 * (MCP가 입력 JSON을 스키마로 검사해서, 스키마에 없는 필드가 오면 도구를 부르지 못함). 판단에는 색·종류·무늬만 씁니다.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public record ColorGarment(
        @ToolParam(description = "clothes_inspect가 붙인 id. 예: g1") String id,
        @ToolParam(description = "top | bottom | outer | dress | shoes | other") String category,
        @ToolParam(required = false, description = "옷 이름. 예: 셔츠") String name,
        @ToolParam(description = "black white gray navy beige brown khaki blue red pink orange yellow green purple unknown") String colorFamily,
        @ToolParam(required = false, description = "색 이름. 예: 하늘색, 차콜 그레이") String colorName,
        @ToolParam(required = false, description = "light | mid | dark") String tone,
        @ToolParam(required = false, description = "무늬. 없으면 null") String pattern,
        @ToolParam(required = false, description = "소재 추정") String material,
        @ToolParam(required = false, description = "기타 메모") String notes,
        @ToolParam(required = false, description = "formal | smart | casual | sporty | unknown") String formality,
        @ToolParam(required = false, description = "얼룩·찢어짐(판단에 쓰지 않음)") List<Map<String, Object>> defects,
        @ToolParam(required = false, description = "사진 속 위치(판단에 쓰지 않음)") Map<String, Object> box,
        @ToolParam(required = false, description = "left | center | right") String position,
        @ToolParam(required = false, description = "0~1") Double confidence) {

    public ColorHarmony.Item item(int index) {
        return new ColorHarmony.Item(id == null || id.isBlank() ? "g" + (index + 1) : id, category, name,
                colorFamily, colorName, tone, pattern);
    }
}
