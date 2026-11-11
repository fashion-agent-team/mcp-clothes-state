package com.fashionagent.clothesstate;

import com.fashionagent.clothesstate.inspect.ClothesInspectService;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.tool.annotation.Tool;
import org.springframework.ai.tool.annotation.ToolParam;
import org.springframework.stereotype.Component;

/**
 * MCP 도구 clothes_inspect (앱 레포 docs/contracts/mcp-tools.md). 옷 추천 2단계에서 agent가 사진마다 직접 부릅니다
 * (GPT가 고르지 않음). 실패해도 예외를 던지지 않고 {"error": {"code", "message"}}를 돌려줍니다.
 */
@Component
public class ClothesTools {

    private static final Logger log = LoggerFactory.getLogger(ClothesTools.class);

    private final ClothesInspectService service;

    public ClothesTools(ClothesInspectService service) {
        this.service = service;
    }

    @Tool(name = "clothes_inspect", description = """
            옷 사진 한 장에서 옷 한 벌마다 종류(category: top, bottom, outer, dress, shoes, other)와 색(colorName), \
            무늬(pattern: plain, stripe, check, floral, dot, graphic, animal, other, 모르면 null), \
            얼룩(stain)과 찢어짐·구멍(tear)을 찾습니다(우리가 학습한 모델을 이 서버에서 돌림). 애매한 결함과 무늬를 모르는 옷은 \
            allowGpt=true(사용자 동의)일 때만 서버가 그 부분(무늬는 내려놓은 사진만)을 GPT에게 물어 확인하고, 찍은 사진 전체는 \
            외부로 보내지 않습니다. patternCheckedBy가 gpt면 무늬도 조금 약하게 말합니다. 옷 추천 2단계에서 앱이 보낸 사진마다 \
            부르세요. garments[].defects는 바로 알립니다(checkedBy가 gpt면 "~인 것 같아요"처럼 조금 약하게). \
            pendingHelper가 있으면 확인하지 못한 것이니 "확실하지 않은 부분이 있어요"라고만 말합니다. \
            where는 사용자가 "어디야?"라고 물을 때만 말합니다. \
            lighting이 dark면 밝은 곳에서 다시 찍어 달라고 안내합니다.""")
    public Object clothesInspect(
            @ToolParam(description = "옷 사진 JPEG base64. 입은 사진도 좌우를 뒤집지 않고 원본 그대로. agent가 넣습니다.") String imageBase64,
            @ToolParam(required = false, description = "laid(내려놓은 옷을 위에서 찍음, 기본값) 또는 worn(옷을 입고 폰 앞에 섬)") String mode,
            @ToolParam(required = false, description = "mode=worn일 때만: front, side, back") String view,
            @ToolParam(required = false, description = "사용자가 GPT 확인에 동의했으면 true. 그때만 애매한 부분 조각을 GPT에게 물음. 기본 false") Boolean allowGpt) {
        try {
            return service.inspect(imageBase64, mode, view, allowGpt);
        } catch (ClothesStateException e) {
            return error(e.code(), e.getMessage());
        } catch (RuntimeException e) {
            log.warn("clothes_inspect 실패: {}", e.toString());
            return error("upstream_failed", "지금은 옷 상태를 확인하지 못했어요.");
        }
    }

    static Map<String, Object> error(String code, String message) {
        return Map.of("error", Map.of("code", code, "message", message));
    }
}
