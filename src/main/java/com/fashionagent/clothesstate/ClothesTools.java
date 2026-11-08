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
            옷 사진 한 장에서 얼룩(stain)과 찢어짐·구멍(tear)을 찾습니다(우리가 학습한 모델을 이 서버에서 돌림, 사진은 외부로 \
            보내지 않고 저장하지 않음). 옷 추천 2단계에서 앱이 보낸 사진마다 부르세요. 확실한 것은 garments[].defects에 있고 \
            바로 알립니다. 애매한 것은 pendingHelper.candidates에 있으니 note를 읽고 "지인에게 물어볼까요?"라고 묻고, \
            사용자가 원할 때만 helper_ask로 넘깁니다. where는 사용자가 "어디야?"라고 물을 때만 말합니다. \
            lighting이 dark면 밝은 곳에서 다시 찍어 달라고 안내합니다.""")
    public Object clothesInspect(
            @ToolParam(description = "옷 사진 JPEG base64. 입은 사진도 좌우를 뒤집지 않고 원본 그대로. agent가 넣습니다.") String imageBase64,
            @ToolParam(required = false, description = "laid(내려놓은 옷을 위에서 찍음, 기본값) 또는 worn(옷을 입고 폰 앞에 섬)") String mode,
            @ToolParam(required = false, description = "mode=worn일 때만: front, side, back") String view) {
        try {
            return service.inspect(imageBase64, mode, view);
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
