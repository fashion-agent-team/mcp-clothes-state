package com.fashionagent.clothesstate;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.BDDMockito.given;
import static org.mockito.BDDMockito.then;
import static org.mockito.Mockito.mock;

import com.fashionagent.clothesstate.color.ColorGarment;
import com.fashionagent.clothesstate.color.ColorHarmony;
import com.fashionagent.clothesstate.inspect.ClothesInspectService;
import com.fashionagent.clothesstate.inspect.InspectResult;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * MCP 도구 계층 테스트. REST 서버의 컨트롤러 테스트(@WebMvcTest)와 같은 자리입니다.
 * 옷 상태 서비스는 Mock으로 바꾸고(정상 동작한다고 가정), 도구가 서비스에 값을 그대로 넘기는지와
 * 돌려주는 결과·오류 코드만 봅니다. 진짜 서버로 끝까지 보는 건 McpServerTest입니다.
 *
 * <p>MCP에는 HTTP 상태 코드가 없어서, 실패는 {"error": {"code", "message"}}로 돌려줍니다
 * (bad_request ≈ 400, upstream_failed ≈ 502).
 */
class ClothesToolsTest {

    private ClothesInspectService service;
    private ClothesTools tools;

    @BeforeEach
    void setUp() {
        service = mock(ClothesInspectService.class);
        tools = new ClothesTools(service);
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> errorOf(Object result) {
        assertThat(result).isInstanceOf(Map.class);
        return (Map<String, Object>) ((Map<String, Object>) result).get("error");
    }

    private static ColorGarment garment(String id, String category, String colorFamily) {
        return new ColorGarment(id, category, null, colorFamily, null, null, null, null, null, null, null, null, null,
                null);
    }

    @Test
    @DisplayName("clothes_inspect Given 입은 사진과 GPT 동의 When 부르면 Then mode·view·allowGpt를 그대로 넘기고 서비스 결과를 돌려준다")
    void inspectOk() {
        // given
        InspectResult found = mock(InspectResult.class);
        given(service.inspect("jpegBase64", "worn", "front", true)).willReturn(found);

        // when
        Object result = tools.clothesInspect("jpegBase64", "worn", "front", true);

        // then
        assertThat(result).isSameAs(found);
        then(service).should().inspect("jpegBase64", "worn", "front", true);
    }

    @Test
    @DisplayName("clothes_inspect Given 선택 값을 비움 When 부르면 Then null 그대로 넘긴다(기본값은 서비스가 정함)")
    void inspectDefaultsLeftToService() {
        // given
        InspectResult found = mock(InspectResult.class);
        given(service.inspect("jpegBase64", null, null, null)).willReturn(found);

        // when
        Object result = tools.clothesInspect("jpegBase64", null, null, null);

        // then
        assertThat(result).isSameAs(found);
    }

    @Test
    @DisplayName("clothes_inspect Given 서비스가 bad_image를 던짐 When 부르면 Then 예외 없이 같은 code와 한국어 message를 돌려준다")
    void inspectKnownError() {
        // given
        given(service.inspect(any(), any(), any(), any()))
                .willThrow(new ClothesStateException("bad_image", "사진을 읽지 못했어요."));

        // when
        Object result = tools.clothesInspect("not-an-image", null, null, false);

        // then
        assertThat(errorOf(result)).containsEntry("code", "bad_image").containsEntry("message", "사진을 읽지 못했어요.");
    }

    @Test
    @DisplayName("clothes_inspect Given 모델이 알 수 없는 예외 When 부르면 Then upstream_failed로 감싸고 내부 메시지는 숨긴다")
    void inspectUnexpectedError() {
        // given
        given(service.inspect(any(), any(), any(), any())).willThrow(new IllegalStateException("onnx session closed"));

        // when
        Object result = tools.clothesInspect("jpegBase64", "laid", null, false);

        // then
        Map<String, Object> error = errorOf(result);
        assertThat(error).containsEntry("code", "upstream_failed");
        assertThat(error.get("message").toString()).doesNotContain("onnx");
    }

    @Test
    @DisplayName("outfit_color_check Given 옷 목록이 비었거나 null When 부르면 Then no_garments를 돌려준다")
    void colorNoGarments() {
        assertThat(errorOf(tools.outfitColorCheck(List.of(), null, null))).containsEntry("code", "no_garments");
        assertThat(errorOf(tools.outfitColorCheck(null, null, null))).containsEntry("code", "no_garments");
        then(service).shouldHaveNoInteractions();
    }

    @Test
    @DisplayName("outfit_color_check Given 옷 11벌 When 부르면 Then bad_request(10벌까지)")
    void colorTooMany() {
        // given
        List<ColorGarment> eleven = Collections.nCopies(11, garment(null, "top", "white"));

        // when
        Object result = tools.outfitColorCheck(eleven, "casual", "ok");

        // then
        assertThat(errorOf(result)).containsEntry("code", "bad_request");
    }

    @Test
    @DisplayName("outfit_color_check Given 모르는 자리(party) When 부르면 Then bad_request")
    void colorUnknownOccasion() {
        Object result = tools.outfitColorCheck(List.of(garment("g1", "top", "white")), "party", null);

        assertThat(errorOf(result)).containsEntry("code", "bad_request");
    }

    @Test
    @DisplayName("outfit_color_check Given 흰 셔츠와 남색 바지, 자리 비움 When 부르면 Then casual로 보고 색 판정 결과를 돌려준다")
    void colorOkDefaultsCasual() {
        // given
        List<ColorGarment> outfit = List.of(garment("g1", "top", "white"), garment("g2", "bottom", "navy"));

        // when
        Object result = tools.outfitColorCheck(outfit, " ", "ok");

        // then
        assertThat(result).isInstanceOf(ColorHarmony.Result.class)
                .isEqualTo(tools.outfitColorCheck(outfit, "casual", "ok"));
        assertThat(((ColorHarmony.Result) result).verdict()).isNotBlank();
        then(service).shouldHaveNoInteractions();
    }

    @Test
    @DisplayName("outfit_color_check Given id가 빈 옷 When 부르면 Then 순서대로 g1·g2 id를 붙여 판단한다")
    void colorFillsMissingIds() {
        // given 장례식에 빨간 상의 → 바꿀 옷 id가 나와야 함
        List<ColorGarment> outfit = List.of(garment(null, "top", "red"), garment(null, "bottom", "black"));

        // when
        ColorHarmony.Result result = (ColorHarmony.Result) tools.outfitColorCheck(outfit, "mourning", "ok");

        // then
        assertThat(result.verdict()).isEqualTo("bad");
        assertThat(result.issues()).isNotEmpty();
        assertThat(result.issues().get(0).swapGarmentId()).isEqualTo("g1");
    }
}
