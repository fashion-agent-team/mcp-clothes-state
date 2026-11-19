package com.fashionagent.clothesstate.wear;

import static org.assertj.core.api.Assertions.assertThat;

import com.fashionagent.clothesstate.verify.GptCheckProperties;
import java.time.Duration;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

class OpenAiWearCheckerTest {

    private static GptCheckProperties gpt(String key) {
        return new GptCheckProperties(key, "gpt-4.1-mini", "http://localhost:1", Duration.ofSeconds(1), 0.3, 512, 64, 0,
                0.3, 0.25, "", 30, Duration.ofSeconds(1));
    }

    private static WearCheckProperties wear(boolean enabled, String model) {
        return new WearCheckProperties(enabled, model, Duration.ofSeconds(1), 768, 0.7, 0.25, 0.12, 3);
    }

    @Test
    @DisplayName("키가 없거나 꺼져 있으면 준비 안 됨, 아무것도 보내지 않음")
    void notReady() {
        assertThat(new OpenAiWearChecker(gpt(""), wear(true, "")).ready()).isFalse();
        assertThat(new OpenAiWearChecker(gpt("k"), wear(false, "")).ready()).isFalse();
        assertThat(new OpenAiWearChecker(gpt(""), wear(true, "")).check(new java.awt.image.BufferedImage(8, 8, 1),
                "shirt", "front")).isNull();
    }

    @Test
    @DisplayName("응답에서 아는 종류만 읽고, 같은 종류는 확신도 높은 하나, 모양이 다르면 null")
    void parse() {
        String ok = "{\"choices\":[{\"message\":{\"content\":\"{\\\"issues\\\":[{\\\"kind\\\":\\\"Tag_Out\\\",\\\"confidence\\\":0.8},"
                + "{\\\"kind\\\":\\\"tag_out\\\",\\\"confidence\\\":0.9},{\\\"kind\\\":\\\"wrinkle\\\",\\\"confidence\\\":0.9}]}\"}}]}";
        assertThat(OpenAiWearChecker.parse(ok, "m")).containsExactly(new WearChecker.Issue("tag_out", 0.9, "m"));
        String none = "{\"choices\":[{\"message\":{\"content\":\"{\\\"issues\\\":[]}\"}}]}";
        assertThat(OpenAiWearChecker.parse(none, "m")).isEmpty();
        assertThat(OpenAiWearChecker.parse("{\"choices\":[{\"message\":{\"content\":\"hello\"}}]}", "m")).isNull();
        assertThat(OpenAiWearChecker.parse("{\"choices\":[{\"message\":{\"content\":\"{}\"}}]}", "m")).isNull();
    }

    @Test
    @DisplayName("요청: 옷 이름과 방향이 문장에 들어가고, 모델을 따로 안 정하면 clothes.gpt.model")
    void request() {
        String s = new OpenAiWearChecker(gpt("k"), wear(true, "")).request("AAAA", "shirt", "back").toString();
        assertThat(s).contains("gpt-4.1-mini", "json_object", "data:image/jpeg;base64,AAAA", "detail=high",
                "wearing a shirt, seen from the back", "visible from the back", "buttons_misaligned");
        assertThat(new OpenAiWearChecker(gpt("k"), wear(true, "gpt-4.1")).model()).isEqualTo("gpt-4.1");
        assertThat(OpenAiWearChecker.prompt(null, "front")).contains("wearing a garment, seen from the front",
                "zipper/fly at the back");
    }
}
