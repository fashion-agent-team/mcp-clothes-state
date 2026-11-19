package com.fashionagent.clothesstate.pattern;

import static org.assertj.core.api.Assertions.assertThat;

import com.fashionagent.clothesstate.verify.GptCheckProperties;
import java.awt.image.BufferedImage;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

class OpenAiPatternVerifierTest {

    private static GptCheckProperties props(String key) {
        return new GptCheckProperties(key, "gpt-4.1-mini", "http://localhost:1", Duration.ofSeconds(1), 0.3, 512, 64, 0,
                0.3, 0.25, "", 30, Duration.ofSeconds(1));
    }

    /** OpenAI 응답 모양으로 감싸기. */
    private static String reply(String content) throws Exception {
        var json = tools.jackson.databind.json.JsonMapper.builder().build();
        return json.writeValueAsString(Map.of("choices", List.of(Map.of("message", Map.of("content", content)))));
    }

    @Test
    @DisplayName("응답의 pattern이 8종 중 하나면 그 이름, 아니면 null")
    void parse() throws Exception {
        var v = new OpenAiPatternVerifier(props("k"));
        assertThat(v.parse(reply("{\"pattern\": \"Check\", \"confidence\": 0.8}")))
                .isEqualTo(new PatternVerifier.Answer("check", 0.8, "gpt-4.1-mini"));
        assertThat(v.parse(reply("{\"pattern\": \"tartan\"}"))).isNull();
        assertThat(v.parse(reply("not json"))).isNull();
    }

    @Test
    @DisplayName("요청: 공개 시험과 같은 문장, detail low, 키가 없으면 보내지 않음")
    @SuppressWarnings("unchecked")
    void request() {
        var v = new OpenAiPatternVerifier(props("k"));
        Map<String, Object> body = v.request("AAAA");
        assertThat(body.get("model")).isEqualTo("gpt-4.1-mini");
        var content = (List<Map<String, Object>>) ((List<Map<String, Object>>) body.get("messages")).getFirst().get("content");
        assertThat(content.getFirst().get("text")).isEqualTo(OpenAiPatternVerifier.PROMPT);
        assertThat(((Map<String, Object>) content.get(1).get("image_url")).get("detail")).isEqualTo("low");
        assertThat(new OpenAiPatternVerifier(props("")).ask(new BufferedImage(10, 10, BufferedImage.TYPE_INT_RGB)))
                .isNull();
    }

    @Test
    @DisplayName("긴 변 512로 줄임")
    void shrink() {
        var out = OpenAiPatternVerifier.shrink(new BufferedImage(1024, 600, BufferedImage.TYPE_INT_RGB), 512);
        assertThat(out.getWidth()).isEqualTo(512);
        assertThat(out.getHeight()).isEqualTo(300);
    }
}
