package com.fashionagent.clothesstate.verify;

import static org.assertj.core.api.Assertions.assertThat;

import java.time.Duration;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

class OpenAiCropVerifierTest {

    private final OpenAiCropVerifier v = new OpenAiCropVerifier(new GptCheckProperties("", "gpt-4.1-mini",
            "http://localhost", Duration.ofSeconds(1), 0.3, 512, 64, 0, 0.3, 0.25, "", 30, Duration.ofSeconds(1)));

    @Test
    @DisplayName("키가 없으면 준비 안 됨, 아무것도 보내지 않음")
    void noKey() {
        assertThat(v.ready()).isFalse();
        assertThat(v.verify(new java.awt.image.BufferedImage(8, 8, 1), "stain")).isNull();
    }

    @Test
    @DisplayName("응답 JSON에서 label·confidence를 읽고, 모양이 다르면 null")
    void parse() {
        String ok = "{\"choices\":[{\"message\":{\"content\":\"{\\\"label\\\":\\\"Stain\\\",\\\"confidence\\\":0.86}\"}}]}";
        assertThat(v.parse(ok, "m")).isEqualTo(new CropVerifier.Verdict("stain", 0.86, "m"));
        String none = "{\"choices\":[{\"message\":{\"content\":\"{\\\"label\\\":\\\"none\\\",\\\"confidence\\\":0.7}\"}}]}";
        assertThat(v.parse(none, "m").defect(0)).isFalse();
        assertThat(v.parse("{\"choices\":[{\"message\":{\"content\":\"hello\"}}]}", "m")).isNull();
        assertThat(v.parse("{\"choices\":[{\"message\":{\"content\":\"{\\\"label\\\":\\\"dirt\\\"}\"}}]}", "m")).isNull();
    }

    @Test
    @DisplayName("요청: 사용자 메시지 하나에 시험과 같은 문장, 그림은 detail high")
    void request() {
        String s = v.request("AAAA").toString();
        assertThat(s).contains("gpt-4.1-mini", "json_object", "data:image/jpeg;base64,AAAA", "detail=high",
                "Printed patterns");
    }
}
