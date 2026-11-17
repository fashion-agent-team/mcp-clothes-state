package com.fashionagent.clothesstate;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.Map;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

class OrtSupportTest {

    @Test
    @DisplayName("ONNX Runtime 설정: 기본은 arena·memPattern 끄고 prepack 켬, 환경 변수로 바꾸고 잘못된 숫자는 기본값")
    void settings() {
        var d = OrtSupport.Settings.from(Map.of());
        assertThat(d).isEqualTo(new OrtSupport.Settings(false, false, true, 0, false));
        var s = OrtSupport.Settings.from(Map.of("CLOTHES_ORT_ARENA", "true", "CLOTHES_ORT_PREPACK", "false",
                "CLOTHES_ORT_THREADS", "4", "CLOTHES_ORT_SHARED_THREADS", "TRUE"));
        assertThat(s).isEqualTo(new OrtSupport.Settings(true, false, false, 4, true));
        assertThat(OrtSupport.Settings.from(Map.of("CLOTHES_ORT_THREADS", "four")).threads()).isZero();
    }
}
