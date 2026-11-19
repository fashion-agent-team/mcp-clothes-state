package com.fashionagent.clothesstate.inspect;

import static org.assertj.core.api.Assertions.assertThat;

import com.fashionagent.clothesstate.detect.Detection;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

class WhereTest {

    private static Detection at(double cx, double cy) {
        return new Detection("stain", 0.9, cx - 0.05, cy - 0.05, 0.1, 0.1);
    }

    @Test
    @DisplayName("laid: 사진 기준(사진 왼쪽 = 사용자 왼쪽)")
    void laid() {
        assertThat(Where.of(at(0.1, 0.1), "laid", null)).isEqualTo("옷 왼쪽 위쪽");
        assertThat(Where.of(at(0.9, 0.5), "laid", null)).isEqualTo("옷 오른쪽");
        assertThat(Where.of(at(0.5, 0.9), "laid", null)).isEqualTo("옷 아래쪽");
        assertThat(Where.of(at(0.5, 0.5), "laid", null)).isEqualTo("옷 가운데");
    }

    @Test
    @DisplayName("worn 앞모습: 사진 왼쪽은 입은 사람의 오른쪽")
    void wornFront() {
        assertThat(Where.of(at(0.1, 0.2), "worn", "front")).isEqualTo("몸 오른쪽 위쪽");
        assertThat(Where.of(at(0.9, 0.5), "worn", null)).isEqualTo("몸 왼쪽");
    }

    @Test
    @DisplayName("worn 뒷모습은 사진과 같은 방향, 옆모습은 왼쪽·오른쪽을 말하지 않음")
    void wornBackSide() {
        assertThat(Where.of(at(0.1, 0.5), "worn", "back")).isEqualTo("등 왼쪽");
        assertThat(Where.of(at(0.1, 0.9), "worn", "side")).isEqualTo("옆 아래쪽");
    }
}
