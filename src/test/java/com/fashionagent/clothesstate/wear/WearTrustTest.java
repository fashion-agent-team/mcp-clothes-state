package com.fashionagent.clothesstate.wear;

import static org.assertj.core.api.Assertions.assertThat;

import java.nio.file.Path;
import java.util.Map;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class WearTrustTest {

    @TempDir
    Path tmp;

    static WearLearnProperties props() {
        return new WearLearnProperties(true, 200, 50, 10, 0.95, 0.9, 0.1, 0.9, 0.05, 0.1, "");
    }

    /** 정상 negatives개(모델 0.02), 실수 positives개(모델 0.97)를 GPT 답으로. */
    static void fill(WearTrust t, String kind, int negatives, int positives) {
        for (int i = 0; i < negatives; i++) {
            t.record(kind, 0.02, false);
        }
        for (int i = 0; i < positives; i++) {
            t.record(kind, 0.97, true);
        }
    }

    @Test
    @DisplayName("GPT 답 50개·실수 10개 이상이고 정상 95%·실수 90% 이상 맞혀야 믿음")
    void trusted() {
        var t = new WearTrust(props(), null, "v", () -> 0.5);
        fill(t, "buttons_misaligned", 45, 4);
        assertThat(t.stats("buttons_misaligned").trusted()).isFalse(); // 49개, 실수 4개
        fill(t, "buttons_misaligned", 0, 6);
        assertThat(t.stats("buttons_misaligned").trusted()).isTrue();
        assertThat(t.anyTrusted()).isTrue();
        // GPT는 실수라는데 모델이 놓친 게 많으면(지금 실제 사진처럼) 믿지 않음
        for (int i = 0; i < 5; i++) {
            t.record("buttons_misaligned", 0.01, true);
        }
        assertThat(t.stats("buttons_misaligned").recall()).isLessThan(0.9);
        assertThat(t.stats("buttons_misaligned").trusted()).isFalse();
    }

    @Test
    @DisplayName("건너뛰기: 볼 수 있는 실수가 모두 믿을 만하고 모델이 확신할 때만, 그래도 5%는 물음")
    void canSkip() {
        double[] r = {0.5};
        var t = new WearTrust(props(), null, "v", () -> r[0]);
        fill(t, "buttons_misaligned", 60, 12);
        fill(t, "inside_out", 60, 12);
        assertThat(t.canSkip(Map.of("buttons_misaligned", 0.03, "inside_out", 0.95))).isTrue();
        assertThat(t.canSkip(Map.of("buttons_misaligned", 0.4))).isFalse(); // 애매함
        assertThat(t.canSkip(Map.of("buttons_misaligned", 0.03, "tag_out", 0.0))).isFalse(); // 택은 아직 못 믿음
        assertThat(t.canSkip(Map.of())).isFalse();
        r[0] = 0.01;
        assertThat(t.canSkip(Map.of("buttons_misaligned", 0.03))).isFalse(); // 감시용으로 물음
    }

    @Test
    @DisplayName("저장: GPT가 실수라 함·엇갈림·애매함은 늘, 둘 다 확신하는 정상은 10%만")
    void worthSaving() {
        double[] r = {0.5};
        var t = new WearTrust(props(), null, "v", () -> r[0]);
        assertThat(t.worthSaving(0.01, true)).isTrue();
        assertThat(t.worthSaving(0.95, false)).isTrue();
        assertThat(t.worthSaving(0.3, false)).isTrue();
        assertThat(t.worthSaving(0.01, false)).isFalse();
        r[0] = 0.05;
        assertThat(t.worthSaving(0.01, false)).isTrue();
    }

    @Test
    @DisplayName("기록은 파일로 남고, 모델 버전이 바뀌면 처음부터")
    void persist() {
        Path f = tmp.resolve("wear-trust.json");
        var t = new WearTrust(props(), f, "wear3", () -> 0.5);
        fill(t, "buttons_misaligned", 50, 10);
        t.save();
        assertThat(new WearTrust(props(), f, "wear3", () -> 0.5).stats("buttons_misaligned").samples()).isEqualTo(60);
        assertThat(new WearTrust(props(), f, "wear4", () -> 0.5).stats("buttons_misaligned").samples()).isZero();
    }

    @Test
    @DisplayName("창 크기(200)보다 많으면 오래된 것부터 버림, 꺼져 있으면 아무것도 안 함")
    void windowAndOff() {
        var t = new WearTrust(props(), null, "v", () -> 0.5);
        fill(t, "inside_out", 250, 0);
        assertThat(t.stats("inside_out").samples()).isEqualTo(200);
        var off = new WearTrust(new WearLearnProperties(false, 200, 50, 10, 0.95, 0.9, 0.1, 0.9, 0.05, 0.1, ""), null,
                "v", () -> 0.5);
        fill(off, "inside_out", 60, 12);
        assertThat(off.stats("inside_out").samples()).isZero();
        assertThat(off.canSkip(Map.of("inside_out", 0.0))).isFalse();
        assertThat(off.worthSaving(0.01, true)).isFalse();
    }
}
