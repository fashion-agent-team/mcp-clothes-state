package com.fashionagent.clothesstate.color;

import static org.assertj.core.api.Assertions.assertThat;

import com.fashionagent.clothesstate.color.ColorHarmony.Item;
import com.fashionagent.clothesstate.color.ColorHarmony.Result;
import java.util.List;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/** 어울림 점수(ColorScore). 판단(verdict)과 점수 구간이 서로 맞는지도 봅니다: good ≥ 80, bad < 60. */
class ColorScoreTest {

    private static Item top(String id, String family, String name, String tone) {
        return new Item(id, "top", "니트", family, name, tone, null);
    }

    private static Item pants(String id, String family, String name, String tone) {
        return new Item(id, "bottom", "바지", family, name, tone, null);
    }

    private static Result check(String occasion, Item... items) {
        return ColorHarmony.check(List.of(items), occasion, "ok");
    }

    @Test
    @DisplayName("흰 니트 + 검정 바지: 기본색 85 + 명도 대비 5 = 90")
    void neutralContrast() {
        Result r = check("casual", top("g1", "white", "흰색", "light"), pants("g2", "black", "검정", "dark"));
        assertThat(r.score()).isEqualTo(90);
        assertThat(r.scheme()).isEqualTo("기본색 조합");
        assertThat(r.outfits().get(0).parts()).extracting(ColorScore.Part::points).containsExactly(85, 5);
        assertThat(r.verdict()).isEqualTo("good");
    }

    @Test
    @DisplayName("빨강 니트 + 회색 바지: 기본색에 포인트 색 하나 92")
    void onePoint() {
        Result r = check("casual", top("g1", "red", "빨강", "mid"), pants("g2", "gray", "회색", "mid"));
        assertThat(r.score()).isEqualTo(92);
        assertThat(r.scheme()).isEqualTo("기본색에 포인트 색 하나");
        assertThat(r.scoreReason()).contains("포인트");
    }

    @Test
    @DisplayName("주황 니트 + 청바지: 반대 색이지만 청바지가 차분해서 보색 포인트 90")
    void complementaryPoint() {
        Result r = check("casual", top("g1", "orange", "주황", "mid"), pants("g2", "blue", "청색", "mid"));
        assertThat(r.score()).isEqualTo(90);
        assertThat(r.outfits().get(0).scheme()).isEqualTo("complementary_point");
        assertThat(r.verdict()).isEqualTo("good");
    }

    @Test
    @DisplayName("빨강 니트 + 초록 바지: 강한 보색끼리 45, verdict bad와 맞음")
    void complementaryClash() {
        Result r = check("casual", top("g1", "red", "빨강", "mid"), pants("g2", "green", "초록", "mid"));
        assertThat(r.score()).isEqualTo(45).isLessThan(ColorScore.OK);
        assertThat(r.verdict()).isEqualTo("bad");
    }

    @Test
    @DisplayName("빨강 니트 + 주황 바지: 유사색 85")
    void analogous() {
        Result r = check("casual", top("g1", "red", "빨강", "mid"), pants("g2", "orange", "주황", "mid"));
        assertThat(r.score()).isEqualTo(85);
        assertThat(r.scheme()).isEqualTo("유사색 조합");
    }

    @Test
    @DisplayName("면접에 빨강 니트 + 검정 바지: 포인트 색 92 - 자리 35 = 57")
    void occasionPenalty() {
        Result r = check("formal", top("g1", "red", "빨강", "mid"), pants("g2", "black", "검정", "dark"));
        assertThat(r.score()).isEqualTo(57).isLessThan(ColorScore.OK);
        assertThat(r.verdict()).isEqualTo("bad");
    }

    @Test
    @DisplayName("검정 니트 + 검정 바지: 모두 같은 색이라 85 - 5 = 80")
    void allSame() {
        Result r = check("casual", top("g1", "black", "검정", "dark"), pants("g2", "black", "검정", "dark"));
        assertThat(r.score()).isEqualTo(80);
    }

    @Test
    @DisplayName("옷이 한 벌뿐이면 점수 없음")
    void single() {
        Result r = check("casual", top("g1", "red", "빨강", "mid"));
        assertThat(r.score()).isNull();
        assertThat(r.outfits()).isEmpty();
    }

    @Test
    @DisplayName("상의 두 벌 비교: 조합마다 점수, 높은 순(회색 바지에 흰 니트 > 초록 니트)")
    void comparison() {
        Result r = check("casual", top("g1", "green", "초록", "mid"), top("g2", "white", "흰색", "light"),
                pants("g3", "red", "빨강", "mid"));
        assertThat(r.outfits()).hasSize(2);
        assertThat(r.outfits().get(0).garmentIds()).containsExactly("g2", "g3");
        assertThat(r.outfits().get(1).garmentIds()).containsExactly("g1", "g3");
        assertThat(r.score()).isEqualTo(r.outfits().get(0).score());
    }

    @Test
    @DisplayName("같은 옷이면 늘 같은 점수")
    void deterministic() {
        Item a = top("g1", "yellow", "노랑", "mid");
        Item b = pants("g2", "navy", "네이비", "dark");
        assertThat(check("casual", a, b).score()).isEqualTo(check("casual", a, b).score());
    }
}
