package com.fashionagent.clothesstate.color;

import static org.assertj.core.api.Assertions.assertThat;

import com.fashionagent.clothesstate.color.ColorHarmony.Item;
import com.fashionagent.clothesstate.color.ColorHarmony.Result;
import java.util.List;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

class ColorHarmonyTest {

    private static Item top(String id, String family, String name, String tone) {
        return new Item(id, "top", "니트", family, name, tone, null);
    }

    private static Item shirt(String id, String family, String name, String tone) {
        return new Item(id, "top", "셔츠", family, name, tone, null);
    }

    private static Item pants(String id, String family, String name, String tone) {
        return new Item(id, "bottom", "바지", family, name, tone, null);
    }

    private static Item outer(String id, String family, String name, String tone) {
        return new Item(id, "outer", "자켓", family, name, tone, null);
    }

    private static Result check(String occasion, Item... items) {
        return ColorHarmony.check(List.of(items), occasion, "ok");
    }

    private static List<String> rules(Result r) {
        return r.issues().stream().map(ColorHarmony.Issue::rule).toList();
    }

    @Nested
    @DisplayName("면접·발표(formal)")
    class Formal {

        @Test
        @DisplayName("빨간 바지는 bad, 바꿀 옷은 바지 (사용자 질문: 면접에 빨간 슬랙스)")
        void redPants() {
            Result r = check("formal", shirt("g1", "white", "흰색", "light"), pants("g2", "red", "빨강", "mid"));
            assertThat(r.verdict()).isEqualTo("bad");
            assertThat(r.issues().get(0).swapGarmentId()).isEqualTo("g2");
            assertThat(r.speech()).startsWith("빨강 바지는 격식 있는 자리에 튀어요. 검정·네이비·회색 바지가 무난해요.");
        }

        @Test
        @DisplayName("빨간 상의도 bad (2026-11-11 정지원: 면접에 빨간 상의 입고 오면?)")
        void redTop() {
            Result r = check("formal", top("g1", "red", "빨강", "mid"), pants("g2", "black", "검정", "dark"));
            assertThat(r.verdict()).isEqualTo("bad");
            assertThat(r.speech()).contains("빨강 니트는 격식 있는 자리에 튀어요. 흰색·하늘색이나 네이비·회색 같은 차분한 색 니트가 무난해요.");
        }

        @Test
        @DisplayName("하늘색 셔츠 + 네이비 바지는 good, 버건디 니트는 note(ok)")
        void paleAndDark() {
            Result good = check("formal", shirt("g1", "blue", "하늘색", "light"), pants("g2", "navy", "네이비", "dark"));
            assertThat(good.verdict()).isEqualTo("good");
            assertThat(good.speech()).isEqualTo("색 조합이 잘 어울려요.");

            Result burgundy = check("formal", top("g1", "red", "어두운 빨강", "dark"), pants("g2", "gray", "회색", "mid"));
            assertThat(burgundy.verdict()).isEqualTo("ok");
            assertThat(burgundy.speech()).contains("어두운 빨강 니트도 괜찮지만");
        }

        @Test
        @DisplayName("연한 색이라도 하의는 bad (연분홍 바지)")
        void palePants() {
            assertThat(check("formal", pants("g1", "pink", "연분홍", "light")).verdict()).isEqualTo("bad");
        }
    }

    @Nested
    @DisplayName("장례식·결혼식·출근")
    class Occasions {

        @Test
        @DisplayName("장례식: 흰 셔츠 + 검정 바지 good, 베이지 바지 bad")
        void mourning() {
            assertThat(check("mourning", shirt("g1", "white", "흰색", "light"), pants("g2", "black", "검정", "dark"))
                    .verdict()).isEqualTo("good");
            Result r = check("mourning", shirt("g1", "white", "흰색", "light"), pants("g2", "beige", "베이지", "light"));
            assertThat(r.verdict()).isEqualTo("bad");
            assertThat(r.speech()).contains("베이지 바지가 눈에 띌 수 있어요");
        }

        @Test
        @DisplayName("결혼식: 흰 셔츠만이면 bad, 위에 네이비 자켓이 있으면 괜찮음")
        void wedding() {
            assertThat(check("wedding", shirt("g1", "white", "흰색", "light"), pants("g2", "gray", "회색", "mid"))
                    .verdict()).isEqualTo("bad");
            assertThat(check("wedding", shirt("g1", "white", "흰색", "light"), outer("g3", "navy", "네이비", "dark"),
                    pants("g2", "gray", "회색", "mid")).issues()).extracting(ColorHarmony.Issue::rule)
                    .doesNotContain("occasion_wedding");
        }

        @Test
        @DisplayName("출근: 노란 바지는 note(ok)")
        void business() {
            Result r = check("business", shirt("g1", "white", "흰색", "light"), pants("g2", "yellow", "노랑", "mid"));
            assertThat(r.verdict()).isEqualTo("ok");
            assertThat(rules(r)).containsExactly("occasion_business");
        }
    }

    @Nested
    @DisplayName("옷끼리 조합(일정 없는 날)")
    class Pairs {

        @Test
        @DisplayName("빨강 + 초록, 주황 + 파랑 자켓은 반대 색이라 bad, 하의를 바꾸자고 함")
        void complementary() {
            Result r = check("casual", top("g1", "red", "빨강", "mid"), pants("g2", "green", "초록", "mid"));
            assertThat(r.verdict()).isEqualTo("bad");
            assertThat(r.issues().get(0).rule()).isEqualTo("clash_complementary");
            assertThat(r.issues().get(0).swapGarmentId()).isEqualTo("g2");
            assertThat(r.speech()).startsWith("빨강 니트와 초록 바지는 서로 반대 색이라 많이 튀어요. 초록 바지를");
            assertThat(check("casual", top("g1", "orange", "주황", "mid"), outer("g3", "blue", "파랑", "mid")).verdict())
                    .isEqualTo("bad");
        }

        @Test
        @DisplayName("빨강 + 주황, 파랑 + 보라(가까운 색)는 괜찮고, 빨강 상의 + 노랑 치마는 note")
        void nearAndMid() {
            assertThat(check("casual", top("g1", "red", "빨강", "mid"), pants("g2", "orange", "주황", "mid")).verdict())
                    .isEqualTo("good");
            assertThat(check("casual", top("g1", "blue", "파랑", "mid"), pants("g2", "purple", "보라", "mid")).verdict())
                    .isEqualTo("good");
            Item skirt = new Item("g2", "bottom", "치마", "yellow", "노랑", "mid", null);
            Result r = check("casual", top("g1", "red", "빨강", "mid"), skirt);
            assertThat(rules(r)).containsExactly("clash_strong");
            assertThat(r.verdict()).isEqualTo("ok");
        }

        @Test
        @DisplayName("청바지(하의 파랑)는 기본색: 노란 티 + 청바지는 good")
        void denim() {
            assertThat(check("casual", top("g1", "yellow", "노랑", "mid"), pants("g2", "blue", "파랑", "mid")).verdict())
                    .isEqualTo("good");
        }

        @Test
        @DisplayName("검정 자켓 + 네이비 바지는 info(헷갈리는 어두운 색, 말하지 않음), 어두운 사진이면 건너뜀")
        void darkMix() {
            Result r = check("casual", outer("g1", "black", "검정", "dark"), pants("g2", "navy", "네이비", "dark"));
            // 색 인식이 검정·네이비를 자주 헷갈려서(AI Hub 검증) 지금은 말하지 않고 결과에만 남김
            assertThat(r.verdict()).isEqualTo("good");
            assertThat(r.issues().get(0).level()).isEqualTo("info");
            assertThat(r.issues().get(0).say()).startsWith("검정 자켓과 네이비 바지를 같이 입으신 것 같아요.");
            assertThat(r.speech()).isEqualTo("색 조합이 잘 어울려요.");

            Result dark = ColorHarmony.check(List.of(outer("g1", "black", "검정", "dark"), pants("g2", "navy", "네이비", "dark")),
                    "casual", "dark");
            assertThat(dark.issues()).isEmpty();
            assertThat(dark.speech()).contains("어두워서 색이 정확하지 않을 수 있어요.");
        }

        @Test
        @DisplayName("강한 색 3가지는 bad (빨강 상의 + 노랑 자켓 + 초록 바지)")
        void tooMany() {
            Result r = check("casual", top("g1", "red", "빨강", "mid"), outer("g3", "yellow", "노랑", "mid"),
                    pants("g2", "green", "초록", "mid"));
            assertThat(rules(r).get(0)).isEqualTo("too_many_colors");
            assertThat(r.speech()).startsWith("강한 색이 3가지라 복잡해 보여요. 초록 바지를").doesNotContain("반대 색");
        }

        @Test
        @DisplayName("같은 종류 두 벌(비교 중)은 서로 조합으로 보지 않음: 빨강 셔츠 vs 초록 셔츠")
        void alternatives() {
            Result r = check("casual", shirt("g1", "red", "빨강", "mid"), shirt("g2", "green", "초록", "mid"));
            assertThat(r.issues()).isEmpty();
        }

        @Test
        @DisplayName("색을 모르는 옷은 빼고 봄")
        void unknown() {
            Result r = check("casual", top("g1", "unknown", null, null), pants("g2", "black", "검정", "dark"));
            assertThat(r.verdict()).isEqualTo("good");
            assertThat(r.speech()).contains("색을 알아보지 못한 옷은 빼고 봤어요.");
        }
    }

    @Test
    @DisplayName("GarmentColor: 채도 낮고 밝은 파랑은 하늘색, 진한 파랑은 그대로 파랑")
    void paleNames() {
        assertThat(GarmentColor.name(0xB3D4F0).colorName()).isEqualTo("하늘색");
        assertThat(GarmentColor.name(0x1E88E5).colorName()).isNotEqualTo("하늘색");
        assertThat(GarmentColor.name(0xB3D4F0).tone()).isEqualTo("light");
    }
}
