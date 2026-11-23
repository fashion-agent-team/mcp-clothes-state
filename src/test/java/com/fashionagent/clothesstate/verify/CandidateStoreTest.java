package com.fashionagent.clothesstate.verify;

import static org.assertj.core.api.Assertions.assertThat;

import java.awt.image.BufferedImage;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Clock;
import java.time.Duration;
import java.time.Instant;
import java.time.ZoneId;
import java.util.stream.Stream;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import tools.jackson.databind.JsonNode;
import tools.jackson.databind.json.JsonMapper;

/** GPT가 얼룩이라고 한 조각 등 다시 학습할 후보 저장과 30일 지우기. 진짜 폴더 대신 임시 폴더를 씁니다. */
class CandidateStoreTest {

    @TempDir
    Path root;

    private final JsonMapper json = JsonMapper.builder().build();
    /** 2026-11-23 15:00 KST */
    private final Clock clock = Clock.fixed(Instant.parse("2026-11-23T06:00:00Z"), ZoneId.of("Asia/Seoul"));

    private GptCheckProperties props(String dir) {
        return new GptCheckProperties("k", "gpt-4.1-mini", "http://localhost", Duration.ofSeconds(1), 0.3, 512, 64, 0,
                0.3, 0.25, dir, 30, Duration.ofMillis(1500));
    }

    private static BufferedImage image() {
        return new BufferedImage(16, 16, BufferedImage.TYPE_INT_RGB);
    }

    @Test
    @DisplayName("save Given GPT가 얼룩이라고 함 When 저장하면 Then 오늘 날짜 폴더에 .jpg와 같은 이름의 .json(reviewed=false)")
    void savesJpgAndJson() throws Exception {
        CandidateStore store = new CandidateStore(props(root.toString()), clock);

        Path jpg = store.save(image(), "stain", 0.42, new CropVerifier.Verdict("stain", 0.91, "gpt-4.1-mini"));

        assertThat(jpg).exists().hasParent(root.resolve("2026-11-23"));
        JsonNode meta = json.readTree(Files.readString(Path.of(jpg.toString().replace(".jpg", ".json"))));
        assertThat(meta.path("label").asString()).isEqualTo("stain");
        assertThat(meta.path("modelConfidence").asDouble()).isEqualTo(0.42);
        assertThat(meta.path("gptConfidence").asDouble()).isEqualTo(0.91);
        assertThat(meta.path("reviewed").asBoolean(true)).isFalse();
    }

    @Test
    @DisplayName("savePattern Given GPT 답에 따옴표·줄바꿈이 섞임 When 저장하면 Then .json이 깨지지 않고 그대로 읽힌다")
    void jsonIsEscaped() throws Exception {
        CandidateStore store = new CandidateStore(props(root.toString()), clock);

        Path jpg = store.savePattern(image(), "stripe\",\"reviewed\":true,\"x\":\"\n", 0.8, "gpt", "solid", 0.4);

        JsonNode meta = json.readTree(Files.readString(Path.of(jpg.toString().replace(".jpg", ".json"))));
        assertThat(meta.path("task").asString()).isEqualTo("pattern");
        assertThat(meta.path("label").asString()).startsWith("stripe\"");
        assertThat(meta.path("reviewed").asBoolean(true)).isFalse();
        assertThat(jpg.getFileName().toString()).startsWith("pattern-");
    }

    @Test
    @DisplayName("saveWear Given 실수 종류 When 저장하면 Then wear-종류-아이디 이름, 종류에 경로 글자가 있어도 폴더 밖으로 나가지 않는다")
    void wearFileName() {
        CandidateStore store = new CandidateStore(props(root.toString()), clock);

        Path ok = store.saveWear(image(), "button", true, 0.7, "gpt", 0.6, "v4");
        Path odd = store.saveWear(image(), "../../etc", false, 0.2, "gpt", 0.1, "v4");

        assertThat(ok.getFileName().toString()).startsWith("wear-button-");
        assertThat(odd.normalize()).hasParent(root.resolve("2026-11-23"));
    }

    @Test
    @DisplayName("Given 폴더를 비움 When 저장하면 Then 저장하지 않고 null")
    void disabled() {
        CandidateStore store = new CandidateStore(props(""), clock);

        assertThat(store.enabled()).isFalse();
        assertThat(store.save(image(), "stain", 0.4, new CropVerifier.Verdict("stain", 0.9, "gpt"))).isNull();
    }

    @Test
    @DisplayName("pruneNow Given 31일 전·30일 전·오늘 폴더와 날짜가 아닌 폴더 When 새 저장 없이 지우기만 돌면 Then 31일 전 폴더만 지운다")
    void prunesWithoutNewSaves() throws Exception {
        // given
        Path old = Files.createDirectories(root.resolve("2026-10-23"));
        Files.writeString(old.resolve("a.json"), "{}");
        Path edge = Files.createDirectories(root.resolve("2026-10-24"));
        Path today = Files.createDirectories(root.resolve("2026-11-23"));
        Path other = Files.createDirectories(root.resolve("notes"));
        CandidateStore store = new CandidateStore(props(root.toString()), clock);

        // when
        store.pruneNow();

        // then
        assertThat(old).doesNotExist();
        assertThat(edge).exists();
        assertThat(today).exists();
        assertThat(other).exists();
        try (Stream<Path> left = Files.list(root)) {
            assertThat(left.count()).isEqualTo(3);
        }
    }
}
