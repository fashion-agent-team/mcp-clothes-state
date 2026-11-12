package com.fashionagent.clothesstate.inspect;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.fashionagent.clothesstate.ClothesStateException;
import com.fashionagent.clothesstate.ClothesStateProperties;
import com.fashionagent.clothesstate.FakeDetector;
import com.fashionagent.clothesstate.FakeGarmentDetector;
import com.fashionagent.clothesstate.FakeVerifier;
import com.fashionagent.clothesstate.verify.CandidateStore;
import com.fashionagent.clothesstate.verify.CropVerifier;
import com.fashionagent.clothesstate.verify.GptCheckProperties;
import com.fashionagent.clothesstate.detect.Detection;
import java.util.List;
import com.fashionagent.clothesstate.Photos;
import java.awt.Color;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.util.unit.DataSize;

class ClothesInspectServiceTest {

    private final ClothesStateProperties props = new ClothesStateProperties("none", "yolo26n-v6", 512, 0.5, 0.3, 0.5,
            5, 25, DataSize.ofMegabytes(8), "none", "yolo26n-fashion12", 0.4, 0.25, 0.7, 6,
            List.of(com.fashionagent.clothesstate.detect.OnnxGarmentDetector.KINDS12), 0.1);
    private final FakeDetector detector = new FakeDetector();
    private final FakeGarmentDetector garments = new FakeGarmentDetector();
    @org.junit.jupiter.api.io.TempDir
    java.nio.file.Path tmp;
    private final FakeVerifier verifier = new FakeVerifier();
    private final com.fashionagent.clothesstate.FakeEmbedder embedder = new com.fashionagent.clothesstate.FakeEmbedder();
    private final com.fashionagent.clothesstate.FakePattern pattern = new com.fashionagent.clothesstate.FakePattern();
    private final com.fashionagent.clothesstate.FakePatternVerifier patternGpt = new com.fashionagent.clothesstate.FakePatternVerifier();
    private ClothesInspectService service;

    @org.junit.jupiter.api.BeforeEach
    void setUp() {
        var gpt = new GptCheckProperties("test-key", "gpt-4.1-mini", "http://localhost", java.time.Duration.ofSeconds(1),
                0.3, 512, 64, 0, 0.3, 0.25, tmp.toString(), 30);
        service = new ClothesInspectService(props, detector, garments, gpt, verifier, new CandidateStore(gpt), embedder, pattern, patternGpt);
    }
    private final String photo = Photos.jpeg(new Color(200, 200, 200));

    @Test
    @DisplayName("0.5 이상은 defects(바로 알림), 0.3~0.5는 pendingHelper, 0.3 미만은 버림")
    void thresholds() {
        InspectResult r = service.inspect(photo, null, null);
        assertThat(r.lighting()).isEqualTo("ok");
        assertThat(r.source()).isEqualTo("model");
        assertThat(r.modelVersion()).isEqualTo("yolo26n-v6");
        var g = r.garments().getFirst();
        assertThat(g.id()).isEqualTo("g1");
        assertThat(g.category()).isEqualTo("other");
        assertThat(g.name()).isEqualTo("옷");
        assertThat(g.position()).isEqualTo("center");
        assertThat(g.defects()).singleElement().satisfies(d -> {
            assertThat(d.kind()).isEqualTo("stain");
            assertThat(d.where()).isEqualTo("옷 오른쪽 위쪽");
            assertThat(d.confidence()).isEqualTo(0.82);
        });
        assertThat(r.pendingHelper().garmentId()).isEqualTo("g1");
        assertThat(r.pendingHelper().note()).isEqualTo("확실하지 않은 부분이 있어요.");
        assertThat(r.pendingHelper().candidates()).extracting(InspectResult.Candidate::kind,
                InspectResult.Candidate::where).containsExactly(
                        org.assertj.core.groups.Tuple.tuple("stain", "옷 왼쪽 아래쪽"),
                        org.assertj.core.groups.Tuple.tuple("tear", "옷 가운데"));
        assertThat(r.pendingHelper().candidates().getFirst().box())
                .isEqualTo(new InspectResult.Box(0.1, 0.8, 0.05, 0.05));
    }

    @Test
    @DisplayName("키가 있으면 애매한 것은 GPT에게 묻고, 결함이라고 한 것만 defects(checkedBy=gpt)와 학습 후보로")
    void gptCheck() throws Exception {
        verifier.ready = true;
        verifier.answer = kind -> "tear".equals(kind) ? new CropVerifier.Verdict("tear", 0.8, "fake")
                : new CropVerifier.Verdict("none", 0.9, "fake");
        InspectResult r = service.inspect(photo, null, null, true);
        var g = r.garments().getFirst();
        assertThat(g.defects()).extracting(InspectResult.Defect::kind, InspectResult.Defect::checkedBy).containsExactly(
                org.assertj.core.groups.Tuple.tuple("stain", "model"),
                org.assertj.core.groups.Tuple.tuple("tear", "gpt"));
        assertThat(r.pendingHelper()).isNull();
        assertThat(verifier.sizes).hasSize(2);
        try (var files = java.nio.file.Files.walk(tmp)) {
            var saved = files.filter(java.nio.file.Files::isRegularFile).map(p -> p.getFileName().toString()).toList();
            assertThat(saved).hasSize(2).anyMatch(n -> n.endsWith(".jpg")).anyMatch(n -> n.endsWith(".json"));
        }
    }

    @Test
    @DisplayName("동의 없음(allowGpt null·false)이나 GPT가 답하지 못하면 예전처럼 pendingHelper, 아무것도 저장하지 않음")
    void gptOff() throws Exception {
        verifier.ready = true;
        assertThat(service.inspect(photo, null, null, false).pendingHelper().candidates()).hasSize(2);
        assertThat(service.inspect(photo, null, null).pendingHelper().candidates()).hasSize(2);
        assertThat(verifier.sizes).isEmpty();
        verifier.answer = kind -> null;
        assertThat(service.inspect(photo, null, null, true).pendingHelper().candidates()).hasSize(2);
        try (var files = java.nio.file.Files.walk(tmp)) {
            assertThat(files.filter(java.nio.file.Files::isRegularFile)).isEmpty();
        }
    }

    @Test
    @DisplayName("입은 사진 위쪽(얼굴이 들어갈 수 있음) 후보는 GPT에게 보내지 않음")
    void wornTopNotSent() {
        verifier.ready = true;
        detector.next = List.of(new Detection("stain", 0.41, 0.4, 0.05, 0.05, 0.05),
                new Detection("stain", 0.41, 0.4, 0.6, 0.05, 0.05));
        InspectResult r = service.inspect(photo, "worn", "front", true);
        assertThat(verifier.sizes).hasSize(1);
        assertThat(r.pendingHelper().candidates()).hasSize(1);
    }

    @Test
    @DisplayName("결함 조각: 박스 30% 넓힘, 64px보다 작으면 늘림, 긴 변 512 이하")
    void defectCrop() {
        var img = new java.awt.image.BufferedImage(2000, 1000, java.awt.image.BufferedImage.TYPE_INT_RGB);
        var small = ClothesInspectService.defectCrop(img, new Detection("stain", 0.4, 0.5, 0.5, 0.01, 0.02), 0.3, 64,
                512);
        assertThat(small.getWidth()).isEqualTo(64);
        assertThat(small.getHeight()).isEqualTo(64);
        var mid = ClothesInspectService.defectCrop(img, new Detection("stain", 0.4, 0.1, 0.1, 0.1, 0.1), 0.3, 64, 512);
        assertThat(mid.getWidth()).isEqualTo(320);
        assertThat(mid.getHeight()).isEqualTo(160);
        var big = ClothesInspectService.defectCrop(img, new Detection("stain", 0.4, 0, 0, 1, 1), 0.3, 64, 512);
        assertThat(big.getWidth()).isEqualTo(512);
        assertThat(big.getHeight()).isEqualTo(256);
    }

    @Test
    @DisplayName("기준 0.2: GPT를 쓰면 0.2~0.3도 묻고, 못 쓰면 0.3 미만은 예전처럼 버림. 3초(여기선 0.2초)를 넘기면 확인 못 한 것으로")
    void fallbackAndTimeout() {
        var p02 = new ClothesStateProperties("none", "yolo26n-v6", 512, 0.5, 0.2, 0.5, 5, 25, DataSize.ofMegabytes(8),
                "none", "yolo26n-fashion12", 0.4, 0.25, 0.7, 6,
                List.of(com.fashionagent.clothesstate.detect.OnnxGarmentDetector.KINDS12), 0.1);
        var gpt = new GptCheckProperties("k", "m", "http://localhost", java.time.Duration.ofMillis(200), 0.3, 512, 64, 0,
                0.3, 0.25, "", 30);
        var svc = new ClothesInspectService(p02, detector, garments, gpt, verifier, new CandidateStore(gpt), embedder, pattern, patternGpt);
        verifier.ready = true;
        // GPT 안 씀: 0.41·0.33만 후보, 0.20은 버림
        assertThat(svc.inspect(photo, null, null).pendingHelper().candidates()).hasSize(2);
        // GPT 씀: 0.41·0.33·0.20 세 개를 물음
        svc.inspect(photo, null, null, true);
        assertThat(verifier.sizes).hasSize(3);
        // GPT가 늦으면 확인 못 한 것으로(0.3 이상만 후보)
        verifier.answer = kind -> {
            try {
                Thread.sleep(2000);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            return new CropVerifier.Verdict("stain", 0.9, "fake");
        };
        long t = System.nanoTime();
        InspectResult r = svc.inspect(photo, null, null, true);
        assertThat((System.nanoTime() - t) / 1_000_000).isLessThan(1500);
        assertThat(r.pendingHelper().candidates()).hasSize(2);
        assertThat(r.garments().getFirst().defects()).hasSize(1);
    }

    @Test
    @DisplayName("학습 후보는 보관 기간(30일)이 지난 날짜 폴더를 지움")
    void prune() throws Exception {
        var gpt = new GptCheckProperties("k", "m", "http://localhost", java.time.Duration.ofSeconds(1), 0.3, 512, 64, 0,
                0.3, 0.25, tmp.toString(), 30);
        java.nio.file.Files.createDirectories(tmp.resolve("2026-10-01"));
        java.nio.file.Files.writeString(tmp.resolve("2026-10-01/a.json"), "{}");
        java.nio.file.Files.createDirectories(tmp.resolve("2026-11-01"));
        new CandidateStore(gpt).prune(java.time.LocalDate.of(2026, 11, 9));
        assertThat(tmp.resolve("2026-10-01")).doesNotExist();
        assertThat(tmp.resolve("2026-11-01")).exists();
    }

    @Test
    @DisplayName("애매한 것이 없으면 pendingHelper는 null")
    void noPending() {
        detector.next = detector.next.subList(0, 1);
        assertThat(service.inspect(photo, "laid", null).pendingHelper()).isNull();
    }

    @Test
    @DisplayName("worn이면 입은 사람 몸 기준 where, position 없음")
    void worn() {
        InspectResult r = service.inspect(photo, "worn", "front");
        assertThat(r.garments().getFirst().position()).isNull();
        assertThat(r.garments().getFirst().defects().getFirst().where()).isEqualTo("몸 왼쪽 위쪽");
    }

    @Test
    @DisplayName("어두운 사진은 판단하지 않고 lighting=dark")
    void dark() {
        InspectResult r = service.inspect(Photos.jpeg(new Color(10, 10, 12)), null, null);
        assertThat(r.lighting()).isEqualTo("dark");
        assertThat(r.garments()).isEmpty();
    }

    @Test
    @DisplayName("잘못된 mode·view, 모델 없음은 오류 코드")
    void errors() {
        assertThatThrownBy(() -> service.inspect(photo, "hanging", null))
                .isInstanceOfSatisfying(ClothesStateException.class, e -> assertThat(e.code()).isEqualTo("bad_request"));
        assertThatThrownBy(() -> service.inspect(photo, "worn", "top"))
                .isInstanceOfSatisfying(ClothesStateException.class, e -> assertThat(e.code()).isEqualTo("bad_request"));
        detector.ready = false;
        assertThatThrownBy(() -> service.inspect(photo, null, null))
                .isInstanceOfSatisfying(ClothesStateException.class, e -> assertThat(e.code()).isEqualTo("not_ready"));
    }

    @Test
    @DisplayName("옷 종류 모델이 찾은 옷마다 종류·박스 안 색·위치, 결함은 그 옷에 붙고 where는 옷 기준")
    void garments() {
        garments.ready = true;
        garments.next = List.of(
                new Detection("bottom", 0.88, 0.55, 0.10, 0.40, 0.80),
                new Detection("top", 0.91, 0.05, 0.10, 0.40, 0.60),
                new Detection("shoes", 0.20, 0.40, 0.90, 0.10, 0.05));
        detector.next = List.of(
                new Detection("stain", 0.82, 0.38, 0.12, 0.04, 0.04),
                new Detection("tear", 0.42, 0.85, 0.75, 0.05, 0.05));
        String photo = Photos.halves(new Color(250, 250, 250), new Color(15, 30, 90), new Color(30, 30, 30));
        InspectResult r = service.inspect(photo, "laid", null);
        assertThat(r.modelVersion()).isEqualTo("yolo26n-v6+yolo26n-fashion12");
        assertThat(r.garments()).extracting(InspectResult.Garment::id, InspectResult.Garment::category,
                InspectResult.Garment::name, InspectResult.Garment::position).containsExactly(
                        org.assertj.core.groups.Tuple.tuple("g1", "top", "상의", "left"),
                        org.assertj.core.groups.Tuple.tuple("g2", "bottom", "하의", "right"));
        var top = r.garments().get(0);
        var bottom = r.garments().get(1);
        assertThat(top.colorFamily()).isEqualTo("navy");
        assertThat(bottom.colorFamily()).isEqualTo("black");
        assertThat(top.confidence()).isEqualTo(0.91);
        assertThat(top.box()).isEqualTo(new InspectResult.Box(0.05, 0.1, 0.4, 0.6));
        assertThat(top.defects()).singleElement().satisfies(d -> assertThat(d.where()).isEqualTo("옷 오른쪽 위쪽"));
        assertThat(bottom.defects()).isEmpty();
        assertThat(r.pendingHelper().garmentId()).isEqualTo("g2");
        assertThat(r.pendingHelper().candidates()).singleElement()
                .satisfies(c -> assertThat(c.where()).isEqualTo("옷 오른쪽 아래쪽"));
    }

    @Test
    @DisplayName("벡터 모델이 있으면 옷마다 그 옷 박스로 embedding, embeddingModel도 같이")
    void embeddings() {
        embedder.ready = true;
        garments.ready = true;
        garments.next = List.of(
                new Detection("bottom", 0.88, 0.55, 0.10, 0.40, 0.80),
                new Detection("top", 0.91, 0.05, 0.10, 0.40, 0.60));
        InspectResult r = service.inspect(photo, "laid", null);
        assertThat(r.embeddingModel()).isEqualTo("fake-embed");
        assertThat(r.garments()).extracting(InspectResult.Garment::embedding)
                .containsExactly(new float[] {0.05f, 0.1f, 1f}, new float[] {0.55f, 0.1f, 1f});
        assertThat(embedder.calls).hasSize(2);
        assertThat(embedder.calls.getFirst()).containsExactly(0.05, 0.1, 0.4, 0.6);
    }

    @Test
    @DisplayName("벡터 모델이 없으면 embedding·embeddingModel은 null, 실패하면 그 옷만 null이고 검사 결과는 그대로")
    void embeddingsOffOrFailing() {
        InspectResult off = service.inspect(photo, null, null);
        assertThat(off.embeddingModel()).isNull();
        assertThat(off.garments().getFirst().embedding()).isNull();
        assertThat(embedder.calls).isEmpty();
        embedder.ready = true;
        embedder.fail = true;
        InspectResult failed = service.inspect(photo, null, null);
        assertThat(failed.garments().getFirst().embedding()).isNull();
        assertThat(failed.garments().getFirst().defects()).hasSize(1);
        // 사진 전체를 옷 한 벌로 볼 때는 사진 전체 박스
        assertThat(embedder.calls.getFirst()).containsExactly(0.0, 0.0, 1.0, 1.0);
    }

    @Test
    @DisplayName("무늬 모델이 있으면 옷마다 pattern, 신발은 null, modelVersion 뒤에 무늬 모델 이름")
    void patterns() {
        String photo = Photos.stripes(new Color(20, 30, 80), new Color(240, 240, 240));
        assertThat(service.inspect(photo, null, null).garments().getFirst().pattern()).isNull();
        pattern.ready = true;
        garments.ready = true;
        garments.next = List.of(
                new Detection("shirt", 0.91, 0.05, 0.10, 0.40, 0.60),
                new Detection("shoes", 0.88, 0.55, 0.70, 0.20, 0.20));
        InspectResult r = service.inspect(photo, "laid", null);
        assertThat(r.modelVersion()).endsWith("+fake-pattern");
        assertThat(r.garments()).extracting(InspectResult.Garment::pattern).containsExactly("stripe", null);
        assertThat(pattern.calls).isEqualTo(1);
        // 모르면 null, 실패해도 검사 결과는 그대로
        pattern.answer = null;
        assertThat(service.inspect(photo, "laid", null).garments().getFirst().pattern()).isNull();
        pattern.fail = true;
        InspectResult failed = service.inspect(photo, "laid", null);
        assertThat(failed.garments().getFirst().pattern()).isNull();
        assertThat(failed.garments()).hasSize(2);
    }

    @Test
    @DisplayName("옷 색이 한 가지뿐인데 모델이 줄무늬라고 하면 버림(모니터 화면 줄·청바지 결), 두 색이면 두 번째 색과 함께")
    void patternNeedsTwoColors() {
        pattern.ready = true;
        garments.ready = true;
        garments.next = List.of(new Detection("shirt", 0.91, 0.05, 0.10, 0.60, 0.80));
        var plain = service.inspect(photo, "laid", null).garments().getFirst();
        assertThat(plain.pattern()).isNull();
        assertThat(plain.patternCheckedBy()).isNull();
        assertThat(plain.subColorName()).isNull();
        var striped = service.inspect(Photos.stripes(new Color(20, 30, 80), new Color(240, 240, 240)), "laid", null)
                .garments().getFirst();
        assertThat(striped.pattern()).isEqualTo("stripe");
        assertThat(striped.subColorName()).isNotNull();
    }

    @Test
    @DisplayName("무늬 모름: 내려놓은 사진·동의했을 때만 옷 박스를 GPT에게 물어 patternCheckedBy=gpt")
    void patternGpt() {
        String photo = Photos.stripes(new Color(20, 30, 80), new Color(240, 240, 240));
        pattern.ready = true;
        pattern.answer = null; // 모델은 모름
        patternGpt.ready = true;
        garments.ready = true;
        garments.next = List.of(new Detection("shirt", 0.91, 0.05, 0.10, 0.40, 0.60));
        // 동의 안 함 → 묻지 않음
        var g0 = service.inspect(photo, "laid", null, false).garments().getFirst();
        assertThat(g0.pattern()).isNull();
        assertThat(g0.patternCheckedBy()).isNull();
        assertThat(patternGpt.calls).isZero();
        // 입은 사진 → 묻지 않음
        service.inspect(photo, "worn", "front", true);
        assertThat(patternGpt.calls).isZero();
        // 내려놓은 사진 + 동의 → 옷 박스만 보냄
        var g = service.inspect(photo, "laid", null, true).garments().getFirst();
        assertThat(g.pattern()).isEqualTo("check");
        assertThat(g.patternCheckedBy()).isEqualTo("gpt");
        assertThat(patternGpt.calls).isEqualTo(1);
        assertThat(patternGpt.lastWidth).isEqualTo(128); // 320 × 0.40
        // 옷 박스 그림과 GPT 답을 무늬 학습 후보로 저장(그림 + json)
        try (var files = java.nio.file.Files.walk(tmp)) {
            var saved = files.filter(f -> f.getFileName().toString().startsWith("pattern-")).toList();
            assertThat(saved).hasSize(2);
            var json = saved.stream().filter(f -> f.toString().endsWith(".json")).findFirst().orElseThrow();
            assertThat(java.nio.file.Files.readString(json)).contains("\"task\":\"pattern\"", "\"label\":\"check\"",
                    "\"modelTop\":\"dot\"", "\"reviewed\":false");
        } catch (java.io.IOException e) {
            throw new java.io.UncheckedIOException(e);
        }
        // 모델이 확실하면 묻지 않고 model
        pattern.answer = "stripe";
        var g2 = service.inspect(photo, "laid", null, true).garments().getFirst();
        assertThat(g2.patternCheckedBy()).isEqualTo("model");
        assertThat(patternGpt.calls).isEqualTo(1);
        // GPT가 답을 못 하면 그대로 모름
        pattern.answer = null;
        patternGpt.answer = null;
        assertThat(service.inspect(photo, "laid", null, true).garments().getFirst().pattern()).isNull();
    }

    @Test
    @DisplayName("옷 종류 모델이 아무것도 못 찾으면 사진 전체를 옷 한 벌(other)로")
    void noGarmentFound() {
        garments.ready = true;
        garments.next = List.of(new Detection("top", 0.2, 0.1, 0.1, 0.5, 0.5));
        InspectResult r = service.inspect(photo, "laid", null);
        assertThat(service.inspect(photo, "worn", "front").garments().getFirst().category()).isEqualTo("other");
        assertThat(r.garments()).singleElement().satisfies(g -> {
            assertThat(g.category()).isEqualTo("other");
            assertThat(g.box()).isEqualTo(new InspectResult.Box(0, 0, 1, 1));
        });
    }

    @Test
    @DisplayName("겹친 두 옷이면 결함은 더 작은 옷(안쪽)에, worn은 위쪽 옷부터")
    void owner() {
        var outer = new ClothesInspectService.Slot(GarmentKind.of("outer"), new InspectResult.Box(0.1, 0.1, 0.8, 0.6), 0.9, false);
        var top = new ClothesInspectService.Slot(GarmentKind.of("top"), new InspectResult.Box(0.35, 0.15, 0.3, 0.4), 0.8, false);
        assertThat(ClothesInspectService.owner(List.of(outer, top), new Detection("stain", 0.7, 0.45, 0.3, 0.05, 0.05)))
                .isSameAs(top);
        assertThat(ClothesInspectService.owner(List.of(outer, top), new Detection("stain", 0.7, 0.15, 0.3, 0.05, 0.05)))
                .isSameAs(outer);
        assertThat(ClothesInspectService.owner(List.of(outer, top), new Detection("stain", 0.7, 0.5, 0.95, 0.02, 0.02)))
                .isNotNull();

        garments.ready = true;
        garments.next = List.of(
                new Detection("shoes", 0.8, 0.4, 0.9, 0.2, 0.08),
                new Detection("top", 0.9, 0.3, 0.2, 0.4, 0.35));
        InspectResult r = service.inspect(this.photo, "worn", "front");
        assertThat(r.garments()).extracting(InspectResult.Garment::category).containsExactly("top", "shoes");
        assertThat(r.garments()).allSatisfy(g -> assertThat(g.position()).isNull());
    }

    @Test
    @DisplayName("내려놓은 옷은 0.25부터 옷으로 보고, 입은 사진은 0.4부터")
    void laidThreshold() {
        garments.ready = true;
        garments.next = List.of(new Detection("top", 0.33, 0.1, 0.1, 0.8, 0.8));
        assertThat(service.inspect(photo, "laid", null).garments().getFirst().category()).isEqualTo("top");
        assertThat(service.inspect(photo, "worn", "front").garments().getFirst().category()).isEqualTo("other");
    }

    @Test
    @DisplayName("세부 종류는 category·이름·격식·소재로 바뀜")
    void fineKinds() {
        garments.ready = true;
        garments.next = List.of(
                new Detection("sweater", 0.8, 0.05, 0.1, 0.4, 0.6),
                new Detection("shorts", 0.7, 0.55, 0.1, 0.4, 0.8));
        var g = service.inspect(photo, "laid", null).garments();
        assertThat(g).extracting(InspectResult.Garment::category, InspectResult.Garment::name,
                InspectResult.Garment::formality, InspectResult.Garment::material).containsExactly(
                        org.assertj.core.groups.Tuple.tuple("top", "니트", "unknown", "니트"),
                        org.assertj.core.groups.Tuple.tuple("bottom", "반바지", "casual", null));
        assertThat(GarmentKind.of("coat")).isEqualTo(new GarmentKind("outer", "코트", "smart", null));
        assertThat(GarmentKind.of("outer_unsure")).isEqualTo(new GarmentKind("outer", "겉옷", "unknown", null));
        assertThat(GarmentKind.of("모름")).isEqualTo(GarmentKind.OTHER);
    }
}
