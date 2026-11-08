package com.fashionagent.clothesstate.inspect;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.fashionagent.clothesstate.ClothesStateException;
import com.fashionagent.clothesstate.ClothesStateProperties;
import com.fashionagent.clothesstate.FakeDetector;
import com.fashionagent.clothesstate.FakeGarmentDetector;
import com.fashionagent.clothesstate.detect.Detection;
import java.util.List;
import com.fashionagent.clothesstate.Photos;
import java.awt.Color;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.util.unit.DataSize;

class ClothesInspectServiceTest {

    private final ClothesStateProperties props = new ClothesStateProperties("none", "yolo26n-v6", 512, 0.5, 0.3, 0.5,
            5, 25, DataSize.ofMegabytes(8), "none", "yolo26n-fashion5", 0.4, 0.7, 6);
    private final FakeDetector detector = new FakeDetector();
    private final FakeGarmentDetector garments = new FakeGarmentDetector();
    private final ClothesInspectService service = new ClothesInspectService(props, detector, garments);
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
                new Detection("shoes", 0.30, 0.40, 0.90, 0.10, 0.05));
        detector.next = List.of(
                new Detection("stain", 0.82, 0.38, 0.12, 0.04, 0.04),
                new Detection("tear", 0.42, 0.85, 0.75, 0.05, 0.05));
        String photo = Photos.halves(new Color(250, 250, 250), new Color(15, 30, 90), new Color(30, 30, 30));
        InspectResult r = service.inspect(photo, "laid", null);
        assertThat(r.modelVersion()).isEqualTo("yolo26n-v6+yolo26n-fashion5");
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
    @DisplayName("옷 종류 모델이 아무것도 못 찾으면 사진 전체를 옷 한 벌(other)로")
    void noGarmentFound() {
        garments.ready = true;
        garments.next = List.of(new Detection("top", 0.2, 0.1, 0.1, 0.5, 0.5));
        InspectResult r = service.inspect(photo, "laid", null);
        assertThat(r.garments()).singleElement().satisfies(g -> {
            assertThat(g.category()).isEqualTo("other");
            assertThat(g.box()).isEqualTo(new InspectResult.Box(0, 0, 1, 1));
        });
    }

    @Test
    @DisplayName("겹친 두 옷이면 결함은 더 작은 옷(안쪽)에, worn은 위쪽 옷부터")
    void owner() {
        var outer = new ClothesInspectService.Slot("outer", new InspectResult.Box(0.1, 0.1, 0.8, 0.6), 0.9, false);
        var top = new ClothesInspectService.Slot("top", new InspectResult.Box(0.35, 0.15, 0.3, 0.4), 0.8, false);
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
}
