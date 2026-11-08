package com.fashionagent.clothesstate.inspect;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.fashionagent.clothesstate.ClothesStateException;
import com.fashionagent.clothesstate.ClothesStateProperties;
import com.fashionagent.clothesstate.FakeDetector;
import com.fashionagent.clothesstate.Photos;
import java.awt.Color;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.util.unit.DataSize;

class ClothesInspectServiceTest {

    private final ClothesStateProperties props = new ClothesStateProperties("none", "yolo26n-v6", 512, 0.5, 0.3, 0.5,
            5, 25, DataSize.ofMegabytes(8));
    private final FakeDetector detector = new FakeDetector();
    private final ClothesInspectService service = new ClothesInspectService(props, detector);
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
}
