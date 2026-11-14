package com.fashionagent.clothesstate.wear;

import java.awt.image.BufferedImage;
import java.util.List;

/** 입은 옷 조각 하나에서 옷 실수를 찾습니다(지금은 GPT). */
public interface WearChecker {

    /** 실수 종류. */
    List<String> KINDS = List.of("inside_out", "backwards", "buttons_misaligned", "tag_out");

    /** 확인하지 않는 checker(시험용·키 없음). */
    WearChecker NONE = new WearChecker() {
        @Override
        public boolean ready() {
            return false;
        }

        @Override
        public List<Issue> check(BufferedImage garment, String garmentName, String view) {
            return null;
        }
    };

    boolean ready();

    /**
     * @param garment 옷 박스 조각(얼굴이 들어갈 수 있는 곳은 이미 가림)
     * @param garmentName 우리 모델이 본 옷 종류(영어, 예: shirt)
     * @param view front 또는 back
     * @return 찾은 실수(없으면 빈 목록). 실패하거나 시간이 지나면 null
     */
    List<Issue> check(BufferedImage garment, String garmentName, String view);

    record Issue(String kind, double confidence, String model) {
    }
}
