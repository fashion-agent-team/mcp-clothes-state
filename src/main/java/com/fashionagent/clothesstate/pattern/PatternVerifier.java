package com.fashionagent.clothesstate.pattern;

import java.awt.image.BufferedImage;

/** 무늬 모델이 "모름"이라고 한 옷 한 벌을 한 번 더 확인합니다(지금은 GPT). */
public interface PatternVerifier {

    boolean ready();

    /**
     * @param garment 옷 박스만 잘라 낸 그림
     * @return 답. 실패하거나 시간이 지나면 null
     */
    Answer ask(BufferedImage garment);

    /** high면 그림을 detail high로 보냅니다(무늬 두 모델을 섞어 쓸 때, 입은 옷의 작은 조각용). */
    default Answer ask(BufferedImage garment, boolean high) {
        return ask(garment);
    }

    /** pattern은 plain·stripe·check·floral·dot·graphic·animal·other 중 하나. */
    record Answer(String pattern, double confidence, String model) {
    }
}
