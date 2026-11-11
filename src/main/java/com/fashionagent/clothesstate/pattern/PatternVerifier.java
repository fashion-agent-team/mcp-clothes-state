package com.fashionagent.clothesstate.pattern;

import java.awt.image.BufferedImage;

/** 무늬 모델이 "모름"이라고 한 옷 한 벌을 한 번 더 확인합니다(지금은 GPT). */
public interface PatternVerifier {

    boolean ready();

    /**
     * @param garment 옷 박스만 잘라 낸 그림(내려놓은 옷 사진에서만 씀)
     * @return plain·stripe·check·floral·dot·graphic·animal·other 중 하나. 실패하거나 시간이 지나면 null
     */
    String ask(BufferedImage garment);
}
