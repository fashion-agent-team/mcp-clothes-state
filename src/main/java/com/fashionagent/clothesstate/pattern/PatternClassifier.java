package com.fashionagent.clothesstate.pattern;

import java.awt.image.BufferedImage;

/** 옷 박스 하나 → 무늬. 사진은 이 서버 안에서만 쓰고 저장하지 않습니다. */
public interface PatternClassifier {

    boolean ready();

    /** 응답 modelVersion에 붙는 이름. */
    String version();

    /**
     * box는 사진 비율(0~1) x, y, w, h. pattern은 plain·stripe·check·floral·dot·graphic·animal·other 중 하나이고,
     * 무늬라고 봤지만 확신이 기준보다 낮으면 null(모름)입니다.
     */
    Result classify(BufferedImage image, double x, double y, double w, double h);

    /** top은 1등 클래스, confidence는 그 확률. */
    record Result(String pattern, String top, double confidence) {
    }
}
