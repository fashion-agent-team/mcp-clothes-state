package com.fashionagent.clothesstate.pattern;

import java.awt.image.BufferedImage;
import java.util.List;

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

    /**
     * 두 모델을 섞어 쓰는지(FashionSigLIP LoRA + YOLO). 섞어 쓰면 "모름"인 옷은 입은 사진도 GPT에게 묻고(detail high),
     * GPT 답이 모델 3등 안에 있을 때만 받습니다(docs/lab.md 7번, val로 고른 규칙).
     */
    default boolean mixed() {
        return false;
    }

    /** top은 1등 클래스, confidence는 그 확률, ranked는 확률 높은 순 1~3등. */
    record Result(String pattern, String top, double confidence, List<String> ranked) {

        public Result(String pattern, String top, double confidence) {
            this(pattern, top, confidence, List.of(top));
        }
    }
}
