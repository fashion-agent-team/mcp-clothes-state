package com.fashionagent.clothesstate.embed;

import java.awt.image.BufferedImage;

/** 옷 박스 하나 → 길이 1로 정규화한 이미지 벡터. 사진은 저장하지 않고 숫자만 돌려줍니다. */
public interface GarmentEmbedder {

    boolean ready();

    /** 응답의 embeddingModel(옷장이 같은 모델 벡터끼리만 비교하도록). */
    String version();

    /** box는 사진 비율(0~1) x, y, w, h. */
    float[] embed(BufferedImage image, double x, double y, double w, double h);

    /** 옷 이름 글 벡터 확인(종류 다시 보기·옷 아닌 것 빼기). 없으면 null. */
    default GarmentTextCheck textCheck() {
        return null;
    }
}
