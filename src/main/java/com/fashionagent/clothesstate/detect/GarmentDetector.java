package com.fashionagent.clothesstate.detect;

import java.awt.image.BufferedImage;
import java.util.List;

/** 사진에서 옷 한 벌 한 벌을 찾습니다. kind는 약속 문서 category(top | bottom | outer | dress | shoes). */
public interface GarmentDetector {

    boolean ready();

    List<Detection> detect(BufferedImage image, double minConfidence);
}
