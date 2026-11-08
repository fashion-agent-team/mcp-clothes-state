package com.fashionagent.clothesstate.detect;

import java.awt.image.BufferedImage;
import java.util.List;

/** 사진 한 장 → 결함 후보(확신도 minConfidence 이상, 겹친 박스 정리 후). */
public interface DefectDetector {

    boolean ready();

    List<Detection> detect(BufferedImage image, double minConfidence);
}
