package com.fashionagent.clothesstate;

import com.fashionagent.clothesstate.detect.Detection;
import com.fashionagent.clothesstate.detect.GarmentDetector;
import java.awt.image.BufferedImage;
import java.util.List;

/** 시험용 옷 종류 모델. 기본은 준비 안 됨(사진 전체를 옷 한 벌로 보는 흐름). */
public class FakeGarmentDetector implements GarmentDetector {

    public List<Detection> next = List.of();
    public boolean ready = false;

    @Override
    public boolean ready() {
        return ready;
    }

    @Override
    public List<Detection> detect(BufferedImage image, double minConfidence) {
        return next.stream().filter(d -> d.confidence() >= minConfidence).toList();
    }
}
