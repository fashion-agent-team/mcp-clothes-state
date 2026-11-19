package com.fashionagent.clothesstate;

import com.fashionagent.clothesstate.detect.DefectDetector;
import com.fashionagent.clothesstate.detect.Detection;
import java.awt.image.BufferedImage;
import java.util.List;

/** 시험용: 모델 대신 정해 둔 결과를 돌려줌(실제 모델 파일 없이 흐름을 시험). */
public class FakeDetector implements DefectDetector {

    public List<Detection> next = List.of(
            new Detection("stain", 0.82, 0.70, 0.10, 0.08, 0.06),
            new Detection("stain", 0.41, 0.10, 0.80, 0.05, 0.05),
            new Detection("tear", 0.33, 0.45, 0.45, 0.10, 0.10),
            new Detection("stain", 0.20, 0.50, 0.50, 0.05, 0.05));
    public boolean ready = true;

    @Override
    public boolean ready() {
        return ready;
    }

    @Override
    public List<Detection> detect(BufferedImage image, double minConfidence) {
        return next.stream().filter(d -> d.confidence() >= minConfidence).toList();
    }
}
