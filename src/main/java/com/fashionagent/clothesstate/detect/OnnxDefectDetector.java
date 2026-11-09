package com.fashionagent.clothesstate.detect;

import java.awt.image.BufferedImage;
import java.util.List;
import org.springframework.core.io.Resource;

/**
 * 우리가 학습한 얼룩·찢어짐 모델(YOLO26n v6). 모델 파일이 없으면 {@link #ready()}가 false이고
 * clothes_inspect는 not_ready를 돌려줍니다.
 */
public class OnnxDefectDetector implements DefectDetector, AutoCloseable {

    private final OnnxYolo yolo;

    public OnnxDefectDetector(Resource model, int size, double nmsIou) {
        this.yolo = new OnnxYolo(model, size, nmsIou, YoloOutput.KINDS, "옷 결함");
    }

    @Override
    public boolean ready() {
        return yolo.ready();
    }

    @Override
    public List<Detection> detect(BufferedImage image, double minConfidence) {
        return yolo.detect(image, minConfidence);
    }

    @Override
    public void close() throws Exception {
        yolo.close();
    }
}
