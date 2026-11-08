package com.fashionagent.clothesstate.detect;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import org.springframework.core.io.Resource;

/**
 * 옷 종류 모델(YOLO26n, Fashionpedia CC BY 4.0로 학습, 5종). 입은 사진은 잘 찾고, 내려놓은 옷은 아직 약합니다
 * (2026-11-09 평가). 모델 파일이 없으면 {@link #ready()}가 false이고, clothes_inspect는 사진 전체를 옷 한 벌로 봅니다.
 */
public class OnnxGarmentDetector implements GarmentDetector, AutoCloseable {

    /** 학습 클래스 순서 = 약속 문서 category 이름. */
    static final String[] KINDS = {"top", "bottom", "outer", "dress", "shoes"};

    private final OnnxYolo yolo;
    private final double sameGarmentIou;

    public OnnxGarmentDetector(Resource model, int size, double nmsIou, double sameGarmentIou) {
        this.yolo = new OnnxYolo(model, size, nmsIou, KINDS, "옷 종류");
        this.sameGarmentIou = sameGarmentIou;
    }

    @Override
    public boolean ready() {
        return yolo.ready();
    }

    @Override
    public List<Detection> detect(BufferedImage image, double minConfidence) {
        return mergeSameGarment(yolo.detect(image, minConfidence), sameGarmentIou);
    }

    /**
     * 종류가 달라도 박스가 거의 같으면 한 벌이다(예: 같은 옷에 상의 0.6·원피스 0.4). 확신도 높은 쪽만 남긴다.
     * 신발은 두 짝이 따로 잡히므로 그대로 둔다.
     */
    static List<Detection> mergeSameGarment(List<Detection> in, double iou) {
        List<Detection> sorted = new ArrayList<>(in);
        sorted.sort(Comparator.comparingDouble(Detection::confidence).reversed());
        List<Detection> kept = new ArrayList<>();
        for (Detection d : sorted) {
            if (kept.stream().noneMatch(k -> k.iou(d) > iou)) {
                kept.add(d);
            }
        }
        return kept;
    }

    @Override
    public void close() throws Exception {
        yolo.close();
    }
}
