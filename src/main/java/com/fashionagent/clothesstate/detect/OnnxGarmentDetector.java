package com.fashionagent.clothesstate.detect;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import org.springframework.core.io.Resource;

/**
 * 옷 종류 모델(YOLO26n, Fashionpedia CC BY 4.0로 학습). 클래스 순서는 clothes.garment-classes로 맞춥니다. 입은 사진은 잘 찾고, 내려놓은 옷은 아직 약합니다
 * (2026-11-09 평가). 모델 파일이 없으면 {@link #ready()}가 false이고, clothes_inspect는 사진 전체를 옷 한 벌로 봅니다.
 */
public class OnnxGarmentDetector implements GarmentDetector, AutoCloseable {

    /** 5종 모델의 학습 클래스 순서. */
    static final String[] KINDS5 = {"top", "bottom", "outer", "dress", "shoes"};
    /** 세부 12종 모델의 학습 클래스 순서(기본값, application.yml clothes.garment-classes). */
    public static final String[] KINDS12 = {"shirt", "tshirt", "sweater", "cardigan", "jacket", "coat", "vest", "pants",
            "shorts", "skirt", "dress", "shoes"};

    private final OnnxYolo yolo;
    private final double sameGarmentIou;

    /** 코트·재킷이 애매할 때 내보내는 종류 이름(GarmentKind에서 "겉옷"). */
    public static final String OUTER_UNSURE = "outer_unsure";

    public OnnxGarmentDetector(Resource model, int size, double nmsIou, double sameGarmentIou, String[] classes) {
        this(model, size, nmsIou, sameGarmentIou, classes, 0);
    }

    /**
     * outerUnsureRatio > 0이면 코트·재킷 중 하나를 1등으로 고른 박스에서 다른 쪽 점수가 1등의 이 배수 이상일 때
     * 종류를 "겉옷"으로 낸다. 0.1이면 내려놓은 옷·중고 의류 652장에서 코트·재킷을 잘못 부른 23장 중 9장이 "겉옷"으로
     * 바뀌고, 맞게 부른 것 중 3장만 "겉옷"이 된다(2026-11-11, fashion12e).
     */
    public OnnxGarmentDetector(Resource model, int size, double nmsIou, double sameGarmentIou, String[] classes,
            double outerUnsureRatio) {
        YoloOutput.UnsurePair unsure = outerUnsureRatio > 0
                ? new YoloOutput.UnsurePair("jacket", "coat", outerUnsureRatio, OUTER_UNSURE)
                : null;
        this.yolo = new OnnxYolo(model, size, nmsIou, classes, "옷 종류", unsure);
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
