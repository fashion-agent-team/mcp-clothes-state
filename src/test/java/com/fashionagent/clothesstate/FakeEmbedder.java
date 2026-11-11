package com.fashionagent.clothesstate;

import com.fashionagent.clothesstate.embed.GarmentEmbedder;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;

/** 시험용 옷 벡터 모델. 기본은 준비 안 됨(embedding 없이 지금처럼). 부른 박스를 기억합니다. */
public class FakeEmbedder implements GarmentEmbedder {

    public boolean ready = false;
    public boolean fail = false;
    public final List<double[]> calls = new ArrayList<>();

    @Override
    public boolean ready() {
        return ready;
    }

    @Override
    public String version() {
        return "fake-embed";
    }

    @Override
    public float[] embed(BufferedImage image, double x, double y, double w, double h) {
        calls.add(new double[] {x, y, w, h});
        if (fail) {
            throw new IllegalStateException("boom");
        }
        return new float[] {(float) x, (float) y, 1f};
    }
}
