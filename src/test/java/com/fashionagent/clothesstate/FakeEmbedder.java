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
    /** 있으면 박스마다 이 벡터를 차례로 돌려줌(마지막 것을 반복). 글 벡터 확인 시험용. */
    public List<float[]> vectors = null;
    public com.fashionagent.clothesstate.embed.GarmentTextCheck textCheck = null;

    @Override
    public com.fashionagent.clothesstate.embed.GarmentTextCheck textCheck() {
        return textCheck;
    }

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
        if (vectors != null) {
            return vectors.get(Math.min(calls.size() - 1, vectors.size() - 1));
        }
        return new float[] {(float) x, (float) y, 1f};
    }
}
