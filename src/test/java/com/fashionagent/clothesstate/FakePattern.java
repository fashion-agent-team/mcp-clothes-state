package com.fashionagent.clothesstate;

import com.fashionagent.clothesstate.pattern.PatternClassifier;
import java.awt.image.BufferedImage;

/** 시험용 무늬 모델. 기본은 준비 안 됨(pattern null). 준비되면 answer를 돌려줍니다. */
public class FakePattern implements PatternClassifier {

    public double confidence = 0.9;

    public boolean ready = false;
    public boolean fail = false;
    public String answer = "stripe";
    public int calls = 0;

    @Override
    public boolean ready() {
        return ready;
    }

    @Override
    public String version() {
        return "fake-pattern";
    }

    @Override
    public Result classify(BufferedImage image, double x, double y, double w, double h) {
        calls++;
        if (fail) {
            throw new IllegalStateException("boom");
        }
        return new Result(answer, answer == null ? "dot" : answer, confidence);
    }
}
