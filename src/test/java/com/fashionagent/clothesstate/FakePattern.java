package com.fashionagent.clothesstate;

import com.fashionagent.clothesstate.pattern.PatternClassifier;
import java.awt.image.BufferedImage;
import java.util.List;

/** 시험용 무늬 모델. 기본은 준비 안 됨(pattern null). 준비되면 answer를 돌려줍니다. */
public class FakePattern implements PatternClassifier {

    public double confidence = 0.9;

    public boolean ready = false;
    public boolean fail = false;
    public String answer = "stripe";
    public int calls = 0;
    /** 두 모델을 섞어 쓰는 흉내(GPT 규칙이 바뀜). ranked는 1~3등(없으면 1등만). */
    public boolean mixed = false;
    public List<String> ranked = null;

    @Override
    public boolean ready() {
        return ready;
    }

    @Override
    public String version() {
        return "fake-pattern";
    }

    @Override
    public boolean mixed() {
        return mixed;
    }

    @Override
    public Result classify(BufferedImage image, double x, double y, double w, double h) {
        calls++;
        if (fail) {
            throw new IllegalStateException("boom");
        }
        String top = answer == null ? "dot" : answer;
        return new Result(answer, top, confidence, ranked == null ? List.of(top) : ranked);
    }
}
