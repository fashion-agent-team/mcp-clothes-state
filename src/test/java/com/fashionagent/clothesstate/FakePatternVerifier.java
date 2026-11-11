package com.fashionagent.clothesstate;

import com.fashionagent.clothesstate.pattern.PatternVerifier;
import java.awt.image.BufferedImage;

/** 시험용 GPT 무늬 확인. 기본은 준비 안 됨. 부른 횟수와 받은 그림 크기를 기억합니다. */
public class FakePatternVerifier implements PatternVerifier {

    public boolean ready = false;
    public String answer = "check";
    public int calls = 0;
    public int lastWidth = -1;

    @Override
    public boolean ready() {
        return ready;
    }

    @Override
    public String ask(BufferedImage garment) {
        calls++;
        lastWidth = garment.getWidth();
        return answer;
    }
}
