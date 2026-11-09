package com.fashionagent.clothesstate;

import com.fashionagent.clothesstate.verify.CropVerifier;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

/** 시험용: GPT 대신 정해 둔 판단을 돌려줌. 기본은 키 없음(ready=false). */
public class FakeVerifier implements CropVerifier {

    public boolean ready = false;
    public Function<String, Verdict> answer = kind -> new Verdict("none", 0.9, "fake");
    public final List<int[]> sizes = new ArrayList<>();

    @Override
    public boolean ready() {
        return ready;
    }

    @Override
    public synchronized Verdict verify(BufferedImage crop, String modelKind) {
        sizes.add(new int[] {crop.getWidth(), crop.getHeight()});
        return answer.apply(modelKind);
    }
}
