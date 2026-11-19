package com.fashionagent.clothesstate;

import com.fashionagent.clothesstate.wear.WearChecker;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;

/** 시험용 옷 실수 확인. 받은 조각을 기억하고 정해 둔 답을 돌려줌. */
public class FakeWearChecker implements WearChecker {

    public boolean ready = true;
    public List<Issue> answer = List.of();
    public final List<BufferedImage> pieces = new ArrayList<>();
    public final List<String> asked = new ArrayList<>();

    @Override
    public boolean ready() {
        return ready;
    }

    @Override
    public synchronized List<Issue> check(BufferedImage garment, String garmentName, String view) {
        pieces.add(garment);
        asked.add(garmentName + "/" + view);
        return answer;
    }
}
