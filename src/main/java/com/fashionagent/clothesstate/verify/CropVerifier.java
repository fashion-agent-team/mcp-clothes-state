package com.fashionagent.clothesstate.verify;

import java.awt.image.BufferedImage;

/** 결함 후보 조각 하나를 한 번 더 확인합니다(지금은 GPT). */
public interface CropVerifier {

    boolean ready();

    /**
     * @param crop 결함 후보 둘레만 잘라 낸 조각(옷 전체 사진이 아님)
     * @param modelKind 우리 모델이 본 종류(stain 또는 tear)
     * @return 판단. 실패하거나 시간이 지나면 null(그때는 확실하지 않은 후보로 둠)
     */
    Verdict verify(BufferedImage crop, String modelKind);

    /** label은 stain, tear, none 중 하나. */
    record Verdict(String label, double confidence, String model) {

        public boolean defect(double minConfidence) {
            return !"none".equals(label) && confidence >= minConfidence;
        }
    }
}
