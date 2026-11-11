package com.fashionagent.clothesstate;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.context.properties.bind.DefaultValue;
import org.springframework.util.unit.DataSize;

/**
 * application.yml의 clothes.*. 기준값 0.5 / 0.2는 정지원 2026-11-09(docs/training-log.md "사진 단위 정답률").
 * 옷 종류 기준(garment-threshold)은 아직 첫 값이라 실제 사진으로 조정합니다.
 */
@ConfigurationProperties("clothes")
public record ClothesStateProperties(
        /** ONNX 모델 위치 (Spring Resource 경로). 바꾸려면 환경 변수 CLOTHES_MODEL. */
        @DefaultValue("classpath:models/yolo26n_v6.onnx") String model,
        /** 응답의 modelVersion. */
        @DefaultValue("yolo26n-v6") String modelVersion,
        /** 모델 입력 크기(정사각형 한 변). */
        @DefaultValue("512") int inputSize,
        /** 이 확신도 이상이면 결함으로 보고 바로 알림. */
        @DefaultValue("0.5") double tellThreshold,
        /** 이 확신도 이상 tell 미만이면 애매함(GPT에게 물음, 키가 없으면 지인 확인 후보). 미만은 버림. */
        @DefaultValue("0.2") double askThreshold,
        /** 같은 종류 박스가 이만큼(IoU) 겹치면 하나로 합침. */
        @DefaultValue("0.5") double nmsIou,
        /** 지인에게 한 번에 물을 후보 최대 개수. */
        @DefaultValue("5") int maxCandidates,
        /** 사진 평균 밝기(L*)가 이보다 낮으면 lighting=dark로 보고 판단하지 않음. */
        @DefaultValue("25") double darkLightness,
        /**
         * 사진의 노란 기운(Images.warmCast, b* 아래쪽 10%)이 이보다 높으면 lighting=warm. 검사는 그대로 하고, 색이
         * 틀릴 수 있다고 알리며 헷갈리는 색은 colorAlternative를 채웁니다. 실제 전구색·주광색 사진으로 다시 맞출 값.
         */
        @DefaultValue("12") double warmCast,
        /** 사진 base64를 풀었을 때 최대 크기. 앱은 약 6MB까지 보냄. */
        @DefaultValue("8MB") DataSize maxImageBytes,
        /** 옷 종류 모델 위치. 바꾸려면 환경 변수 CLOTHES_GARMENT_MODEL. 없으면 사진 전체를 옷 한 벌(other)로 봄. */
        @DefaultValue("classpath:models/yolo26n_fashion12.onnx") String garmentModel,
        /** 옷 종류 모델 이름(modelVersion 뒤에 붙음). */
        @DefaultValue("yolo26n-fashion12") String garmentModelVersion,
        /** 입은 사진에서 옷으로 볼 최소 확신도. */
        @DefaultValue("0.4") double garmentThreshold,
        /**
         * 내려놓은 옷 사진의 최소 확신도. 모델이 내려놓은 옷에는 확신도를 낮게 주고(0.33~0.93), 종류는 틀려도 얼룩처럼
         * 위험하지 않아 낮게 둡니다.
         */
        @DefaultValue("0.25") double laidGarmentThreshold,
        /** 종류가 달라도 박스가 이만큼(IoU) 겹치면 같은 옷으로 봄. */
        @DefaultValue("0.7") double sameGarmentIou,
        /** 한 사진에서 돌려줄 옷 최대 개수. */
        @DefaultValue("6") int maxGarments,
        /** 옷 종류 모델의 클래스 순서(학습 순서). 5종 모델을 쓰면 top,bottom,outer,dress,shoes. */
        @DefaultValue({"shirt", "tshirt", "sweater", "cardigan", "jacket", "coat", "vest", "pants", "shorts", "skirt",
                "dress", "shoes"}) java.util.List<String> garmentClasses) {
}
