package com.fashionagent.clothesstate;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.context.properties.bind.DefaultValue;
import org.springframework.util.unit.DataSize;

/**
 * application.yml의 clothes.*. 기준값은 docs/model-comparison.md "최종 결정"(정지원 2026-11-08).
 * 이 서버(ONNX, 정사각형 입력)의 확신도는 학습 때 PyTorch 값과 조금 다를 수 있어 실제 사진으로 한 번 더 확인합니다.
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
        /** 이 확신도 이상 tell 미만이면 애매함(지인 확인 후보). 미만은 버림. */
        @DefaultValue("0.3") double askThreshold,
        /** 같은 종류 박스가 이만큼(IoU) 겹치면 하나로 합침. */
        @DefaultValue("0.5") double nmsIou,
        /** 지인에게 한 번에 물을 후보 최대 개수. */
        @DefaultValue("5") int maxCandidates,
        /** 사진 평균 밝기(L*)가 이보다 낮으면 lighting=dark로 보고 판단하지 않음. */
        @DefaultValue("25") double darkLightness,
        /** 사진 base64를 풀었을 때 최대 크기. 앱은 약 6MB까지 보냄. */
        @DefaultValue("8MB") DataSize maxImageBytes) {
}
