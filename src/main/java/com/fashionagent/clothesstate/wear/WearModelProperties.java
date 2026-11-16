package com.fashionagent.clothesstate.wear;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.context.properties.bind.DefaultValue;

/**
 * clothes.wear-model.*: 입은 사진 옷 실수를 이 서버 안에서 보는 우리 모델 세 개(YOLO26n-cls). Fashionpedia(CC BY 4.0) 사진에
 * 실수를 합성해 학습했습니다(training/wear, 2026-11-13). 사진을 밖으로 보내지 않아 동의 없이도 돌립니다.
 * 파일이 없으면 그 모델만 건너뜁니다.
 */
@ConfigurationProperties("clothes.wear-model")
public record WearModelProperties(
        @DefaultValue("true") boolean enabled,
        /** 목 부분 택 있음/없음. 뒷모습에서 보이면 tag_out, 앞모습에서 보이면 backwards. */
        @DefaultValue("classpath:models/yolo26n_wear_tag.onnx") String tagModel,
        /** 단추 여밈 옷 한쪽이 한 칸 내려감(밑단·깃 높이 어긋남). */
        @DefaultValue("classpath:models/yolo26n_wear_button.onnx") String buttonModel,
        /** 안팎 뒤집힘(솔기 시접·비친 프린트). */
        @DefaultValue("classpath:models/yolo26n_wear_inside.onnx") String insideModel,
        @DefaultValue("yolo26n-wear3") String version,
        @DefaultValue("0.8") double tagThreshold,
        @DefaultValue("0.8") double buttonThreshold,
        /** 2차(in2) 합성 검증: 0.7에서 90.0% / 오탐 1.9%, 0.8에서 88.1% / 1.1%. */
        @DefaultValue("0.7") double insideThreshold,
        /**
         * 목 부분을 옆으로 옮기거나 작게 잘라 한 번 더 보는 창(7개)의 택 기준. 몸을 틀어 목이 옷 박스 가운데에 없는 사진용
         * (2026-11-14 실제 뒷모습 사진: 가운데 창 0.23, 옆 창 1.0). 합성 검증 400장: 가운데만 0.8 → 찾음 88.9%·오탐 0.8%,
         * 창 8개 모두 0.95 → 92.8%·1.2%. 창이 많아 오탐이 늘어서 옆 창만 기준을 높게.
         */
        @DefaultValue("0.95") double tagScanThreshold) {
}
