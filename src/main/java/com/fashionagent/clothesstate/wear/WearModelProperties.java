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
        @DefaultValue("yolo26n-wear1") String version,
        @DefaultValue("0.8") double tagThreshold,
        @DefaultValue("0.8") double buttonThreshold,
        @DefaultValue("0.8") double insideThreshold) {
}
