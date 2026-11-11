package com.fashionagent.clothesstate.embed;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.context.properties.bind.DefaultValue;

/**
 * clothes.embed.*: 옷마다 이미지 벡터(옷장에서 같은 옷인지 판단용, 이슈 #15). 모델 파일(372MB)은 레포에 넣지 않고 경로로
 * 받습니다. 비어 있거나 파일이 없으면 embedding 없이 지금처럼 응답합니다.
 */
@ConfigurationProperties("clothes.embed")
public record EmbedProperties(
        /** Marqo/marqo-fashionSigLIP vision ONNX 파일 경로. 환경 변수 CLOTHES_EMBED_MODEL. */
        @DefaultValue("") String model,
        /** 응답의 embeddingModel. 옷장은 이 값이 같은 벡터끼리만 비교합니다. */
        @DefaultValue("marqo-fashionSigLIP") String version,
        /** 모델 입력 크기(정사각형 한 변). */
        @DefaultValue("224") int inputSize,
        /** 옷 박스를 사방으로 박스 크기의 이만큼 넓혀 자름. */
        @DefaultValue("0.03") double margin,
        /** 정규화 (x/255 - mean) / std. FashionSigLIP은 0.5 / 0.5. */
        @DefaultValue("0.5") float mean,
        @DefaultValue("0.5") float std,
        /** 좌우 뒤집은 사진 벡터와 평균할지. */
        @DefaultValue("true") boolean flipAverage,
        /** 응답 숫자 소수점 자리(768개라 길이를 줄임). 5자리면 코사인 차이 0.00001 수준. */
        @DefaultValue("5") int decimals) {
}
