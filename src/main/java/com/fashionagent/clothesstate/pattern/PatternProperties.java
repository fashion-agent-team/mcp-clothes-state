package com.fashionagent.clothesstate.pattern;

import java.util.List;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.context.properties.bind.DefaultValue;

/**
 * clothes.pattern.*: 옷 무늬 모델(YOLO26n-cls, Fashionpedia·Second-Hand Fashion CC BY 4.0로 학습, docs/training-log.md).
 * 모델 파일이 없으면 pattern은 지금처럼 null입니다.
 */
@ConfigurationProperties("clothes.pattern")
public record PatternProperties(
        /** 모델 위치(Spring Resource 경로). 환경 변수 CLOTHES_PATTERN_MODEL. */
        @DefaultValue("classpath:models/yolo26n_pattern8.onnx") String model,
        /** 응답 modelVersion 뒤에 붙는 이름. */
        @DefaultValue("yolo26n-pattern8") String version,
        /** 학습 클래스 순서. */
        @DefaultValue({"animal", "check", "dot", "floral", "graphic", "other", "plain", "stripe"}) List<String> classes,
        /** 모델 입력 크기. 짧은 변을 이만큼으로 줄이고 가운데를 정사각형으로 자름(학습 평가와 같음). */
        @DefaultValue("224") int inputSize,
        /** 옷 박스를 사방으로 박스 크기의 이만큼 넓혀 자름(학습 데이터와 같은 8%). */
        @DefaultValue("0.08") double margin,
        /** 무늬라고 말하는 최소 확신도. 1등이 무늬인데 이보다 낮으면 pattern=null(모름). 무지는 그대로 plain. */
        @DefaultValue("0.7") double sureThreshold) {
}
