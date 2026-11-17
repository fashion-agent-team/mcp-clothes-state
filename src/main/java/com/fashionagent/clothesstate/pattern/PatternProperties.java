package com.fashionagent.clothesstate.pattern;

import java.util.List;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.context.properties.bind.ConstructorBinding;
import org.springframework.boot.context.properties.bind.DefaultValue;

/**
 * clothes.pattern.*: 옷 무늬 모델(YOLO26n-cls, Fashionpedia·Second-Hand Fashion CC BY 4.0로 학습, docs/training-log.md).
 * 모델 파일이 없으면 pattern은 지금처럼 null입니다.
 *
 * <p>loraModel이 있으면 FashionSigLIP + LoRA 무늬 모델(Fashionpedia·Second-Hand Fashion·AI Hub K-Fashion로 학습,
 * docs/lab.md 7번)과 위 모델의 확률을 섞어 씁니다(loraWeight : 1 - loraWeight). 파일은 레포에 넣지 않고 젯슨에 둡니다.
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
        @DefaultValue("0.7") double sureThreshold,
        /** FashionSigLIP LoRA 무늬 ONNX 경로(입력 RGB 0~1 224, 출력 8종 확률). 비우면 위 모델만. 환경 변수 CLOTHES_PATTERN_LORA_MODEL. */
        @DefaultValue("") String loraModel,
        /** 섞을 때 LoRA 모델 몫. val(Fashionpedia val 2,376장)로 고른 0.7. */
        @DefaultValue("0.7") double loraWeight,
        /** 섞은 확률로 무늬라고 말하는 최소 확신도. val로 고른 0.6. */
        @DefaultValue("0.6") double loraSureThreshold,
        /** 섞어 쓸 때 modelVersion 뒤에 붙는 이름. */
        @DefaultValue("fsiglip-lora-pattern8+yolo26n-pattern8") String loraVersion) {

    @ConstructorBinding
    public PatternProperties {
    }

    /** LoRA 모델 없이(예전 설정) 만들 때. */
    public PatternProperties(String model, String version, List<String> classes, int inputSize, double margin,
            double sureThreshold) {
        this(model, version, classes, inputSize, margin, sureThreshold, "", 0.7, 0.6,
                "fsiglip-lora-pattern8+yolo26n-pattern8");
    }
}
