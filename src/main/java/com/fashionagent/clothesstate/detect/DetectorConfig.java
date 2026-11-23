package com.fashionagent.clothesstate.detect;

import com.fashionagent.clothesstate.ClothesStateProperties;
import com.fashionagent.clothesstate.color.ColorModel;
import com.fashionagent.clothesstate.embed.EmbedProperties;
import com.fashionagent.clothesstate.embed.OnnxGarmentEmbedder;
import com.fashionagent.clothesstate.pattern.OnnxPatternClassifier;
import com.fashionagent.clothesstate.pattern.PatternProperties;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.io.ResourceLoader;

@Configuration
public class DetectorConfig {

    @Bean(destroyMethod = "close")
    OnnxDefectDetector defectDetector(ClothesStateProperties props, ResourceLoader loader) {
        return new OnnxDefectDetector(loader.getResource(props.model()), props.inputSize(), props.nmsIou());
    }

    @Bean(destroyMethod = "close")
    OnnxGarmentDetector garmentDetector(ClothesStateProperties props, ResourceLoader loader) {
        return new OnnxGarmentDetector(loader.getResource(props.garmentModel()), props.inputSize(), props.nmsIou(),
                props.sameGarmentIou(),
                props.garmentClasses().toArray(String[]::new), props.outerUnsureRatio());
    }

    /** 옷마다 이미지 벡터(이슈 #15). clothes.embed.model이 비었거나 파일이 없으면 embedding 없이 응답. */
    @Bean(destroyMethod = "close")
    OnnxGarmentEmbedder garmentEmbedder(EmbedProperties props,
            com.fashionagent.clothesstate.embed.TextCheckProperties text, ResourceLoader loader) {
        var check = text.vectors() == null || text.vectors().isBlank() ? null
                : new com.fashionagent.clothesstate.embed.GarmentTextCheck(loader.getResource(text.vectors()),
                        text.kindVectors() == null || text.kindVectors().isBlank() ? null
                                : loader.getResource(text.kindVectors()),
                        text);
        return new OnnxGarmentEmbedder(props, check);
    }

    /** 애매한 옷만 So400m 벡터 서버로 다시 고르는 보조 경로. clothes.so400m.url이 비면 꺼짐. */
    @Bean
    com.fashionagent.clothesstate.embed.So400mAssist so400mAssist(
            com.fashionagent.clothesstate.embed.So400mAssistProperties props, ResourceLoader loader) {
        return new com.fashionagent.clothesstate.embed.So400mAssist(props,
                props.probe() == null || props.probe().isBlank() ? null : loader.getResource(props.probe()),
                props.colorModel() == null || props.colorModel().isBlank() ? null : loader.getResource(props.colorModel()));
    }

    /** 옷 무늬(8종). 모델 파일이 없으면 pattern 없이 응답. clothes.pattern.lora-model 파일이 있으면 두 모델을 섞어 씀. */
    @Bean(destroyMethod = "close")
    OnnxPatternClassifier patternClassifier(PatternProperties props, ResourceLoader loader) {
        return new OnnxPatternClassifier(loader.getResource(props.model()), props.loraModel(), props);
    }

    /** 옷 색 인식 모델(K-Fashion 20색). 바꾸려면 환경 변수 CLOTHES_COLOR_MODEL, 끄려면 none. */
    @Bean(destroyMethod = "close")
    ColorModel colorModel(@Value("${clothes.color-model:classpath:models/color20_cnn.onnx}") String path,
            ResourceLoader loader) {
        return new ColorModel(loader.getResource(path));
    }
}
