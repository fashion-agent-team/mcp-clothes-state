package com.fashionagent.clothesstate.detect;

import com.fashionagent.clothesstate.ClothesStateProperties;
import com.fashionagent.clothesstate.color.ColorModel;
import com.fashionagent.clothesstate.embed.EmbedProperties;
import com.fashionagent.clothesstate.embed.OnnxGarmentEmbedder;
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
    OnnxGarmentEmbedder garmentEmbedder(EmbedProperties props) {
        return new OnnxGarmentEmbedder(props);
    }

    /** 옷 색 인식 모델(K-Fashion 20색). 바꾸려면 환경 변수 CLOTHES_COLOR_MODEL, 끄려면 none. */
    @Bean(destroyMethod = "close")
    ColorModel colorModel(@Value("${clothes.color-model:classpath:models/color20_cnn.onnx}") String path,
            ResourceLoader loader) {
        return new ColorModel(loader.getResource(path));
    }
}
