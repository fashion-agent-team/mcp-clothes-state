package com.fashionagent.clothesstate.detect;

import com.fashionagent.clothesstate.ClothesStateProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.io.ResourceLoader;

@Configuration
public class DetectorConfig {

    @Bean(destroyMethod = "close")
    OnnxDefectDetector defectDetector(ClothesStateProperties props, ResourceLoader loader) {
        return new OnnxDefectDetector(loader.getResource(props.model()), props.inputSize(), props.nmsIou());
    }
}
