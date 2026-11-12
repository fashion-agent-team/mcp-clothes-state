package com.fashionagent.clothesstate.color;

import java.io.File;
import java.util.Map;
import java.util.TreeMap;
import javax.imageio.ImageIO;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;
import org.springframework.core.io.ClassPathResource;

/**
 * 무늬 시험 사진(TWO_COLOR_EVAL_DIR/&lt;무늬&gt;/*.jpg, 옷 박스 조각)에서 두 번째 색이 몇 %나 잡히는지 셉니다.
 * 무늬 옷인데 두 번째 색이 없으면 서버가 그 무늬를 버리므로, 무늬 클래스는 높을수록, plain은 낮을수록 좋습니다.
 * 평소에는 건너뜀. 예: TWO_COLOR_EVAL_DIR=.../test_sh ./gradlew test --tests '*TwoColorEvalTest'
 */
@EnabledIfEnvironmentVariable(named = "TWO_COLOR_EVAL_DIR", matches = ".+")
class TwoColorEvalTest {

    @Test
    void eval() throws Exception {
        File root = new File(System.getenv("TWO_COLOR_EVAL_DIR"));
        int limit = Integer.parseInt(System.getenv().getOrDefault("TWO_COLOR_EVAL_LIMIT", "300"));
        Map<String, int[]> counts = new TreeMap<>();
        try (ColorModel model = new ColorModel(new ClassPathResource("models/color20_cnn.onnx"))) {
            for (File dir : root.listFiles(File::isDirectory)) {
                File[] files = dir.listFiles((d, n) -> n.endsWith(".jpg"));
                java.util.Arrays.sort(files);
                int[] c = new int[2];
                for (int i = 0; i < Math.min(limit, files.length); i++) {
                    var img = ImageIO.read(files[i]);
                    if (img == null) {
                        continue;
                    }
                    var main = GarmentColor.fromModel(model.probabilities(img), GarmentColor.boxRgb(img, "laid"));
                    c[0]++;
                    if (GarmentColor.secondColor(img, "laid", main) != null) {
                        c[1]++;
                    }
                }
                counts.put(dir.getName(), c);
            }
        }
        counts.forEach((k, c) -> System.out.printf("TWO_COLOR %-8s n=%4d 두번째색 %5.1f%%%n", k, c[0], 100.0 * c[1] / Math.max(1, c[0])));
    }
}
