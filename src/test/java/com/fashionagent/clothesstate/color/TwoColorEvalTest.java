package com.fashionagent.clothesstate.color;

import com.fashionagent.clothesstate.pattern.OnnxPatternClassifier;
import com.fashionagent.clothesstate.pattern.PatternProperties;
import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import javax.imageio.ImageIO;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;
import org.springframework.core.io.ClassPathResource;

/**
 * 무늬 시험 사진(TWO_COLOR_EVAL_DIR/&lt;정답 무늬&gt;/*.jpg, 옷 박스 조각)으로 "색이 한 가지뿐이면 무늬를 버린다" 규칙을 잽니다.
 * 무늬 모델(기준 0.7) 결과에, 두 번째 색이 없고 무늬 확률이 T보다 낮으면 버리는 규칙을 T마다 적용해
 * 정답 무늬별로 맞힘·모름·다른 무늬 비율과, 무지 옷을 무늬라고 한 비율을 출력합니다. T=0은 규칙 없음, T=1.01은 늘 버림.
 * 평소에는 건너뜀. 예: TWO_COLOR_EVAL_DIR=.../test_sh ./gradlew test --tests '*TwoColorEvalTest' -i | grep TWO_COLOR
 */
@EnabledIfEnvironmentVariable(named = "TWO_COLOR_EVAL_DIR", matches = ".+")
class TwoColorEvalTest {

    private record Row(String gt, String pattern, double conf, boolean sub) {
    }

    @Test
    void eval() throws Exception {
        File root = new File(System.getenv("TWO_COLOR_EVAL_DIR"));
        int limit = Integer.parseInt(System.getenv().getOrDefault("TWO_COLOR_EVAL_LIMIT", "300"));
        var props = new PatternProperties("", "v", List.of("animal", "check", "dot", "floral", "graphic", "other",
                "plain", "stripe"), 224, 0.08, 0.7);
        List<Row> rows = new ArrayList<>();
        try (ColorModel model = new ColorModel(new ClassPathResource("models/color20_cnn.onnx"));
                var pat = new OnnxPatternClassifier(new ClassPathResource("models/yolo26n_pattern8.onnx"), props)) {
            for (File dir : root.listFiles(File::isDirectory)) {
                File[] files = dir.listFiles((d, n) -> n.endsWith(".jpg"));
                java.util.Arrays.sort(files);
                for (int i = 0; i < Math.min(limit, files.length); i++) {
                    var img = ImageIO.read(files[i]);
                    if (img == null) {
                        continue;
                    }
                    var main = GarmentColor.fromModel(model.probabilities(img), GarmentColor.boxRgb(img, "laid"));
                    var r = pat.classify(img, 0, 0, 1, 1);
                    rows.add(new Row(dir.getName(), r.pattern(), r.confidence(),
                            GarmentColor.secondColor(img, "laid", main) != null));
                }
            }
        }
        for (double t : new double[] {0, 0.8, 0.9, 0.95, 1.01}) {
            Map<String, int[]> c = new TreeMap<>(); // n, 맞힘, 모름, 다른 무늬
            for (Row row : rows) {
                String p = row.pattern();
                if (p != null && !"plain".equals(p) && !"graphic".equals(p) && !row.sub() && row.conf() < t) {
                    p = null;
                }
                int[] k = c.computeIfAbsent(row.gt(), x -> new int[4]);
                k[0]++;
                if (row.gt().equals(p)) {
                    k[1]++;
                } else if (p == null) {
                    k[2]++;
                } else {
                    k[3]++;
                }
            }
            StringBuilder sb = new StringBuilder(String.format("TWO_COLOR T=%.2f", t));
            c.forEach((gt, k) -> sb.append(String.format(" | %s n=%d 맞힘 %.1f 모름 %.1f 틀림 %.1f", gt, k[0],
                    100.0 * k[1] / k[0], 100.0 * k[2] / k[0], 100.0 * k[3] / k[0])));
            System.out.println(sb);
        }
    }
}
