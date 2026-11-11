import com.fashionagent.clothesstate.ClothesStateProperties;
import com.fashionagent.clothesstate.color.ColorHarmony;
import com.fashionagent.clothesstate.detect.DefectDetector;
import com.fashionagent.clothesstate.detect.OnnxGarmentDetector;
import com.fashionagent.clothesstate.inspect.ClothesInspectService;
import com.fashionagent.clothesstate.inspect.InspectResult;
import com.fashionagent.clothesstate.verify.CandidateStore;
import com.fashionagent.clothesstate.verify.CropVerifier;
import com.fashionagent.clothesstate.verify.GptCheckProperties;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;
import java.util.stream.Stream;
import org.springframework.core.io.FileSystemResource;
import org.springframework.util.unit.DataSize;

/**
 * 색 조합 규칙 검증용: 사진 폴더의 입은 사진마다 실제 서버 순서(clothes_inspect worn → outfit_color_check)를 돌려
 * 옷 색과 판정을 TSV로 씁니다. 얼룩 모델·GPT는 쓰지 않습니다. 사진은 저장하지 않습니다.
 *
 * 실행: java -cp "BOOT-INF/classes:BOOT-INF/lib/*:." ColorBench <사진 폴더> <fashion12.onnx> <out.tsv> [색 모델.onnx|none] [worn|laid]
 * 색 모델을 주면 서버처럼 색 인식 모델(ColorModel)로 색을 정하고, none이거나 빼면 픽셀 규칙만 씁니다.
 * TSV에 옷 박스(box=x,y,w,h, 사진 대비 비율)도 써서 눈으로 확인할 때 씁니다.
 */
public class ColorBench {
    public static void main(String[] args) throws Exception {
        Path dir = Path.of(args[0]);
        var props = new ClothesStateProperties("none", "none", 512, 0.5, 0.2, 0.5, 5, 25, DataSize.ofMegabytes(20),
                args[1], "yolo26n-fashion12", 0.4, 0.25, 0.7, 6,
                List.of("shirt", "tshirt", "sweater", "cardigan", "jacket", "coat", "vest", "pants", "shorts", "skirt",
                        "dress", "shoes"));
        var garments = new OnnxGarmentDetector(new FileSystemResource(args[1]), 512, 0.5, 0.7,
                props.garmentClasses().toArray(String[]::new));
        DefectDetector none = new DefectDetector() {
            public boolean ready() { return true; }
            public List<com.fashionagent.clothesstate.detect.Detection> detect(java.awt.image.BufferedImage i, double c) { return List.of(); }
        };
        CropVerifier off = new CropVerifier() {
            public boolean ready() { return false; }
            public Verdict verify(java.awt.image.BufferedImage c, String k) { return null; }
        };
        var gpt = new GptCheckProperties("", "m", "http://localhost", Duration.ofSeconds(1), 0.3, 512, 64, 0, 0.3, 0.25, "", 30);
        var colors = new com.fashionagent.clothesstate.color.ColorModel(
                args.length > 3 && !"none".equals(args[3]) ? new FileSystemResource(args[3]) : null);
        String mode = args.length > 4 ? args[4] : "worn";
        var service = new ClothesInspectService(props, none, garments, gpt, off, new CandidateStore(gpt), colors);

        List<Path> photos;
        try (Stream<Path> s = Files.walk(dir)) {
            photos = s.filter(p -> p.toString().toLowerCase().endsWith(".jpg")).sorted().toList();
        }
        int done = 0;
        long t0 = System.currentTimeMillis();
        try (PrintWriter out = new PrintWriter(Files.newBufferedWriter(Path.of(args[2]), StandardCharsets.UTF_8))) {
            out.println("img\tlighting\tgarments\tcasual\tcasual_rules\tformal\tformal_rules\tbusiness\tbusiness_rules");
            for (Path p : photos) {
                try {
                    String b64 = Base64.getEncoder().encodeToString(Files.readAllBytes(p));
                    InspectResult r = service.inspect(b64, mode, "front");
                    List<ColorHarmony.Item> items = new ArrayList<>();
                    StringBuilder g = new StringBuilder();
                    for (InspectResult.Garment x : r.garments()) {
                        items.add(new ColorHarmony.Item(x.id(), x.category(), x.name(), x.colorFamily(), x.colorName(),
                                x.tone(), null));
                        if (g.length() > 0) g.append(';');
                        g.append(x.category()).append(':').append(x.name()).append(':').append(x.colorFamily())
                                .append(':').append(x.tone()).append(':').append(x.colorName()).append(':')
                                .append(String.format("%.3f,%.3f,%.3f,%.3f", x.box().x(), x.box().y(), x.box().w(),
                                        x.box().h()));
                    }
                    StringBuilder line = new StringBuilder(p.getFileName() + "\t" + r.lighting() + "\t" + g);
                    for (String occ : List.of("casual", "formal", "business")) {
                        var h = ColorHarmony.check(items, occ, r.lighting());
                        line.append('\t').append(h.verdict()).append('\t')
                                .append(String.join(",", h.issues().stream().map(ColorHarmony.Issue::rule).toList()));
                    }
                    out.println(line);
                } catch (Exception e) {
                    out.println(p.getFileName() + "\terror\t" + e.getClass().getSimpleName() + "\t\t\t\t\t\t");
                }
                if (++done % 100 == 0) {
                    out.flush();
                    System.err.printf("%d/%d %.1fs%n", done, photos.size(), (System.currentTimeMillis() - t0) / 1000.0);
                }
            }
        }
        System.err.println("끝: " + done + "장");
    }
}
