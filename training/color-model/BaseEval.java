import com.fashionagent.clothesstate.color.GarmentColor;
import java.io.*;
import java.nio.file.*;
import javax.imageio.ImageIO;
/** 정답 상자로 자른 옷 조각마다 지금 서버 색 인식(GarmentColor.inBox worn)을 돌려 family를 씀. */
public class BaseEval {
    public static void main(String[] a) throws Exception {
        try (BufferedReader r = Files.newBufferedReader(Path.of(a[1])); PrintWriter w = new PrintWriter(a[2], "UTF-8")) {
            String line;
            while ((line = r.readLine()) != null) {
                String id = line.split(",", 2)[0];
                var img = ImageIO.read(new File(a[0], id + ".jpg"));
                var res = GarmentColor.inBox(img, "worn");
                w.println(id + "," + res.colorFamily() + "," + res.tone() + "," + res.colorName());
            }
        }
    }
}
