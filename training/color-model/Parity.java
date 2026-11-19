import com.fashionagent.clothesstate.color.ColorModel;
import java.io.*;
import java.nio.file.*;
import javax.imageio.ImageIO;
import org.springframework.core.io.FileSystemResource;
/** 조각 파일마다 서버 ColorModel 확률을 CSV로: name,p0..p19 */
public class Parity {
    public static void main(String[] a) throws Exception {
        var m = new ColorModel(new FileSystemResource(a[0]));
        try (PrintWriter w = new PrintWriter(a[2], "UTF-8")) {
            for (String line : Files.readAllLines(Path.of(a[1]))) {
                float[] p = m.probabilities(ImageIO.read(new File(line)));
                StringBuilder sb = new StringBuilder(line);
                for (float v : p) sb.append(',').append(v);
                w.println(sb);
            }
        }
    }
}
