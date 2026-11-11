import com.fashionagent.clothesstate.color.ColorFeatures;
import java.io.*;
import java.nio.*;
import java.nio.channels.FileChannel;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;
import javax.imageio.ImageIO;

/** 옷 조각마다 ColorFeatures를 뽑아 float32 행렬(little endian)로 씀. 순서는 meta 줄 순서. */
public class FeatDump {
    public static void main(String[] a) throws Exception {
        List<String> ids = new ArrayList<>();
        for (String line : Files.readAllLines(Path.of(a[1]))) ids.add(line.split(",", 2)[0]);
        int L = ColorFeatures.LENGTH;
        float[][] rows = new float[ids.size()][];
        ExecutorService ex = Executors.newFixedThreadPool(4);
        List<Future<?>> fs = new ArrayList<>();
        int chunk = 2000;
        for (int s = 0; s < ids.size(); s += chunk) {
            final int st = s;
            fs.add(ex.submit(() -> {
                for (int i = st; i < Math.min(st + chunk, ids.size()); i++) {
                    try {
                        rows[i] = ColorFeatures.of(ImageIO.read(new File(a[0], ids.get(i) + (a.length > 3 ? a[3] : ".jpg"))));
                    } catch (IOException e) { throw new UncheckedIOException(e); }
                }
            }));
        }
        for (Future<?> f : fs) f.get();
        ex.shutdown();
        ByteBuffer bb = ByteBuffer.allocate(ids.size() * L * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (float[] r : rows) for (float v : r) bb.putFloat(v);
        Files.write(Path.of(a[2]), bb.array());
        System.out.println(ids.size() + " x " + L);
    }
}
