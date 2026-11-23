package com.fashionagent.clothesstate.embed;

import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.ArrayList;
import java.util.Base64;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import javax.imageio.ImageIO;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.Resource;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.json.JsonMapper;

/**
 * 애매한 옷만 SigLIP2 So400m 벡터 서버(so400m-embed/)에 박스 조각을 보내 벡터를 받고, So400m probe로 종류를 다시 고릅니다.
 * 서버가 없거나 느리거나 답이 이상하면 null을 돌려주고, 부르는 쪽은 지금 결과를 그대로 씁니다(사진 답이 늦어지지 않게).
 */
public class So400mAssist {

    private static final Logger log = LoggerFactory.getLogger(So400mAssist.class);
    private static final ExecutorService WORKERS = Executors.newVirtualThreadPerTaskExecutor();
    /** 꺼진 보조 경로(시험·설정 없음). */
    public static final So400mAssist OFF = new So400mAssist(null, null);

    private final So400mAssistProperties props;
    private final List<String> names = new ArrayList<>();
    private final List<float[]> vectors = new ArrayList<>();
    private final HttpClient http;
    private final JsonMapper json = JsonMapper.builder().build();
    /** 벡터 서버 입력 크기(So400m/14-224). */
    static final int SIZE = 224;

    public So400mAssist(So400mAssistProperties props, Resource probe) {
        this.props = props;
        this.http = props == null ? null : HttpClient.newBuilder().connectTimeout(props.timeout()).build();
        if (props == null || props.url() == null || props.url().isBlank()) {
            return;
        }
        if (probe == null || !probe.exists()) {
            log.warn("So400m 종류 벡터 파일이 없어서 보조 경로를 꺼요.");
            return;
        }
        try (InputStream in = probe.getInputStream()) {
            for (JsonNode p : json.readTree(in).path("prompts")) {
                JsonNode v = p.path("vector");
                float[] f = new float[v.size()];
                for (int i = 0; i < f.length; i++) {
                    f[i] = (float) v.get(i).asDouble();
                }
                names.add(p.path("name").asString());
                vectors.add(f);
            }
            log.info("So400m 보조 경로를 켰어요: 종류 벡터 {}개, 기다리는 시간 {}ms", vectors.size(), props.timeout().toMillis());
        } catch (IOException | RuntimeException e) {
            log.warn("So400m 종류 벡터를 읽지 못해 보조 경로를 꺼요: {}", e.toString());
            names.clear();
            vectors.clear();
        }
    }

    public boolean enabled() {
        return !vectors.isEmpty();
    }

    public So400mAssistProperties props() {
        return props;
    }

    /** 다시 물을 옷인지: 켜져 있고, 모델 확신도 < maxConfidence, 지금 종류 점수 1·2등 차이 < maxGap. */
    public boolean wanted(Double confidence, Map<String, Double> kindScores) {
        return enabled() && confidence != null && confidence < props.maxConfidence()
                && gap(kindScores) < props.maxGap();
    }

    /** 옷 이름 점수(none·신발 빼고)의 1등 - 2등. 두 개가 안 되면 무한대(묻지 않음). */
    public static double gap(Map<String, Double> scores) {
        double a = Double.NEGATIVE_INFINITY;
        double b = Double.NEGATIVE_INFINITY;
        for (var e : scores.entrySet()) {
            if (GarmentTextCheck.NONE.equals(e.getKey()) || "shoes".equals(e.getKey())) {
                continue;
            }
            double v = e.getValue();
            if (v > a) {
                b = a;
                a = v;
            } else if (v > b) {
                b = v;
            }
        }
        return b == Double.NEGATIVE_INFINITY ? Double.POSITIVE_INFINITY : a - b;
    }

    /**
     * 박스들(x, y, w, h: 0~1)을 동시에 물어 고른 종류 이름을 같은 순서로 돌려줍니다. 시간이 넘었거나 실패한 것은 null.
     * 전체를 timeout 안에 끝냅니다.
     */
    public List<String> pick(BufferedImage img, List<double[]> boxes) {
        List<String> out = new ArrayList<>(Collections.nCopies(boxes.size(), null));
        if (!enabled() || boxes.isEmpty()) {
            return out;
        }
        List<CompletableFuture<String>> fs = new ArrayList<>();
        for (double[] b : boxes) {
            fs.add(CompletableFuture.supplyAsync(() -> pickOne(img, b), WORKERS)
                    .completeOnTimeout(null, props.timeout().toMillis(), TimeUnit.MILLISECONDS)
                    .exceptionally(e -> null));
        }
        for (int i = 0; i < fs.size(); i++) {
            out.set(i, fs.get(i).join());
        }
        return out;
    }

    String pickOne(BufferedImage img, double[] b) {
        try {
            float[] v = embed(img, b);
            return v == null ? null : best(v);
        } catch (IOException | RuntimeException e) {
            log.warn("So400m 벡터 서버 실패, 지금 결과를 써요: {}", e.toString());
            return null;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return null;
        }
    }

    private float[] embed(BufferedImage img, double[] b) throws IOException, InterruptedException {
        // 측정·학습 때와 같게: 넓혀 자르고 회색 정사각형 → PIL BICUBIC과 같은 방법으로 224 → PNG(무손실).
        // JPEG로 보내면 압축 손실 때문에 So400m 벡터가 코사인 0.93~0.98로 달라졌음(2026-11-19 측정).
        int[][] rgb = EmbedInput.squareCrop(img, b[0], b[1], b[2], b[3], props.margin());
        float[][] planes = EmbedInput.resize(rgb, SIZE);
        BufferedImage sq = new BufferedImage(SIZE, SIZE, BufferedImage.TYPE_INT_RGB);
        for (int i = 0; i < SIZE * SIZE; i++) {
            int r = clamp255(planes[0][i]);
            int g = clamp255(planes[1][i]);
            int bl = clamp255(planes[2][i]);
            sq.setRGB(i % SIZE, i / SIZE, (r << 16) | (g << 8) | bl);
        }
        ByteArrayOutputStream png = new ByteArrayOutputStream();
        ImageIO.write(sq, "png", png);
        byte[] bytes = png.toByteArray();
        String body = json.writeValueAsString(Map.of("imageBase64", Base64.getEncoder().encodeToString(bytes)));
        HttpRequest req = HttpRequest.newBuilder(URI.create(props.url().replaceAll("/+$", "") + "/embed"))
                .timeout(props.timeout()).header("content-type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(body)).build();
        HttpResponse<String> res = http.send(req, HttpResponse.BodyHandlers.ofString());
        if (res.statusCode() != 200) {
            log.warn("So400m 벡터 서버 응답 {}", res.statusCode());
            return null;
        }
        JsonNode v = json.readTree(res.body()).path("vector");
        float[] f = new float[v.size()];
        for (int i = 0; i < f.length; i++) {
            f[i] = (float) v.get(i).asDouble();
        }
        return f.length == vectors.getFirst().length ? f : null;
    }

    private static int clamp255(float v) {
        return Math.max(0, Math.min(255, Math.round(v)));
    }

    /** 이름마다 최대 코사인 중 가장 높은 옷 이름(none·신발 제외). */
    String best(float[] image) {
        String best = null;
        double bestScore = Double.NEGATIVE_INFINITY;
        for (int k = 0; k < vectors.size(); k++) {
            String n = names.get(k);
            if (GarmentTextCheck.NONE.equals(n) || "shoes".equals(n)) {
                continue;
            }
            float[] t = vectors.get(k);
            double s = 0;
            for (int i = 0; i < t.length; i++) {
                s += (double) t[i] * image[i];
            }
            if (s > bestScore) {
                best = n;
                bestScore = s;
            }
        }
        return best;
    }
}
