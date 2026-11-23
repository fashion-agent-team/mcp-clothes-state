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
    public static final So400mAssist OFF = new So400mAssist(null, null, null);

    private final So400mAssistProperties props;
    private final List<String> names = new ArrayList<>();
    private final List<float[]> vectors = new ArrayList<>();
    private final HttpClient http;
    private final JsonMapper json = JsonMapper.builder().build();
    /** 벡터 서버 입력 크기(So400m/14-224). */
    static final int SIZE = 224;

    private final List<String> colorFamilies = new ArrayList<>();
    private final List<float[]> colorWeights = new ArrayList<>();
    private float[] colorBias;

    public So400mAssist(So400mAssistProperties props, Resource probe) {
        this(props, probe, null);
    }

    public So400mAssist(So400mAssistProperties props, Resource probe, Resource colorModel) {
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
        if (props.color() && colorModel != null && colorModel.exists()) {
            try (InputStream in = colorModel.getInputStream()) {
                JsonNode root = json.readTree(in);
                for (JsonNode f : root.path("families")) {
                    colorFamilies.add(f.asString());
                }
                for (JsonNode row : root.path("weights")) {
                    float[] w = new float[row.size()];
                    for (int i = 0; i < w.length; i++) {
                        w[i] = (float) row.get(i).asDouble();
                    }
                    colorWeights.add(w);
                }
                JsonNode b = root.path("bias");
                colorBias = new float[b.size()];
                for (int i = 0; i < colorBias.length; i++) {
                    colorBias[i] = (float) b.get(i).asDouble();
                }
                if (colorFamilies.size() != colorWeights.size() || colorBias.length != colorWeights.size()) {
                    throw new IllegalStateException("색 분류기 모양이 맞지 않음");
                }
                log.info("So400m 색 분류기를 켰어요: {}계열", colorFamilies.size());
            } catch (IOException | RuntimeException e) {
                log.warn("So400m 색 분류기를 읽지 못해 색은 지금 방식으로 정해요: {}", e.toString());
                colorFamilies.clear();
                colorWeights.clear();
            }
        }
    }

    /** So400m 색 분류기가 켜져 있는지(보조 경로 주소 + clothes.so400m.color=true + 색 분류기 파일). */
    public boolean colorEnabled() {
        return props != null && props.url() != null && !props.url().isBlank() && !colorWeights.isEmpty();
    }

    /** 지금 색 방식의 답(family)을 So400m 색으로 바꿔도 되는지(clothes.so400m.color-override). */
    public boolean overrides(String currentFamily) {
        String list = props == null ? null : props.colorOverride();
        if (list == null || list.isBlank()) {
            return true;
        }
        for (String f : list.split(",")) {
            if (f.trim().equals(currentFamily)) {
                return true;
            }
        }
        return false;
    }

    /** So400m 벡터 → 색 계열(black…purple). 꺼져 있거나 벡터 길이가 다르면 null. */
    public String colorFamily(float[] v) {
        if (!colorEnabled() || v == null || v.length != colorWeights.getFirst().length) {
            return null;
        }
        int best = -1;
        double bestScore = Double.NEGATIVE_INFINITY;
        for (int k = 0; k < colorWeights.size(); k++) {
            float[] w = colorWeights.get(k);
            double s = colorBias[k];
            for (int i = 0; i < w.length; i++) {
                s += (double) w[i] * v[i];
            }
            if (s > bestScore) {
                best = k;
                bestScore = s;
            }
        }
        return colorFamilies.get(best);
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
        if (!enabled()) {
            return out;
        }
        List<float[]> vs = vectors(img, boxes);
        for (int i = 0; i < vs.size(); i++) {
            out.set(i, vs.get(i) == null ? null : best(vs.get(i)));
        }
        return out;
    }

    /**
     * 박스들의 So400m 벡터를 동시에 받아 같은 순서로 돌려줍니다. 시간이 넘었거나 실패한 것은 null. 전체를 timeout 안에 끝냅니다.
     * 종류 다시 고르기와 색 판정이 같은 벡터를 씁니다(같은 옷을 두 번 묻지 않게).
     */
    public List<float[]> vectors(BufferedImage img, List<double[]> boxes) {
        List<float[]> out = new ArrayList<>(Collections.nCopies(boxes.size(), null));
        if (props == null || props.url() == null || props.url().isBlank() || boxes.isEmpty()) {
            return out;
        }
        List<CompletableFuture<float[]>> fs = new ArrayList<>();
        for (double[] b : boxes) {
            fs.add(CompletableFuture.supplyAsync(() -> vectorOne(img, b), WORKERS)
                    .completeOnTimeout(null, props.timeout().toMillis(), TimeUnit.MILLISECONDS)
                    .exceptionally(e -> null));
        }
        for (int i = 0; i < fs.size(); i++) {
            out.set(i, fs.get(i).join());
        }
        return out;
    }

    float[] vectorOne(BufferedImage img, double[] b) {
        try {
            return embed(img, b);
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
        int want = !vectors.isEmpty() ? vectors.getFirst().length : !colorWeights.isEmpty() ? colorWeights.getFirst().length : f.length;
        return f.length == want ? f : null;
    }

    private static int clamp255(float v) {
        return Math.max(0, Math.min(255, Math.round(v)));
    }

    /** 이름마다 최대 코사인 중 가장 높은 옷 이름(none·신발 제외). */
    public String best(float[] image) {
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
