package com.fashionagent.clothesstate.embed;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.Resource;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.json.JsonMapper;

/**
 * 옷 박스 이미지 벡터(FashionSigLIP)와 옷 이름 글 벡터("a photo of a shirt" 등)의 코사인으로 종류를 봅니다(zero-shot).
 * 글 모델은 서버에서 돌리지 않고 미리 계산한 벡터 파일만 읽습니다. 이름은 옷 종류 모델 클래스와 같고(shirt…shoes),
 * none(빈 벽·바닥·방·얼굴)은 옷이 아닌 것입니다.
 *
 * <p>2026-11-12 측정(DeepFashion2 입은 사진 364벌, 중고 의류 CC0 597벌): 입은 사진에서 옷 종류 모델 70%, 글 벡터 82~85%.
 * 내려놓은 사진은 모델이 93~95%로 강해서 모델 확신도가 낮을 때만 글 벡터를 씁니다.
 */
public class GarmentTextCheck {

    private static final Logger log = LoggerFactory.getLogger(GarmentTextCheck.class);
    public static final String NONE = "none";

    private final List<String> names = new ArrayList<>();
    private final List<float[]> vectors = new ArrayList<>();
    private final TextCheckProperties props;

    public GarmentTextCheck(Resource file, TextCheckProperties props) {
        this.props = props;
        if (file == null || !file.exists()) {
            log.warn("옷 이름 글 벡터 파일이 없어요. 글 벡터 확인 없이 응답해요.");
            return;
        }
        try (InputStream in = file.getInputStream()) {
            JsonNode root = JsonMapper.builder().build().readTree(in);
            for (JsonNode p : root.path("prompts")) {
                JsonNode v = p.path("vector");
                float[] f = new float[v.size()];
                for (int i = 0; i < f.length; i++) {
                    f[i] = (float) v.get(i).asDouble();
                }
                names.add(p.path("name").asString());
                vectors.add(f);
            }
            log.info("옷 이름 글 벡터 {}개를 읽었어요.", names.size());
        } catch (IOException | RuntimeException e) {
            log.warn("옷 이름 글 벡터를 읽지 못했어요: {}", e.toString());
            names.clear();
            vectors.clear();
        }
    }

    public boolean ready() {
        return !vectors.isEmpty();
    }

    public TextCheckProperties props() {
        return props;
    }

    /** 이름마다 가장 높은 코사인(문장 여러 개 중 최대). 벡터 길이가 다르면 빈 맵. */
    public Map<String, Double> scores(float[] image) {
        Map<String, Double> out = new LinkedHashMap<>();
        if (image == null) {
            return out;
        }
        for (int k = 0; k < vectors.size(); k++) {
            float[] t = vectors.get(k);
            if (t.length != image.length) {
                return new LinkedHashMap<>();
            }
            double s = 0;
            for (int i = 0; i < t.length; i++) {
                s += (double) t[i] * image[i];
            }
            out.merge(names.get(k), s, Math::max);
        }
        return out;
    }

    /** 가장 맞는 옷 이름(none 제외)과, none 점수가 그보다 얼마나 높은지. */
    public Result check(float[] image) {
        Map<String, Double> s = scores(image);
        String best = null;
        double bestScore = Double.NEGATIVE_INFINITY;
        for (var e : s.entrySet()) {
            if (!NONE.equals(e.getKey()) && e.getValue() > bestScore) {
                best = e.getKey();
                bestScore = e.getValue();
            }
        }
        Double none = s.get(NONE);
        double margin = best == null || none == null ? Double.NEGATIVE_INFINITY : none - bestScore;
        return new Result(best, margin);
    }

    /** name: 옷 종류 모델 클래스 이름(shirt…shoes), 못 정하면 null. noneMargin > 기준이면 옷이 아님. */
    public record Result(String name, double noneMargin) {
        public boolean notGarment(double threshold) {
            return noneMargin > threshold;
        }
    }
}
