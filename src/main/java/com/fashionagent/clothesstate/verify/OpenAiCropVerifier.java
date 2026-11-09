package com.fashionagent.clothesstate.verify;

import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.Base64;
import java.util.List;
import java.util.Map;
import java.util.Set;
import javax.imageio.IIOImage;
import javax.imageio.ImageIO;
import javax.imageio.ImageWriteParam;
import javax.imageio.ImageWriter;
import javax.imageio.stream.ImageOutputStream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.json.JsonMapper;

/**
 * 애매한 결함 조각을 OpenAI Chat Completions(그림 입력)로 묻습니다. 결함 둘레 조각만 보내고 옷 전체 사진은 보내지 않습니다.
 * 키가 없으면 ready()=false라 아무것도 보내지 않습니다.
 */
@Component
public class OpenAiCropVerifier implements CropVerifier {

    private static final Logger log = LoggerFactory.getLogger(OpenAiCropVerifier.class);
    private static final Set<String> LABELS = Set.of("stain", "tear", "none");
    /** 11-09 공개 시험 사진 비교(gpt-check/gpt_crop_eval.py)와 같은 문장. 바꾸면 다시 재야 합니다. */
    static final String PROMPT = "This is a cropped part of a clothing photo. Is there a stain or a tear/hole in the "
            + "fabric? Printed patterns, logos, shadows, wrinkles, seams and fabric texture are not defects (answer "
            + "none). Answer only JSON: {\"label\": \"stain\" | \"tear\" | \"none\", \"confidence\": 0.0-1.0}";

    private final GptCheckProperties props;
    private final HttpClient http;
    private final JsonMapper json = JsonMapper.builder().build();

    public OpenAiCropVerifier(GptCheckProperties props) {
        this.props = props;
        this.http = HttpClient.newBuilder().connectTimeout(props.timeout()).build();
    }

    @Override
    public boolean ready() {
        return props.enabled();
    }

    @Override
    public Verdict verify(BufferedImage crop, String modelKind) {
        if (!ready()) {
            return null;
        }
        try {
            String body = json.writeValueAsString(request(jpegBase64(crop)));
            HttpRequest req = HttpRequest.newBuilder(URI.create(props.baseUrl() + "/chat/completions"))
                    .timeout(props.timeout())
                    .header("Authorization", "Bearer " + props.apiKey())
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(body))
                    .build();
            for (int attempt = 1; ; attempt++) {
                HttpResponse<String> res = http.send(req, HttpResponse.BodyHandlers.ofString());
                Verdict v = res.statusCode() == 200 ? parse(res.body(), props.model()) : null;
                if (v != null || attempt >= 2 || (res.statusCode() >= 400 && res.statusCode() < 500 && res.statusCode() != 429)) {
                    if (v == null) {
                        log.warn("GPT 확인 실패: HTTP {}", res.statusCode());
                    }
                    return v;
                }
            }
        } catch (IOException | RuntimeException e) {
            log.warn("GPT 확인 실패: {}", e.getClass().getSimpleName());
            return null;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return null;
        }
    }

    Map<String, Object> request(String jpegBase64) {
        return Map.of(
                "model", props.model(),
                "response_format", Map.of("type", "json_object"),
                "messages", List.of(Map.of("role", "user", "content", List.of(
                        Map.of("type", "text", "text", PROMPT),
                        Map.of("type", "image_url", "image_url",
                                Map.of("url", "data:image/jpeg;base64," + jpegBase64, "detail", "high"))))));
    }

    /** 응답 JSON → 판단. 모양이 다르면 null. */
    Verdict parse(String responseBody, String model) {
        JsonNode content = json.readTree(responseBody).path("choices").path(0).path("message").path("content");
        if (!content.isString()) {
            return null;
        }
        JsonNode answer;
        try {
            answer = json.readTree(content.asString());
        } catch (RuntimeException e) {
            return null;
        }
        String label = answer.path("label").asString("").strip().toLowerCase();
        if (!LABELS.contains(label)) {
            return null;
        }
        double conf = Math.max(0, Math.min(1, answer.path("confidence").asDouble(0)));
        return new Verdict(label, conf, model);
    }

    /** JPEG 품질 0.9(시험과 같게). */
    static String jpegBase64(BufferedImage img) throws IOException {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        ImageWriter writer = ImageIO.getImageWritersByFormatName("jpg").next();
        try (ImageOutputStream ios = ImageIO.createImageOutputStream(out)) {
            writer.setOutput(ios);
            ImageWriteParam p = writer.getDefaultWriteParam();
            p.setCompressionMode(ImageWriteParam.MODE_EXPLICIT);
            p.setCompressionQuality(0.9f);
            writer.write(null, new IIOImage(img, null, null), p);
        } finally {
            writer.dispose();
        }
        return Base64.getEncoder().encodeToString(out.toByteArray());
    }
}
