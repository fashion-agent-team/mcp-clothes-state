package com.fashionagent.clothesstate.pattern;

import com.fashionagent.clothesstate.verify.GptCheckProperties;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
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
 * 무늬 모델이 "모름"이라고 한 옷을 OpenAI Chat Completions(그림 입력)로 묻습니다(정지원 2026-11-11). 키·모델·시간은
 * 얼룩 확인과 같은 clothes.gpt.*를 씁니다. 옷 박스만 긴 변 512로 줄여 보내고 저장하지 않습니다.
 */
@Component
public class OpenAiPatternVerifier implements PatternVerifier {

    private static final Logger log = LoggerFactory.getLogger(OpenAiPatternVerifier.class);
    static final Set<String> NAMES = Set.of("plain", "stripe", "check", "floral", "dot", "graphic", "animal", "other");
    /** 공개 시험(training/scripts/pattern/gpt_pattern_eval.py)과 같은 문장. 바꾸면 다시 재야 합니다. */
    static final String PROMPT = "This photo shows one garment. What is the main print or pattern of its fabric? "
            + "Choose one: plain (solid color, no print; a small logo or texture still counts as plain), stripe, "
            + "check (plaid, gingham, houndstooth, argyle), floral (flowers or leaves), dot (polka dots or small spots), "
            + "graphic (letters, numbers, cartoon or picture print), animal (leopard, zebra, snake skin), other (any "
            + "other print: abstract, geometric, paisley, camouflage). Answer only JSON: {\"pattern\": \"<one of the "
            + "words>\", \"confidence\": 0.0-1.0}";
    static final int MAX_SIDE = 512;

    private final GptCheckProperties props;
    private final HttpClient http;
    private final JsonMapper json = JsonMapper.builder().build();

    public OpenAiPatternVerifier(GptCheckProperties props) {
        this.props = props;
        this.http = HttpClient.newBuilder().connectTimeout(props.timeout()).build();
    }

    @Override
    public boolean ready() {
        return props.enabled();
    }

    @Override
    public String ask(BufferedImage garment) {
        if (!ready()) {
            return null;
        }
        try {
            String body = json.writeValueAsString(request(jpegBase64(shrink(garment, MAX_SIDE))));
            HttpRequest req = HttpRequest.newBuilder(URI.create(props.baseUrl() + "/chat/completions"))
                    .timeout(props.timeout())
                    .header("Authorization", "Bearer " + props.apiKey())
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(body))
                    .build();
            HttpResponse<String> res = http.send(req, HttpResponse.BodyHandlers.ofString());
            String p = res.statusCode() == 200 ? parse(res.body()) : null;
            if (p == null) {
                log.warn("GPT 무늬 확인 실패: HTTP {}", res.statusCode());
            }
            return p;
        } catch (IOException | RuntimeException e) {
            log.warn("GPT 무늬 확인 실패: {}", e.getClass().getSimpleName());
            return null;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return null;
        }
    }

    Map<String, Object> request(String jpegBase64) {
        return Map.of(
                "model", props.model(),
                "max_completion_tokens", 60,
                "response_format", Map.of("type", "json_object"),
                "messages", List.of(Map.of("role", "user", "content", List.of(
                        Map.of("type", "text", "text", PROMPT),
                        Map.of("type", "image_url", "image_url",
                                Map.of("url", "data:image/jpeg;base64," + jpegBase64, "detail", "low"))))));
    }

    /** 응답 JSON → 무늬 이름. 모양이 다르거나 목록에 없는 이름이면 null. */
    String parse(String responseBody) {
        JsonNode content = json.readTree(responseBody).path("choices").path(0).path("message").path("content");
        if (!content.isString()) {
            return null;
        }
        try {
            String p = json.readTree(content.asString()).path("pattern").asString("").strip().toLowerCase();
            return NAMES.contains(p) ? p : null;
        } catch (RuntimeException e) {
            return null;
        }
    }

    static BufferedImage shrink(BufferedImage img, int maxSide) {
        double s = Math.min(1.0, (double) maxSide / Math.max(img.getWidth(), img.getHeight()));
        int w = Math.max(1, (int) Math.round(img.getWidth() * s));
        int h = Math.max(1, (int) Math.round(img.getHeight() * s));
        BufferedImage out = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = out.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.drawImage(img, 0, 0, w, h, null);
        g.dispose();
        return out;
    }

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
