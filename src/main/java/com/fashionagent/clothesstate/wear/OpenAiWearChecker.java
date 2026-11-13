package com.fashionagent.clothesstate.wear;

import com.fashionagent.clothesstate.verify.GptCheckProperties;
import com.fashionagent.clothesstate.verify.OpenAiCropVerifier;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.json.JsonMapper;

/**
 * 입은 옷 조각을 OpenAI Chat Completions(그림 입력)로 보내 옷 실수를 묻습니다. 키가 없거나 꺼져 있으면 ready()=false.
 * 조각은 저장하지 않습니다(몸이 들어간 사진이라 학습 후보로도 남기지 않음).
 */
@Component
public class OpenAiWearChecker implements WearChecker {

    private static final Logger log = LoggerFactory.getLogger(OpenAiWearChecker.class);

    static final String PROMPT = """
            This is a crop of a person wearing a %s, seen from the %s. Grey areas are hidden on purpose (privacy). \
            Check ONLY for these common dressing mistakes:
            - inside_out: worn inside out (inner seams with overlocked edges or seam allowances showing on the outside, \
            care label visible on the outside, print looks faded or mirrored from behind the fabric).
            - backwards: worn front-to-back. Examples: %s
            - buttons_misaligned: buttons fastened into the wrong buttonholes (the two front edges end at different \
            heights at the hem, the placket is skewed, a button or hole is left over).
            - tag_out: a clothing label or tag sticking out at the neck, waist or side seam.
            Report a mistake only when you can clearly see evidence of it. Normal designs are not mistakes: garments \
            worn open on purpose, decorative exposed seams, asymmetric hems, prints on the back, a shirt with no buttons. \
            Answer only JSON: {"issues": [{"kind": "inside_out" | "backwards" | "buttons_misaligned" | "tag_out", \
            "confidence": 0.0-1.0}]}. Use an empty list if nothing is wrong.""";

    static final String BACKWARDS_FRONT = "the neckline in front is high like a back neckline while a label shows at "
            + "the front neck; a hood or back print appears on the front; trousers with the zipper/fly at the back.";
    static final String BACKWARDS_BACK = "buttons, a zipper, a front pocket, a V-neck or a fly are visible from the "
            + "back; a hood hangs on the chest side.";

    private final GptCheckProperties gpt;
    private final WearCheckProperties props;
    private final HttpClient http;
    private final JsonMapper json = JsonMapper.builder().build();

    public OpenAiWearChecker(GptCheckProperties gpt, WearCheckProperties props) {
        this.gpt = gpt;
        this.props = props;
        this.http = HttpClient.newBuilder().connectTimeout(props.timeout()).build();
    }

    @Override
    public boolean ready() {
        return props.enabled() && gpt.enabled();
    }

    String model() {
        return props.model() == null || props.model().isBlank() ? gpt.model() : props.model();
    }

    @Override
    public List<Issue> check(BufferedImage garment, String garmentName, String view) {
        if (!ready()) {
            return null;
        }
        try {
            String body = json.writeValueAsString(request(OpenAiCropVerifier.jpegBase64(garment), garmentName, view));
            HttpRequest req = HttpRequest.newBuilder(URI.create(gpt.baseUrl() + "/chat/completions"))
                    .timeout(props.timeout())
                    .header("Authorization", "Bearer " + gpt.apiKey())
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(body))
                    .build();
            HttpResponse<String> res = http.send(req, HttpResponse.BodyHandlers.ofString());
            List<Issue> out = res.statusCode() == 200 ? parse(res.body(), model()) : null;
            if (out == null) {
                log.warn("GPT 옷 실수 확인 실패: HTTP {}", res.statusCode());
            }
            return out;
        } catch (IOException | RuntimeException e) {
            log.warn("GPT 옷 실수 확인 실패: {}", e.getClass().getSimpleName());
            return null;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return null;
        }
    }

    static String prompt(String garmentName, String view) {
        boolean back = "back".equals(view);
        String name = garmentName == null || garmentName.isBlank() ? "garment" : garmentName;
        return PROMPT.formatted(name, back ? "back" : "front", back ? BACKWARDS_BACK : BACKWARDS_FRONT);
    }

    Map<String, Object> request(String jpegBase64, String garmentName, String view) {
        return Map.of(
                "model", model(),
                "response_format", Map.of("type", "json_object"),
                "messages", List.of(Map.of("role", "user", "content", List.of(
                        Map.of("type", "text", "text", prompt(garmentName, view)),
                        Map.of("type", "image_url", "image_url",
                                Map.of("url", "data:image/jpeg;base64," + jpegBase64, "detail", "high"))))));
    }

    /** 응답 JSON → 실수 목록. 모양이 다르면 null. 모르는 종류는 버리고, 같은 종류는 확신도 높은 하나만. */
    static List<Issue> parse(String responseBody, String model) {
        JsonMapper json = JsonMapper.builder().build();
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
        JsonNode issues = answer.path("issues");
        if (!issues.isArray()) {
            return null;
        }
        List<Issue> out = new ArrayList<>();
        for (JsonNode n : issues) {
            String kind = n.path("kind").asString("").strip().toLowerCase();
            if (!KINDS.contains(kind)) {
                continue;
            }
            double conf = Math.max(0, Math.min(1, n.path("confidence").asDouble(0)));
            out.removeIf(i -> i.kind().equals(kind) && i.confidence() < conf);
            if (out.stream().noneMatch(i -> i.kind().equals(kind))) {
                out.add(new Issue(kind, conf, model));
            }
        }
        return out;
    }
}
