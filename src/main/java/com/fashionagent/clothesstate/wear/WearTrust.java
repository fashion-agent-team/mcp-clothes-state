package com.fashionagent.clothesstate.wear;

import com.fashionagent.clothesstate.verify.GptCheckProperties;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.function.DoubleSupplier;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.json.JsonMapper;
import tools.jackson.databind.node.ArrayNode;
import tools.jackson.databind.node.ObjectNode;

/**
 * 옷 실수 종류마다 "우리 모델을 GPT 대신 믿어도 되는지"를 잽니다(WearLearnProperties). 모델 버전이 바뀌면 기록을 비웁니다.
 * 기록은 확률과 GPT 답(참/거짓)뿐이고 사진·사용자 정보는 없습니다.
 */
@Component
public final class WearTrust {

    private static final Logger log = LoggerFactory.getLogger(WearTrust.class);

    /** GPT 답 하나: 모델 확률, GPT가 실수라고 했는지. */
    record Answer(double model, boolean gpt) {
    }

    /** 종류별 지금 상태(로그·문서용). */
    public record Stats(int samples, int positives, double specificity, double recall, boolean trusted) {
    }

    private final WearLearnProperties props;
    private final Path file;
    private final String version;
    private final DoubleSupplier random;
    private final Map<String, Deque<Answer>> answers = new LinkedHashMap<>();
    private final JsonMapper json = JsonMapper.builder().build();

    @Autowired
    public WearTrust(WearLearnProperties props, GptCheckProperties gpt, WearModelProperties model) {
        this(props, stateFile(props, gpt), model.version(), Math::random);
    }

    /** 시험용(기록 파일·버전·난수를 직접 넣음). */
    public WearTrust(WearLearnProperties props, Path file, String version, DoubleSupplier random) {
        this.props = props;
        this.file = file;
        this.version = version;
        this.random = random;
        load();
    }

    private static Path stateFile(WearLearnProperties props, GptCheckProperties gpt) {
        if (props.stateFile() != null && !props.stateFile().isBlank()) {
            return Path.of(props.stateFile());
        }
        if (gpt.candidateDir() == null || gpt.candidateDir().isBlank()) {
            return null; // 저장 안 함(재시작하면 처음부터)
        }
        return Path.of(gpt.candidateDir(), "wear-trust.json");
    }

    public boolean enabled() {
        return props.enabled();
    }

    /** GPT 답을 하나 기록. */
    public synchronized void record(String kind, double modelProbability, boolean gptSaysMistake) {
        if (!props.enabled()) {
            return;
        }
        Deque<Answer> q = answers.computeIfAbsent(kind, k -> new ArrayDeque<>());
        q.addLast(new Answer(modelProbability, gptSaysMistake));
        while (q.size() > props.window()) {
            q.removeFirst();
        }
    }

    public synchronized Stats stats(String kind) {
        Deque<Answer> q = answers.getOrDefault(kind, new ArrayDeque<>());
        int neg = 0, negOk = 0, pos = 0, posOk = 0;
        for (Answer a : q) {
            if (a.gpt()) {
                pos++;
                posOk += a.model() >= 0.5 ? 1 : 0;
            } else {
                neg++;
                negOk += a.model() < 0.5 ? 1 : 0;
            }
        }
        double spec = neg == 0 ? 0 : (double) negOk / neg;
        double recall = pos == 0 ? 0 : (double) posOk / pos;
        boolean trusted = q.size() >= props.minSamples() && pos >= props.minPositives()
                && spec >= props.minSpecificity() && recall >= props.minRecall();
        return new Stats(q.size(), pos, spec, recall, trusted);
    }

    /** 믿게 된 실수 종류가 하나라도 있는지. */
    public synchronized boolean anyTrusted() {
        return props.enabled() && answers.keySet().stream().anyMatch(k -> stats(k).trusted());
    }

    /**
     * 이 옷을 GPT에게 묻지 않아도 되는지. 볼 수 있는 실수(probs)가 하나도 없으면 false(모델이 아무것도 안 봄).
     * 모두 믿을 만하고 모두 확신할 때만 true, 그래도 auditRate만큼은 false(물어봄).
     */
    public boolean canSkip(Map<String, Double> probs) {
        if (!props.enabled() || probs == null || probs.isEmpty()) {
            return false;
        }
        for (var e : probs.entrySet()) {
            double p = e.getValue();
            if (p > props.sureLow() && p < props.sureHigh()) {
                return false;
            }
            if (!stats(e.getKey()).trusted()) {
                return false;
            }
        }
        return random.getAsDouble() >= props.auditRate();
    }

    /** 이번 답을 학습 후보로 저장할지: GPT가 실수라 함·모델과 엇갈림·모델이 애매함이면 늘, 둘 다 확신하는 정상이면 saveAgreeRate만큼. */
    public boolean worthSaving(double modelProbability, boolean gptSaysMistake) {
        if (!props.enabled()) {
            return false;
        }
        boolean modelSays = modelProbability >= 0.5;
        if (gptSaysMistake || modelSays != gptSaysMistake
                || (modelProbability > props.sureLow() && modelProbability < props.sureHigh())) {
            return true;
        }
        return random.getAsDouble() < props.saveAgreeRate();
    }

    /** 기록을 파일로(GPT 답을 받은 요청마다 한 번). 실패해도 검사는 계속. */
    public synchronized void save() {
        if (file == null || !props.enabled()) {
            return;
        }
        try {
            ObjectNode root = json.createObjectNode();
            root.put("version", version);
            ObjectNode kinds = root.putObject("kinds");
            answers.forEach((k, q) -> {
                ArrayNode arr = kinds.putArray(k);
                for (Answer a : q) {
                    arr.addArray().add(Math.round(a.model() * 1000) / 1000.0).add(a.gpt());
                }
            });
            Files.createDirectories(file.toAbsolutePath().getParent());
            Path tmp = file.resolveSibling(file.getFileName() + ".tmp");
            Files.writeString(tmp, json.writeValueAsString(root));
            Files.move(tmp, file, java.nio.file.StandardCopyOption.REPLACE_EXISTING,
                    java.nio.file.StandardCopyOption.ATOMIC_MOVE);
        } catch (IOException | RuntimeException e) {
            log.warn("옷 실수 믿음 기록 저장 실패: {}", e.getClass().getSimpleName());
        }
    }

    private void load() {
        if (file == null || !props.enabled() || !Files.isRegularFile(file)) {
            return;
        }
        try {
            JsonNode root = json.readTree(Files.readString(file));
            if (!version.equals(root.path("version").asString(""))) {
                log.info("옷 실수 모델이 바뀌어({} → {}) 믿음 기록을 처음부터 다시 셈", root.path("version").asString(""), version);
                return;
            }
            for (var e : root.path("kinds").properties()) {
                for (JsonNode a : e.getValue()) {
                    record(e.getKey(), a.path(0).asDouble(), a.path(1).asBoolean());
                }
            }
            answers.keySet().forEach(k -> log.info("옷 실수 믿음 {}: {}", k, stats(k)));
        } catch (IOException | RuntimeException e) {
            log.warn("옷 실수 믿음 기록 읽기 실패: {}", e.getClass().getSimpleName());
        }
    }
}
