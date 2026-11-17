package com.fashionagent.clothesstate.pattern;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.List;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.Resource;

/**
 * 옷 무늬 모델(YOLO26n-cls ONNX, 8종)을 이 서버 안에서 돌립니다. 출력은 클래스별 확률(softmax)입니다.
 * 모델 파일이 없으면 {@link #ready()}가 false이고 pattern은 null입니다.
 *
 * <p>FashionSigLIP LoRA 무늬 모델(clothes.pattern.lora-model)도 있으면 두 확률을 loraWeight : 1 - loraWeight로 섞고
 * loraSureThreshold로 정합니다(2026-11-17, 입은 옷 시험 0.845 → 0.900, docs/lab.md 7번). LoRA 쪽은 학습과 같게
 * BICUBIC으로 줄이고, 정규화는 모델 안에 들어 있어 입력은 RGB 0~1입니다.
 */
public class OnnxPatternClassifier implements PatternClassifier, AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(OnnxPatternClassifier.class);
    static final String PLAIN = "plain";

    private final PatternProperties props;
    private final List<String> classes;
    private final OrtEnvironment env;
    private final OrtSession session;
    private final String inputName;
    private final OrtSession lora;
    private final String loraInput;

    public OnnxPatternClassifier(Resource model, PatternProperties props) {
        this(model, null, props);
    }

    /** loraPath: LoRA ONNX 파일 경로(약 350MB라 힙으로 읽지 않고 파일에서 바로 엶). 비었거나 없으면 YOLO만. */
    public OnnxPatternClassifier(Resource model, String loraPath, PatternProperties props) {
        this.props = props;
        this.classes = List.copyOf(props.classes());
        OrtEnvironment e = null;
        OrtSession s = null;
        String in = null;
        if (model != null && model.exists()) {
            try (InputStream is = model.getInputStream()) {
                e = com.fashionagent.clothesstate.OrtSupport.env();
                s = e.createSession(is.readAllBytes(), com.fashionagent.clothesstate.OrtSupport.options());
                in = s.getInputNames().iterator().next();
                log.info("무늬 모델을 읽었어요: {} ({}종)", model.getDescription(), classes.size());
            } catch (IOException | OrtException ex) {
                log.warn("무늬 모델을 읽지 못했어요: {}", ex.toString());
                s = null;
            }
        } else {
            log.warn("무늬 모델 파일이 없어요({}). pattern 없이 응답해요.", model == null ? "" : model.getDescription());
        }
        this.env = e;
        this.session = s;
        this.inputName = in;
        OrtSession l = null;
        String li = null;
        boolean hasPath = loraPath != null && !loraPath.isBlank();
        if (s != null && hasPath && java.nio.file.Files.isRegularFile(java.nio.file.Path.of(loraPath))) {
            try {
                l = e.createSession(loraPath, com.fashionagent.clothesstate.OrtSupport.options());
                li = l.getInputNames().iterator().next();
                log.info("무늬 LoRA 모델도 읽었어요: {} (섞기 {} : {}, 기준 {})", loraPath,
                        props.loraWeight(), 1 - props.loraWeight(), props.loraSureThreshold());
            } catch (OrtException ex) {
                log.warn("무늬 LoRA 모델을 읽지 못했어요. 예전 무늬 모델만 써요: {}", ex.toString());
                l = null;
            }
        } else if (hasPath && s != null) {
            log.info("무늬 LoRA 모델 파일이 없어요({}). 예전 무늬 모델만 써요.", loraPath);
        }
        this.lora = l;
        this.loraInput = li;
    }

    @Override
    public boolean mixed() {
        return lora != null;
    }

    @Override
    public boolean ready() {
        return session != null;
    }

    @Override
    public String version() {
        return lora != null ? props.loraVersion() : props.version();
    }

    @Override
    public Result classify(BufferedImage image, double x, double y, double w, double h) {
        if (session == null) {
            throw new IllegalStateException("pattern model not loaded");
        }
        int size = props.inputSize();
        float[] b = run(session, inputName, PatternInput.prepare(image, x, y, w, h, props.margin(), size), size);
        if (lora == null) {
            return decide(b, classes, props.sureThreshold());
        }
        float[] l = run(lora, loraInput, PatternInput.prepare(image, x, y, w, h, props.margin(), size, true), size);
        return decide(mix(l, b, props.loraWeight()), classes, props.loraSureThreshold());
    }

    /** LoRA 모델 확률만(섞기 전). 입력 맞추기 시험용. */
    float[] loraProbs(BufferedImage image, double x, double y, double w, double h) {
        int size = props.inputSize();
        return run(lora, loraInput, PatternInput.prepare(image, x, y, w, h, props.margin(), size, true), size);
    }

    private float[] run(OrtSession s, String in, float[] chw, int size) {
        try (OnnxTensor t = OnnxTensor.createTensor(env, FloatBuffer.wrap(chw), new long[] {1, 3, size, size});
                OrtSession.Result r = s.run(Map.of(in, t))) {
            Object v = r.get(0).getValue();
            if (!(v instanceof float[][] m) || m[0].length != classes.size()) {
                throw new IllegalStateException("pattern output shape not [1, " + classes.size() + "]");
            }
            return m[0];
        } catch (OrtException ex) {
            throw new IllegalStateException("pattern model run failed: " + ex.getMessage(), ex);
        }
    }

    /** w × lora + (1 - w) × yolo. */
    static float[] mix(float[] lora, float[] yolo, double w) {
        float[] out = new float[lora.length];
        for (int i = 0; i < out.length; i++) {
            out[i] = (float) (w * lora[i] + (1 - w) * yolo[i]);
        }
        return out;
    }

    /** 1등이 무지면 plain, 무늬면 확률이 기준 이상일 때만 그 무늬, 아니면 null(모름). */
    static Result decide(float[] probs, List<String> classes, double sure) {
        int best = 0;
        for (int i = 1; i < probs.length; i++) {
            if (probs[i] > probs[best]) {
                best = i;
            }
        }
        String top = classes.get(best);
        double p = Math.round(probs[best] * 100) / 100.0;
        String pattern = PLAIN.equals(top) || probs[best] >= sure ? top : null;
        return new Result(pattern, top, p, ranked(probs, classes, 3));
    }

    /** 확률 높은 순 k개 이름. */
    static List<String> ranked(float[] probs, List<String> classes, int k) {
        return java.util.stream.IntStream.range(0, probs.length).boxed()
                .sorted((a, c) -> Float.compare(probs[c], probs[a])).limit(k).map(classes::get).toList();
    }

    @Override
    public void close() throws OrtException {
        if (lora != null) {
            lora.close();
        }
        if (session != null) {
            session.close();
        }
    }
}
