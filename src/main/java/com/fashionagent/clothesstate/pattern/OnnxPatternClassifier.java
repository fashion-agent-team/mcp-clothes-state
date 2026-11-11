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
 */
public class OnnxPatternClassifier implements PatternClassifier, AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(OnnxPatternClassifier.class);
    static final String PLAIN = "plain";

    private final PatternProperties props;
    private final List<String> classes;
    private final OrtEnvironment env;
    private final OrtSession session;
    private final String inputName;

    public OnnxPatternClassifier(Resource model, PatternProperties props) {
        this.props = props;
        this.classes = List.copyOf(props.classes());
        OrtEnvironment e = null;
        OrtSession s = null;
        String in = null;
        if (model != null && model.exists()) {
            try (InputStream is = model.getInputStream()) {
                e = OrtEnvironment.getEnvironment();
                s = e.createSession(is.readAllBytes(), new OrtSession.SessionOptions());
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
    }

    @Override
    public boolean ready() {
        return session != null;
    }

    @Override
    public String version() {
        return props.version();
    }

    @Override
    public Result classify(BufferedImage image, double x, double y, double w, double h) {
        if (session == null) {
            throw new IllegalStateException("pattern model not loaded");
        }
        int size = props.inputSize();
        float[] chw = PatternInput.prepare(image, x, y, w, h, props.margin(), size);
        try (OnnxTensor t = OnnxTensor.createTensor(env, FloatBuffer.wrap(chw), new long[] {1, 3, size, size});
                OrtSession.Result r = session.run(Map.of(inputName, t))) {
            Object v = r.get(0).getValue();
            if (!(v instanceof float[][] m) || m[0].length != classes.size()) {
                throw new IllegalStateException("pattern output shape not [1, " + classes.size() + "]");
            }
            return decide(m[0], classes, props.sureThreshold());
        } catch (OrtException ex) {
            throw new IllegalStateException("pattern model run failed: " + ex.getMessage(), ex);
        }
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
        return new Result(pattern, top, p);
    }

    @Override
    public void close() throws OrtException {
        if (session != null) {
            session.close();
        }
    }
}
