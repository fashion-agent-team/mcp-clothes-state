package com.fashionagent.clothesstate.detect;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;
import com.fashionagent.clothesstate.image.Images;
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
 * 우리가 학습한 얼룩·찢어짐 모델(YOLO26n v6)을 ONNX Runtime으로 이 서버 안에서 돌립니다. 사진을 외부 AI로 보내지 않습니다.
 * 모델 파일이 없으면 {@link #ready()}가 false이고 clothes_inspect는 not_ready를 돌려줍니다.
 */
public class OnnxDefectDetector implements DefectDetector, AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(OnnxDefectDetector.class);

    private final int size;
    private final double nmsIou;
    private final OrtEnvironment env;
    private final OrtSession session;
    private final String inputName;
    /** 모델 입력 높이·너비가 정해져 있지 않으면(dynamic) 학습 때처럼 직사각형으로 넣는다. */
    private final boolean rect;

    public OnnxDefectDetector(Resource model, int size, double nmsIou) {
        this.size = size;
        this.nmsIou = nmsIou;
        OrtEnvironment e = null;
        OrtSession s = null;
        String in = null;
        boolean dynamic = false;
        if (model != null && model.exists()) {
            try (InputStream is = model.getInputStream()) {
                byte[] bytes = is.readAllBytes();
                e = OrtEnvironment.getEnvironment();
                s = e.createSession(bytes, new OrtSession.SessionOptions());
                in = s.getInputNames().iterator().next();
                long[] shape = ((TensorInfo) s.getInputInfo().get(in).getInfo()).getShape();
                dynamic = shape.length == 4 && (shape[2] < 0 || shape[3] < 0);
                log.info("옷 결함 모델을 읽었어요: {} ({} bytes, 입력 {}px, {})", model.getDescription(), bytes.length, size,
                        dynamic ? "직사각형" : "정사각형");
            } catch (IOException | OrtException ex) {
                log.warn("옷 결함 모델을 읽지 못했어요: {}", ex.toString());
                s = null;
            }
        } else {
            log.warn("옷 결함 모델 파일이 없어요({}). clothes_inspect는 not_ready를 돌려줍니다.",
                    model == null ? "-" : model.getDescription());
        }
        this.env = e;
        this.session = s;
        this.inputName = in;
        this.rect = dynamic;
    }

    @Override
    public boolean ready() {
        return session != null;
    }

    @Override
    public List<Detection> detect(BufferedImage image, double minConfidence) {
        if (session == null) {
            throw new IllegalStateException("model not loaded");
        }
        Images.Letterbox lb = Images.letterbox(image, size, rect);
        try (OnnxTensor t = OnnxTensor.createTensor(env, FloatBuffer.wrap(lb.chw()),
                new long[] {1, 3, lb.inputH(), lb.inputW()});
                OrtSession.Result r = session.run(Map.of(inputName, t))) {
            float[][][] out = (float[][][]) r.get(0).getValue();
            return YoloOutput.parse(out[0], lb, minConfidence, nmsIou);
        } catch (OrtException e) {
            throw new IllegalStateException("model run failed: " + e.getMessage(), e);
        }
    }

    @Override
    public void close() throws OrtException {
        if (session != null) {
            session.close();
        }
    }
}
