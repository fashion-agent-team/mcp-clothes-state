package com.fashionagent.clothesstate.embed;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;
import java.awt.image.BufferedImage;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Marqo FashionSigLIP vision(ONNX, Apache-2.0)을 이 서버에서 돌려 옷마다 768차원 벡터를 만듭니다(이슈 #15). 사진은 외부로
 * 보내지 않고 저장하지 않습니다. 원본과 좌우 뒤집은 사진 벡터를 각각 길이 1로 맞춰 평균한 뒤 다시 길이 1로 맞춥니다.
 * 모델 파일이 없으면 {@link #ready()}가 false이고 embedding은 null입니다.
 */
public class OnnxGarmentEmbedder implements GarmentEmbedder, AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(OnnxGarmentEmbedder.class);

    private final EmbedProperties props;
    private final OrtEnvironment env;
    private final OrtSession session;
    private final String inputName;
    /** 입력 batch가 정해져 있지 않으면 원본·뒤집기를 한 번에 넣음. */
    private final boolean batch2;

    private final GarmentTextCheck textCheck;

    public OnnxGarmentEmbedder(EmbedProperties props) {
        this(props, null);
    }

    public OnnxGarmentEmbedder(EmbedProperties props, GarmentTextCheck textCheck) {
        this.props = props;
        this.textCheck = textCheck;
        OrtEnvironment e = null;
        OrtSession s = null;
        String in = null;
        boolean dyn = false;
        String path = props.model() == null ? "" : props.model().strip();
        if (!path.isEmpty() && Files.isRegularFile(Path.of(path))) {
            try {
                e = OrtEnvironment.getEnvironment();
                // 372MB라 byte[]로 읽지 않고 경로로 엶
                s = e.createSession(path, new OrtSession.SessionOptions());
                in = s.getInputNames().iterator().next();
                long[] shape = ((TensorInfo) s.getInputInfo().get(in).getInfo()).getShape();
                dyn = shape.length == 4 && shape[0] < 0;
                log.info("옷 벡터 모델을 읽었어요: {} (입력 {} {}, batch {})", path, in, java.util.Arrays.toString(shape),
                        dyn ? "2장 한 번에" : "1장씩");
            } catch (OrtException ex) {
                log.warn("옷 벡터 모델을 읽지 못했어요: {}", ex.toString());
                s = null;
            }
        } else if (!path.isEmpty()) {
            log.warn("옷 벡터 모델 파일이 없어요({}). embedding 없이 응답해요.", path);
        }
        this.env = e;
        this.session = s;
        this.inputName = in;
        this.batch2 = dyn;
    }

    @Override
    public boolean ready() {
        return session != null;
    }

    @Override
    public GarmentTextCheck textCheck() {
        return textCheck != null && textCheck.ready() ? textCheck : null;
    }

    @Override
    public String version() {
        return props.version();
    }

    @Override
    public float[] embed(BufferedImage image, double x, double y, double w, double h) {
        if (session == null) {
            throw new IllegalStateException("embed model not loaded");
        }
        int size = props.inputSize();
        EmbedInput.Pair in = EmbedInput.prepare(image, x, y, w, h, props.margin(), size, props.mean(), props.std());
        try {
            float[][] vecs;
            if (!props.flipAverage()) {
                vecs = run(in.chw(), 1, size);
            } else if (batch2) {
                float[] both = new float[in.chw().length * 2];
                System.arraycopy(in.chw(), 0, both, 0, in.chw().length);
                System.arraycopy(in.flipped(), 0, both, in.chw().length, in.flipped().length);
                vecs = run(both, 2, size);
            } else {
                vecs = new float[][] {run(in.chw(), 1, size)[0], run(in.flipped(), 1, size)[0]};
            }
            return average(vecs, props.decimals());
        } catch (OrtException e) {
            throw new IllegalStateException("embed model run failed: " + e.getMessage(), e);
        }
    }

    private float[][] run(float[] chw, int n, int size) throws OrtException {
        try (OnnxTensor t = OnnxTensor.createTensor(env, FloatBuffer.wrap(chw), new long[] {n, 3, size, size});
                OrtSession.Result r = session.run(Map.of(inputName, t))) {
            Object v = r.get(0).getValue();
            if (v instanceof float[][] m) {
                return m;
            }
            throw new IllegalStateException("embed output shape not [N, D]: " + v.getClass().getSimpleName());
        }
    }

    /** 각각 길이 1로 → 평균 → 다시 길이 1로, 소수점 decimals자리. */
    static float[] average(float[][] vecs, int decimals) {
        int d = vecs[0].length;
        double[] sum = new double[d];
        for (float[] v : vecs) {
            double norm = norm(v);
            for (int i = 0; i < d; i++) {
                sum[i] += v[i] / norm;
            }
        }
        double n = 0;
        for (double s : sum) {
            n += s * s;
        }
        n = Math.sqrt(n);
        double scale = Math.pow(10, decimals);
        float[] out = new float[d];
        for (int i = 0; i < d; i++) {
            out[i] = (float) (Math.round(sum[i] / (n == 0 ? 1 : n) * scale) / scale);
        }
        return out;
    }

    private static double norm(float[] v) {
        double s = 0;
        for (float f : v) {
            s += (double) f * f;
        }
        return s == 0 ? 1 : Math.sqrt(s);
    }

    @Override
    public void close() throws OrtException {
        if (session != null) {
            session.close();
        }
    }
}
