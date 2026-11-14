package com.fashionagent.clothesstate.wear;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import com.fashionagent.clothesstate.pattern.PatternInput;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.Resource;

/**
 * 두 클래스 분류 모델(Ultralytics YOLO26n-cls ONNX, 출력 [1, 2] 확률) 하나. 입력은 무늬 모델과 같은 방식
 * (PatternInput: 자르고 짧은 변을 size로, 가운데 정사각형, x/255). 파일이 없으면 ready()=false.
 */
public class OnnxBinaryClassifier implements AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(OnnxBinaryClassifier.class);

    private final OrtEnvironment env;
    private final OrtSession session;
    private final String inputName;
    private final int size;
    /** 출력에서 "실수 있음" 클래스의 자리(학습 폴더 이름 알파벳 순서). */
    private final int positive;

    public OnnxBinaryClassifier(Resource model, int size, int positive, String what) {
        this.size = size;
        this.positive = positive;
        OrtEnvironment e = null;
        OrtSession s = null;
        String in = null;
        if (model != null && model.exists()) {
            try (InputStream is = model.getInputStream()) {
                e = OrtEnvironment.getEnvironment();
                s = e.createSession(is.readAllBytes(), new OrtSession.SessionOptions());
                in = s.getInputNames().iterator().next();
                log.info("옷 실수 모델({})을 읽었어요: {}", what, model.getDescription());
            } catch (IOException | OrtException ex) {
                log.warn("옷 실수 모델({})을 읽지 못했어요: {}", what, ex.toString());
                s = null;
            }
        } else {
            log.info("옷 실수 모델({}) 파일이 없어요. 건너뛰어요.", what);
        }
        this.env = e;
        this.session = s;
        this.inputName = in;
    }

    public boolean ready() {
        return session != null;
    }

    /** 박스(사진 비율)를 margin만큼 넓혀 잘라 "실수 있음" 확률. */
    public double probability(BufferedImage img, double x, double y, double w, double h, double margin) {
        float[] chw = PatternInput.prepare(img, x, y, w, h, margin, size);
        try (OnnxTensor t = OnnxTensor.createTensor(env, FloatBuffer.wrap(chw), new long[] {1, 3, size, size});
                OrtSession.Result r = session.run(Map.of(inputName, t))) {
            Object v = r.get(0).getValue();
            if (!(v instanceof float[][] m) || m[0].length != 2) {
                throw new IllegalStateException("wear model output shape not [1, 2]");
            }
            return m[0][positive];
        } catch (OrtException ex) {
            throw new IllegalStateException("wear model run failed: " + ex.getMessage(), ex);
        }
    }

    /**
     * 박스를 margin만큼 넓혀 잘라 긴 변 기준 정사각형 회색(114) 바탕 가운데에 놓고 "실수 있음" 확률. 가운데 자르기로
     * 밑단·깃이 잘리지 않게(단추·뒤집힘 v2 학습과 같음).
     */
    public double probabilitySquare(BufferedImage img, double x, double y, double w, double h, double margin) {
        return probability(square(img, x, y, w, h, margin), 0, 0, 1, 1, 0);
    }

    /** build_button_dataset.py crop()과 같은 자르기(int 내림)와 정사각형 채우기. */
    static BufferedImage square(BufferedImage img, double x, double y, double w, double h, double margin) {
        int iw = img.getWidth();
        int ih = img.getHeight();
        double bx = x * iw, by = y * ih, bw = w * iw, bh = h * ih;
        int x0 = Math.max(0, (int) (bx - bw * margin));
        int y0 = Math.max(0, (int) (by - bh * margin));
        int x1 = Math.min(iw, (int) (bx + bw * (1 + margin)));
        int y1 = Math.min(ih, (int) (by + bh * (1 + margin)));
        int cw = Math.max(1, x1 - x0);
        int ch = Math.max(1, y1 - y0);
        int side = Math.max(cw, ch);
        BufferedImage sq = new BufferedImage(side, side, BufferedImage.TYPE_INT_RGB);
        java.awt.Graphics2D g = sq.createGraphics();
        g.setColor(new java.awt.Color(114, 114, 114));
        g.fillRect(0, 0, side, side);
        g.drawImage(img.getSubimage(x0, y0, Math.min(cw, iw - x0), Math.min(ch, ih - y0)), (side - cw) / 2,
                (side - ch) / 2, null);
        g.dispose();
        return sq;
    }

    @Override
    public void close() throws OrtException {
        if (session != null) {
            session.close();
        }
    }
}
