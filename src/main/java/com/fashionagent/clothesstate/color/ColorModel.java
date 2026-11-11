package com.fashionagent.clothesstate.color;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.Resource;

/**
 * 옷 조각 색 인식 모델(ONNX, 이 서버 안에서 돌림). AI Hub K-Fashion 옷 17만 벌의 색 라벨 20가지로 학습한 작은 CNN입니다
 * (training/color-model). 옷 박스 조각을 {@value #SIZE}×{@value #SIZE}로 줄여 넣고 20가지 색의 확률을 받습니다.
 * 모델 파일이 없으면 {@link #ready()}가 false이고, 그때는 예전처럼 픽셀 규칙(GarmentColor)만 씁니다.
 */
public class ColorModel implements AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(ColorModel.class);

    /** 모델 입력 한 변(학습과 같게). */
    public static final int SIZE = 64;

    /** 학습 클래스 순서(K-Fashion 색상 라벨). 네온은 사진이 적어 뺐습니다. */
    public static final String[] CLASSES = {"화이트", "블랙", "베이지", "핑크", "그레이", "스카이블루", "네이비", "브라운", "블루",
        "카키", "레드", "그린", "라벤더", "실버", "옐로우", "와인", "민트", "퍼플", "오렌지", "골드"};

    private final OrtEnvironment env;
    private final OrtSession session;
    private final String inputName;

    public ColorModel(Resource model) {
        OrtEnvironment e = null;
        OrtSession s = null;
        String in = null;
        if (model != null && model.exists()) {
            try (InputStream is = model.getInputStream()) {
                byte[] bytes = is.readAllBytes();
                e = OrtEnvironment.getEnvironment();
                s = e.createSession(bytes, new OrtSession.SessionOptions());
                in = s.getInputNames().iterator().next();
                log.info("색 인식 모델을 읽었어요: {} ({} bytes)", model.getDescription(), bytes.length);
            } catch (IOException | OrtException ex) {
                log.warn("색 인식 모델을 읽지 못했어요: {}", ex.toString());
                s = null;
            }
        } else {
            log.warn("색 인식 모델 파일이 없어요({}). 픽셀 규칙으로 색을 정해요.", model == null ? "-" : model.getDescription());
        }
        this.env = e;
        this.session = s;
        this.inputName = in;
    }

    public boolean ready() {
        return session != null;
    }

    /** 옷 조각 → 20가지 색 확률(CLASSES 순서, 합 1). */
    public float[] probabilities(BufferedImage crop) {
        if (session == null) {
            throw new IllegalStateException("color model not loaded");
        }
        try (OnnxTensor t = OnnxTensor.createTensor(env, FloatBuffer.wrap(input(crop)), new long[] {1, 3, SIZE, SIZE});
                OrtSession.Result r = session.run(java.util.Map.of(inputName, t))) {
            return ((float[][]) r.get(0).getValue())[0];
        } catch (OrtException ex) {
            throw new IllegalStateException("color model failed", ex);
        }
    }

    /**
     * 조각을 {@value #SIZE}×{@value #SIZE}로 늘리거나 줄여(가로세로 비율 무시, 면적 평균) RGB 0~1, CHW 순서로.
     * 학습도 조각을 같은 크기로 늘려(BOX) 넣었습니다.
     */
    static float[] input(BufferedImage crop) {
        Image scaled = crop.getScaledInstance(SIZE, SIZE, Image.SCALE_AREA_AVERAGING);
        BufferedImage img = new BufferedImage(SIZE, SIZE, BufferedImage.TYPE_INT_RGB);
        img.getGraphics().drawImage(scaled, 0, 0, null);
        float[] chw = new float[3 * SIZE * SIZE];
        int plane = SIZE * SIZE;
        for (int y = 0; y < SIZE; y++) {
            for (int x = 0; x < SIZE; x++) {
                int p = img.getRGB(x, y);
                int i = y * SIZE + x;
                chw[i] = ((p >> 16) & 0xFF) / 255f;
                chw[plane + i] = ((p >> 8) & 0xFF) / 255f;
                chw[2 * plane + i] = (p & 0xFF) / 255f;
            }
        }
        return chw;
    }

    @Override
    public void close() throws OrtException {
        if (session != null) {
            session.close();
        }
    }
}
