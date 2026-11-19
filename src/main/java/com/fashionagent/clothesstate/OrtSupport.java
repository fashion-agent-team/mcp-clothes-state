package com.fashionagent.clothesstate;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtLoggingLevel;
import ai.onnxruntime.OrtSession;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 이 서버의 ONNX Runtime 세션(YOLO들, 색, 옷 실수, 무늬 YOLO·LoRA, FashionSigLIP 임베딩)이 같은 환경과 같은 메모리 설정을
 * 쓰게 합니다(2026-11-17, 젯슨 clothes-state 2GB 상한에 닿음). 환경 변수로 바꿉니다.
 * <ul>
 * <li>CLOTHES_ORT_ARENA (기본 false): CPU 메모리 arena. 끄면 한 번 커진 버퍼를 붙잡아 두지 않음</li>
 * <li>CLOTHES_ORT_MEM_PATTERN (기본 false): 입력 크기별 메모리 미리 잡기</li>
 * <li>CLOTHES_ORT_PREPACK (기본 true): 행렬 곱 가중치 미리 펴기. 끄면 조금 줄지만 SigLIP이 약 10% 느려짐</li>
 * <li>CLOTHES_ORT_THREADS (기본 0 = 코어 수): 세션 하나가 쓰는 계산 스레드 수</li>
 * <li>CLOTHES_ORT_SHARED_THREADS (기본 false): true면 모든 세션이 스레드 풀 하나를 같이 씀(세션마다 풀을 만들지 않음)</li>
 * </ul>
 */
public final class OrtSupport {

    private static final Logger log = LoggerFactory.getLogger(OrtSupport.class);
    private static volatile OrtEnvironment env;

    private OrtSupport() {
    }

    record Settings(boolean arena, boolean memPattern, boolean prepack, int threads, boolean sharedThreads) {

        static Settings from(Map<String, String> e) {
            return new Settings(bool(e, "CLOTHES_ORT_ARENA", false), bool(e, "CLOTHES_ORT_MEM_PATTERN", false),
                    bool(e, "CLOTHES_ORT_PREPACK", true), num(e, "CLOTHES_ORT_THREADS", 0),
                    bool(e, "CLOTHES_ORT_SHARED_THREADS", false));
        }

        private static boolean bool(Map<String, String> e, String k, boolean d) {
            String v = e.get(k);
            return v == null || v.isBlank() ? d : Boolean.parseBoolean(v.strip());
        }

        private static int num(Map<String, String> e, String k, int d) {
            try {
                String v = e.get(k);
                return v == null || v.isBlank() ? d : Math.max(0, Integer.parseInt(v.strip()));
            } catch (NumberFormatException ex) {
                return d;
            }
        }
    }

    static final Settings SETTINGS = Settings.from(System.getenv());

    /** 모든 세션이 같이 쓰는 환경. 처음 부를 때 만들고, 스레드 풀을 같이 쓰면 여기서 정합니다. */
    public static OrtEnvironment env() {
        OrtEnvironment e = env;
        if (e != null) {
            return e;
        }
        synchronized (OrtSupport.class) {
            if (env == null) {
                env = create();
                log.info("ONNX Runtime 설정: arena={} memPattern={} prepack={} threads={} sharedThreads={}",
                        SETTINGS.arena(), SETTINGS.memPattern(), SETTINGS.prepack(), SETTINGS.threads(),
                        SETTINGS.sharedThreads());
            }
            return env;
        }
    }

    private static OrtEnvironment create() {
        if (!SETTINGS.sharedThreads()) {
            return OrtEnvironment.getEnvironment();
        }
        try (OrtEnvironment.ThreadingOptions t = new OrtEnvironment.ThreadingOptions()) {
            if (SETTINGS.threads() > 0) {
                t.setGlobalIntraOpNumThreads(SETTINGS.threads());
            }
            t.setGlobalInterOpNumThreads(1);
            return OrtEnvironment.getEnvironment(OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING, "clothes-state", t);
        } catch (OrtException | IllegalStateException ex) {
            log.warn("공유 스레드 풀을 만들지 못해 세션마다 풀을 씁니다: {}", ex.toString());
            return OrtEnvironment.getEnvironment();
        }
    }

    /** 위 설정을 넣은 세션 옵션. 부르는 쪽이 세션을 만든 뒤 닫습니다. */
    public static OrtSession.SessionOptions options() throws OrtException {
        OrtSession.SessionOptions o = new OrtSession.SessionOptions();
        o.setCPUArenaAllocator(SETTINGS.arena());
        o.setMemoryPatternOptimization(SETTINGS.memPattern());
        if (!SETTINGS.prepack()) {
            o.addConfigEntry("session.disable_prepacking", "1");
        }
        if (SETTINGS.sharedThreads()) {
            o.disablePerSessionThreads();
        } else if (SETTINGS.threads() > 0) {
            o.setIntraOpNumThreads(SETTINGS.threads());
        }
        o.setInterOpNumThreads(1);
        return o;
    }
}
