package com.fashionagent.clothesstate.embed;

import java.time.Duration;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.context.properties.bind.DefaultValue;

/**
 * clothes.so400m.*: 옷 종류가 애매한 옷만 더 큰 이미지 모델(SigLIP2 So400m, 다른 프로세스의 벡터 서버)로 다시 고르는 보조 경로.
 * url이 비면 꺼짐(기본). 근거: training/probe/README.md 12차(2026-11-17) — 폰 사진 1,841장에서 옷 15%만 다시 물어
 * 세부 81.4→83.1%, Fashionpedia val 92.6→93.0%.
 */
@ConfigurationProperties("clothes.so400m")
public record So400mAssistProperties(
        /** 벡터 서버 주소(예: http://so400m-embed:8096). 비우면 끔. */
        @DefaultValue("") String url,
        /** So400m 벡터로 학습한 종류 벡터 파일(형식은 text-check.vectors와 같음, 1152차원). */
        @DefaultValue("classpath:models/garment-kind-so400m-probe.json") String probe,
        /** 옷 한 장(여러 벌이면 모두)에 기다리는 최대 시간. 넘기거나 오류면 지금 결과(v4) 그대로. */
        @DefaultValue("800ms") Duration timeout,
        /** 옷 종류 모델 확신도가 이보다 낮을 때만 다시 물음. */
        @DefaultValue("0.7") double maxConfidence,
        /** 지금 종류 벡터(v4)의 1·2등 점수 차이가 이보다 작을 때만 다시 물음. */
        @DefaultValue("0.015") double maxGap,
        /** 사진 한 장에서 다시 묻는 옷 수 상한. */
        @DefaultValue("3") int maxBoxes,
        /** 박스를 사방으로 넓히는 비율(서버 벡터와 같음). */
        @DefaultValue("0.03") double margin,
        /** true면 모든 옷의 색 계열을 So400m 벡터 + 색 분류기로 정함(기본 끔). 실패·시간 초과면 지금 색 그대로. */
        @DefaultValue("false") boolean color,
        /** So400m 색 분류기(선형, 14계열). */
        @DefaultValue("classpath:models/garment-color-so400m.json") String colorModel) {
}
