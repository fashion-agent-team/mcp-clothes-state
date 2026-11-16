package com.fashionagent.clothesstate.wear;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.context.properties.bind.DefaultValue;

/**
 * clothes.wear-learn.*: 옷 실수를 GPT에게 배우며 점점 덜 묻기(정지원 2026-11-16 "지피티로 단추 엇갈린거 학습하면서 다음에는 서서히 안물어볼 수
 * 있게"). 사용자가 GPT 확인에 동의했을 때만 동작합니다.
 * <ol>
 * <li>GPT에게 물을 때마다 실수 종류별로 우리 모델 확률과 GPT 답을 기록(WearTrust, 최근 window개).</li>
 * <li>GPT가 실수라고 한 것, 모델과 GPT가 엇갈린 것, 모델이 애매한 것은 GPT에게 보낸 옷 조각(얼굴 쪽 회색)을 학습 후보로 저장
 * (CandidateStore, wear-종류-아이디.jpg, 30일). 맞힌 정상은 saveAgreeRate만큼만.</li>
 * <li>최근 기록에서 모델이 GPT와 충분히 맞으면(정상·실수 둘 다) 그 실수는 "믿음". 옷 한 벌에 볼 수 있는 실수가 모두 믿을 만하고 모델이
 * 확신하면(sureLow 이하 또는 sureHigh 이상) GPT에게 묻지 않음. 그래도 auditRate만큼은 계속 물어 모델이 틀려지는지 봄.</li>
 * </ol>
 * 쌓인 후보로 모델을 다시 학습하는 건 training/wear/build_from_candidates.py(사람이 확인 후).
 */
@ConfigurationProperties("clothes.wear-learn")
public record WearLearnProperties(
        /** false면 기록·저장·건너뛰기 모두 안 함(지금처럼 동의하면 늘 GPT에게 물음). */
        @DefaultValue("true") boolean enabled,
        /** 종류마다 기억하는 최근 GPT 답 수. */
        @DefaultValue("200") int window,
        /** 믿기 전에 필요한 최소 GPT 답 수. */
        @DefaultValue("50") int minSamples,
        /** 믿기 전에 필요한 GPT가 "실수"라고 한 최소 수(정상만 많이 맞힌 걸로 믿지 않게). */
        @DefaultValue("10") int minPositives,
        /** GPT가 정상이라고 한 것 중 모델도 정상(확률 0.5 미만)이라고 한 비율 최소. */
        @DefaultValue("0.95") double minSpecificity,
        /** GPT가 실수라고 한 것 중 모델도 실수(0.5 이상)라고 한 비율 최소. */
        @DefaultValue("0.9") double minRecall,
        /** 모델 확률이 이 이하면 "확신하는 정상". */
        @DefaultValue("0.1") double sureLow,
        /** 모델 확률이 이 이상이면 "확신하는 실수". */
        @DefaultValue("0.9") double sureHigh,
        /** 믿게 된 뒤에도 GPT에게 계속 묻는 비율(모델이 틀려지는지 감시). */
        @DefaultValue("0.05") double auditRate,
        /** 모델·GPT 모두 정상이라고 한 조각 중 저장하는 비율(정상 학습 데이터). */
        @DefaultValue("0.1") double saveAgreeRate,
        /** 기록 파일. 비우면 학습 후보 폴더(clothes.gpt.candidate-dir)/wear-trust.json. */
        @DefaultValue("") String stateFile) {
}
