package com.fashionagent.clothesstate.verify;

import java.time.Duration;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.context.properties.bind.DefaultValue;

/**
 * clothes.gpt.*: 애매한 얼룩(확신도 ask~tell)을 지인 대신 GPT에게 묻는 설정(정지원 2026-11-09).
 * 키는 환경 변수 OPENAI_API_KEY로만 받습니다. 비어 있으면 GPT에게 묻지 않고 예전처럼 pendingHelper로 돌려줍니다.
 */
@ConfigurationProperties("clothes.gpt")
public record GptCheckProperties(
        /** OpenAI API 키. application.yml에서 ${OPENAI_API_KEY:}로 받음. 로그·응답에 절대 남기지 않음. */
        @DefaultValue("") String apiKey,
        /** 11-09 공개 시험 사진 비교에서 가장 좋았던 모델(사진 단위 정답률 98.1%). */
        @DefaultValue("gpt-4.1-mini") String model,
        @DefaultValue("https://api.openai.com/v1") String baseUrl,
        /** GPT 한 번에 기다리는 최대 시간. 넘기면 확인하지 못한 것으로 보고 pendingHelper로 돌려줌(agent는 MCP를 20초 기다림). */
        @DefaultValue("3s") Duration timeout,
        /** 결함 박스 둘레에 붙일 여유(박스 크기 대비). */
        @DefaultValue("0.3") double cropMargin,
        /** 보내는 조각의 긴 변 최대 픽셀. */
        @DefaultValue("512") int cropMaxSide,
        /** 넓힌 뒤에도 이보다 작으면 가운데 기준으로 이 크기까지 늘림(픽셀). */
        @DefaultValue("64") int cropMinSide,
        /** GPT 확신도가 이 이상일 때만 결함으로 봄. 시험에서는 문턱 없이(0) label만 봤습니다. */
        @DefaultValue("0") double acceptConfidence,
        /** GPT를 쓸 수 없을 때(키 없음·동의 안 함·답 없음) 확실하지 않은 후보로 돌려줄 최소 확신도. 예전 기준 0.3. */
        @DefaultValue("0.3") double fallbackAskThreshold,
        /** 입은 사진에서 이보다 위(사진 높이 비율)에 걸친 조각은 얼굴이 들어갈 수 있어 보내지 않음. */
        @DefaultValue("0.25") double wornSkipTop,
        /**
         * GPT가 얼룩·찢어짐이라고 한 조각만 저장할 폴더(다시 학습할 후보). 비우면 저장하지 않음.
         * 옷 전체 사진이나 깨끗하다고 한 조각은 저장하지 않습니다.
         */
        @DefaultValue("data/train-candidates") String candidateDir,
        /** 학습 후보를 이 날수만큼만 보관하고 지난 날짜 폴더는 지움. */
        @DefaultValue("30") int retentionDays) {

    public boolean enabled() {
        return apiKey != null && !apiKey.isBlank();
    }
}
