package com.fashionagent.clothesstate.wear;

import java.time.Duration;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.context.properties.bind.DefaultValue;

/**
 * clothes.wear-check.*: 입은 모습(worn) 사진에서 옷 실수(뒤집어 입음·앞뒤 바뀜·단추 어긋남·택 튀어나옴)를 GPT에게 묻는 설정
 * (정지원 2026-11-13 "실제로 자주 겪는 실수"). 키는 clothes.gpt.api-key(OPENAI_API_KEY)를 같이 씁니다.
 * 사용자가 GPT 확인에 동의(allowGpt=true)했을 때만 묻고, 얼굴이 들어갈 수 있는 사진 위쪽은 회색으로 가린 옷 박스만 보냅니다.
 */
@ConfigurationProperties("clothes.wear-check")
public record WearCheckProperties(
        /** false면 옷 실수 확인을 하지 않음. */
        @DefaultValue("true") boolean enabled,
        /** 비우면 clothes.gpt.model(gpt-4.1-mini). */
        @DefaultValue("") String model,
        /**
         * GPT 한 번에 기다리는 최대 시간. 2026-11-14: 사진 한 장 응답을 줄이려고(얼룩 1.5초 결정) 6초 → 3초(무늬 확인과 같음).
         * 우리 모델 결과는 GPT를 기다리지 않고 늘 나옴.
         */
        @DefaultValue("3s") Duration timeout,
        /** 보내는 옷 조각 긴 변 최대 픽셀(단추·택이 보이게 얼룩 조각 512보다 크게). */
        @DefaultValue("768") int maxSide,
        /** GPT 확신도가 이 이상일 때만 실수로 말함. 틀리게 말하면 다시 갈아입게 되니 높게. */
        @DefaultValue("0.7") double acceptConfidence,
        /** 앞모습: 사진 높이의 이 비율보다 위는 얼굴이 들어갈 수 있어 회색으로 가림(얼룩 확인 wornSkipTop과 같음). */
        @DefaultValue("0.25") double frontHideTop,
        /** 뒷모습: 뒤통수만 가림. 목 뒤 택은 보여야 해서 앞모습보다 낮게. */
        @DefaultValue("0.12") double backHideTop,
        /** 한 사진에서 확인할 옷 수(신발 빼고, 위에서부터). */
        @DefaultValue("3") int maxGarments) {
}
