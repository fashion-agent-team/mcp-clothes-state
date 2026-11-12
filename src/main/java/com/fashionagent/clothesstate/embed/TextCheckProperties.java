package com.fashionagent.clothesstate.embed;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.context.properties.bind.DefaultValue;

/**
 * clothes.text-check.*: 옷 박스 이미지 벡터를 옷 이름 글 벡터와 견줘 종류를 다시 보고, 옷이 아닌 것(빈 벽·바닥)을 뺍니다.
 * 옷 벡터 모델(clothes.embed.model)이 있을 때만 씁니다. 기준값 근거는 docs/training-log.md "글 벡터로 종류 다시 보기".
 */
@ConfigurationProperties("clothes.text-check")
public record TextCheckProperties(
        /** 미리 계산한 FashionSigLIP 글 벡터(이름별 문장 35개). 비우면 끔. */
        @DefaultValue("classpath:models/garment-text-fsiglip.json") String vectors,
        /** "빈 벽·바닥" 점수가 가장 높은 옷 이름 점수보다 이만큼 넘으면 옷이 아니라고 봄. 실제 옷 1,282벌 최대 0.024, 빈 벽 60장 최소 0.044. */
        @DefaultValue("0.03") double noneMargin,
        /** 내려놓은 사진: 옷 종류 모델 확신도가 이보다 낮으면 글 벡터로 종류를 다시 고름. */
        @DefaultValue("0.7") double keepLaid,
        /** 입은 사진: 옷 종류 모델 확신도가 이보다 낮으면 글 벡터로 종류를 다시 고름(입은 사진은 모델이 약함). */
        @DefaultValue("0.9") double keepWorn) {
}
