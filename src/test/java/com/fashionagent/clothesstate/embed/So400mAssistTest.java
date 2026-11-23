package com.fashionagent.clothesstate.embed;

import static org.assertj.core.api.Assertions.assertThat;

import com.sun.net.httpserver.HttpServer;
import java.awt.image.BufferedImage;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.core.io.ByteArrayResource;

class So400mAssistTest {

    /** 3차원 시험 probe: shirt=(1,0,0), dress=(0,1,0), shoes·none은 고르면 안 됨. */
    static final String PROBE = """
            {"prompts":[{"name":"shirt","vector":[1,0,0]},{"name":"dress","vector":[0,1,0]},
             {"name":"shoes","vector":[0,0,1]},{"name":"none","vector":[0,0,2]}]}""";

    HttpServer server;
    final AtomicInteger calls = new AtomicInteger();

    private String serve(String body, long delayMs) throws Exception {
        server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/embed", ex -> {
            calls.incrementAndGet();
            ex.getRequestBody().readAllBytes();
            try {
                Thread.sleep(delayMs);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            byte[] b = body.getBytes(StandardCharsets.UTF_8);
            ex.getResponseHeaders().add("content-type", "application/json");
            ex.sendResponseHeaders(200, b.length);
            ex.getResponseBody().write(b);
            ex.close();
        });
        server.start();
        return "http://127.0.0.1:" + server.getAddress().getPort();
    }

    @AfterEach
    void stop() {
        if (server != null) {
            server.stop(0);
        }
    }

    private static So400mAssist assist(String url, long timeoutMs) {
        var p = new So400mAssistProperties(url, "x", Duration.ofMillis(timeoutMs), 0.7, 0.015, 3, 0.03, false, "", "");
        return new So400mAssist(p, new ByteArrayResource(PROBE.getBytes()));
    }

    private static final BufferedImage IMG = new BufferedImage(40, 60, BufferedImage.TYPE_INT_RGB);

    @Test
    @DisplayName("주소가 비면 꺼져 있고 아무것도 묻지 않는다")
    void offWithoutUrl() {
        var a = assist("", 800);
        assertThat(a.enabled()).isFalse();
        assertThat(a.wanted(0.3, Map.of("shirt", 0.5, "dress", 0.5))).isFalse();
        assertThat(a.pick(IMG, List.<double[]>of(new double[] {0.1, 0.1, 0.5, 0.5}))).containsExactly((String) null);
        assertThat(So400mAssist.OFF.enabled()).isFalse();
    }

    @Test
    @DisplayName("확신도 0.7 미만이고 1·2등 차이가 0.015 미만일 때만 다시 묻는다(none·신발 점수는 빼고)")
    void wantedOnlyWhenUnsure() throws Exception {
        var a = assist(serve("{\"vector\":[0,1,0]}", 0), 800);
        assertThat(a.wanted(0.5, Map.of("shirt", 0.30, "dress", 0.29, "none", 0.9, "shoes", 0.31))).isTrue();
        assertThat(a.wanted(0.75, Map.of("shirt", 0.30, "dress", 0.29))).isFalse();
        assertThat(a.wanted(0.5, Map.of("shirt", 0.30, "dress", 0.25))).isFalse();
        assertThat(a.wanted(null, Map.of("shirt", 0.30, "dress", 0.29))).isFalse();
        assertThat(So400mAssist.gap(Map.of("shirt", 0.3))).isInfinite();
    }

    @Test
    @DisplayName("벡터 서버가 준 벡터로 probe 1등 종류를 고른다(신발·none은 고르지 않음)")
    void picksFromServerVector() throws Exception {
        var a = assist(serve("{\"vector\":[0.1,0.9,5]}", 0), 800);
        var picks = a.pick(IMG, List.of(new double[] {0.1, 0.1, 0.5, 0.5}, new double[] {0.2, 0.3, 0.6, 0.6}));
        assertThat(picks).containsExactly("dress", "dress");
        assertThat(calls.get()).isEqualTo(2);
    }

    @Test
    @DisplayName("서버가 기다리는 시간보다 늦으면 null(지금 결과를 씀)")
    void timeoutFallsBack() throws Exception {
        var a = assist(serve("{\"vector\":[0,1,0]}", 600), 150);
        long t = System.nanoTime();
        assertThat(a.pick(IMG, List.<double[]>of(new double[] {0.1, 0.1, 0.5, 0.5}))).containsExactly((String) null);
        assertThat((System.nanoTime() - t) / 1_000_000).isLessThan(500);
    }

    @Test
    @DisplayName("벡터 길이가 다르거나 서버가 없으면 null")
    void badAnswerFallsBack() throws Exception {
        var a = assist(serve("{\"vector\":[1,0]}", 0), 800);
        assertThat(a.pick(IMG, List.<double[]>of(new double[] {0.1, 0.1, 0.5, 0.5}))).containsExactly((String) null);
        var b = assist("http://127.0.0.1:9", 300);
        assertThat(b.pick(IMG, List.<double[]>of(new double[] {0.1, 0.1, 0.5, 0.5}))).containsExactly((String) null);
    }

    @Test
    @DisplayName("박스 조각은 측정 때와 같게 224×224 PNG(무손실)로 보낸다")
    void sendsLosslessPng() throws Exception {
        var seen = new java.util.concurrent.atomic.AtomicReference<String>();
        server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/embed", ex -> {
            seen.set(new String(ex.getRequestBody().readAllBytes(), StandardCharsets.UTF_8));
            byte[] b = "{\"vector\":[0,1,0]}".getBytes(StandardCharsets.UTF_8);
            ex.sendResponseHeaders(200, b.length);
            ex.getResponseBody().write(b);
            ex.close();
        });
        server.start();
        var a = assist("http://127.0.0.1:" + server.getAddress().getPort(), 800);
        assertThat(a.pick(IMG, List.<double[]>of(new double[] {0.1, 0.1, 0.5, 0.5}))).containsExactly("dress");
        String b64 = seen.get().replaceAll(".*\"imageBase64\"\\s*:\\s*\"([^\"]+)\".*", "$1");
        byte[] png = java.util.Base64.getDecoder().decode(b64);
        assertThat(png[1]).isEqualTo((byte) 'P');
        assertThat(png[2]).isEqualTo((byte) 'N');
        var im = javax.imageio.ImageIO.read(new java.io.ByteArrayInputStream(png));
        assertThat(im.getWidth()).isEqualTo(224);
        assertThat(im.getHeight()).isEqualTo(224);
    }

    static final String COLOR = """
            {"families":["red","blue"],"weights":[[1,0,0],[0,1,0]],"bias":[0,0]}""";

    private static So400mAssist colorAssist(String url, boolean on) {
        var p = new So400mAssistProperties(url, "x", Duration.ofMillis(800), 0.7, 0.015, 3, 0.03, on, "y", "");
        return new So400mAssist(p, new ByteArrayResource(PROBE.getBytes()), new ByteArrayResource(COLOR.getBytes()));
    }

    @Test
    @DisplayName("색 분류기는 설정(color=true)과 주소가 있을 때만 켜지고, 벡터로 색 계열을 고른다")
    void colorClassifier() {
        assertThat(colorAssist("http://127.0.0.1:9", false).colorEnabled()).isFalse();
        assertThat(colorAssist("", true).colorEnabled()).isFalse();
        var a = colorAssist("http://127.0.0.1:9", true);
        assertThat(a.colorEnabled()).isTrue();
        assertThat(a.colorFamily(new float[] {0.1f, 0.9f, 0f})).isEqualTo("blue");
        assertThat(a.colorFamily(new float[] {0.9f, 0.1f, 0f})).isEqualTo("red");
        assertThat(a.colorFamily(new float[] {1f, 0f})).isNull();
        assertThat(a.colorFamily(null)).isNull();
    }

    @Test
    @DisplayName("색 분류기 파일 모양이 맞지 않으면 색 분류기를 끈다")
    void badColorModelOff() {
        var p = new So400mAssistProperties("http://127.0.0.1:9", "x", Duration.ofMillis(800), 0.7, 0.015, 3, 0.03, true, "y", "");
        var a = new So400mAssist(p, new ByteArrayResource(PROBE.getBytes()),
                new ByteArrayResource("{\"families\":[\"red\"],\"weights\":[[1,0,0],[0,1,0]],\"bias\":[0]}".getBytes()));
        assertThat(a.colorEnabled()).isFalse();
    }

    @Test
    @DisplayName("바꿀 목록(color-override): 비우면 늘, 있으면 목록 안의 계열만")
    void overrideList() {
        var all = new So400mAssist(new So400mAssistProperties("http://x", "x", Duration.ofMillis(800), 0.7, 0.015, 3, 0.03, true, "y", ""), null, null);
        assertThat(all.overrides("orange")).isTrue();
        var neutral = new So400mAssist(new So400mAssistProperties("http://x", "x", Duration.ofMillis(800), 0.7, 0.015, 3, 0.03, true, "y", "black, gray,navy"), null, null);
        assertThat(neutral.overrides("gray")).isTrue();
        assertThat(neutral.overrides("orange")).isFalse();
    }
}
