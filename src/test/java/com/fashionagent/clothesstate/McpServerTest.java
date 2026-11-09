package com.fashionagent.clothesstate;

import static org.assertj.core.api.Assertions.assertThat;

import io.modelcontextprotocol.client.McpClient;
import io.modelcontextprotocol.client.McpSyncClient;
import io.modelcontextprotocol.client.transport.HttpClientStreamableHttpTransport;
import io.modelcontextprotocol.spec.McpSchema;
import java.awt.Color;
import java.util.Base64;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.context.TestConfiguration;
import org.springframework.boot.test.web.server.LocalServerPort;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Primary;
import org.springframework.web.client.RestClient;

/** 실제로 서버를 띄우고 MCP 클라이언트로 /mcp에 붙어 본다(agent가 붙는 방식과 같음). 모델은 FakeDetector로 바꿔 끼움. */
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT,
        properties = {"clothes.model=classpath:models/none.onnx", "clothes.garment-model=classpath:models/none.onnx",
                "clothes.gpt.api-key=", "clothes.gpt.candidate-dir="})
class McpServerTest {

    @TestConfiguration
    static class Fake {
        @Bean
        @Primary
        FakeDetector fakeDetector() {
            return new FakeDetector();
        }
    }

    @LocalServerPort
    int port;

    McpSyncClient client;

    @BeforeEach
    void setUp() {
        var transport = HttpClientStreamableHttpTransport.builder("http://localhost:" + port).endpoint("/mcp").build();
        client = McpClient.sync(transport).build();
        client.initialize();
    }

    @AfterEach
    void close() {
        client.closeGracefully();
    }

    @Test
    @DisplayName("tools list에 clothes_inspect 하나, imageBase64만 필수")
    @SuppressWarnings("unchecked")
    void toolsList() {
        var tools = client.listTools().tools();
        assertThat(tools).extracting(McpSchema.Tool::name).containsExactly("clothes_inspect");
        var schema = tools.getFirst().inputSchema();
        assertThat((List<String>) schema.get("required")).containsExactly("imageBase64");
        assertThat(schema.get("properties").toString()).contains("mode", "view", "allowGpt");
    }

    @Test
    @DisplayName("사진 → garments·defects·pendingHelper·source·modelVersion JSON")
    void inspect() {
        String out = text(call(Map.of("imageBase64", Photos.jpeg(new Color(200, 200, 200)))));
        assertThat(out).contains("\"id\":\"g1\"", "\"kind\":\"stain\"", "\"where\":\"옷 오른쪽 위쪽\"",
                "\"confidence\":0.82", "\"pendingHelper\":{", "\"candidates\":[", "\"lighting\":\"ok\"",
                "\"source\":\"model\"", "\"modelVersion\":\"yolo26n-v6\"").doesNotContain("error");
    }

    @Test
    @DisplayName("잘못된 입력은 error JSON")
    void errors() {
        assertThat(text(call(Map.of("imageBase64", "")))).contains("\"code\":\"no_image\"");
        assertThat(text(call(Map.of("imageBase64", Base64.getEncoder().encodeToString("not an image".getBytes())))))
                .contains("\"code\":\"bad_image\"");
        assertThat(text(call(Map.of("imageBase64", Photos.jpeg(Color.WHITE), "mode", "hanging"))))
                .contains("\"code\":\"bad_request\"");
    }

    @Test
    @DisplayName("actuator health는 UP")
    void health() {
        assertThat(RestClient.create("http://localhost:" + port).get().uri("/actuator/health").retrieve().body(String.class))
                .contains("UP");
    }

    private McpSchema.CallToolResult call(Map<String, Object> args) {
        return client.callTool(new McpSchema.CallToolRequest("clothes_inspect", args));
    }

    private static String text(McpSchema.CallToolResult result) {
        return ((McpSchema.TextContent) result.content().getFirst()).text();
    }
}
