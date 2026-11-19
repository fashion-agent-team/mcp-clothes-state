package com.fashionagent.clothesstate;

import org.springframework.ai.tool.ToolCallbackProvider;
import org.springframework.ai.tool.method.MethodToolCallbackProvider;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.ConfigurationPropertiesScan;
import org.springframework.context.annotation.Bean;

@SpringBootApplication
@ConfigurationPropertiesScan
public class McpClothesStateApplication {

    public static void main(String[] args) {
        SpringApplication.run(McpClothesStateApplication.class, args);
    }

    /** {@link ClothesTools}의 @Tool 메서드를 MCP 도구로 내놓는다. */
    @Bean
    ToolCallbackProvider clothesToolCallbacks(ClothesTools tools) {
        return MethodToolCallbackProvider.builder().toolObjects(tools).build();
    }
}
