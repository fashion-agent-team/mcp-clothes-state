package com.fashionagent.clothesstate.image;

import static org.assertj.core.api.Assertions.assertThat;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.util.Base64;
import javax.imageio.ImageIO;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

class ImagesTest {

    @Test
    @DisplayName("큰 사진은 건너뛰며 읽어 긴 변을 줄이고, 비율은 그대로")
    void subsampledDecode() throws Exception {
        BufferedImage big = new BufferedImage(4000, 3000, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = big.createGraphics();
        g.setColor(Color.RED);
        g.fillRect(0, 0, 2000, 3000);
        g.dispose();
        var out = new ByteArrayOutputStream();
        ImageIO.write(big, "jpg", out);
        BufferedImage img = Images.decode(Base64.getEncoder().encodeToString(out.toByteArray()), 8L << 20);
        assertThat(Math.max(img.getWidth(), img.getHeight())).isLessThanOrEqualTo(2000).isGreaterThanOrEqualTo(1300);
        assertThat((double) img.getWidth() / img.getHeight()).isCloseTo(4.0 / 3, org.assertj.core.data.Offset.offset(0.01));
        assertThat((img.getRGB(10, 10) >> 16) & 0xFF).isGreaterThan(200);
    }

    @Test
    @DisplayName("작은 사진은 그대로 읽음")
    void smallDecode() throws Exception {
        BufferedImage small = new BufferedImage(640, 480, BufferedImage.TYPE_INT_RGB);
        var out = new ByteArrayOutputStream();
        ImageIO.write(small, "png", out);
        BufferedImage img = Images.decode(Base64.getEncoder().encodeToString(out.toByteArray()), 8L << 20);
        assertThat(img.getWidth()).isEqualTo(640);
        assertThat(img.getHeight()).isEqualTo(480);
    }
}
