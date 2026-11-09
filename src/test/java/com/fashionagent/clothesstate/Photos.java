package com.fashionagent.clothesstate;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.Base64;
import javax.imageio.ImageIO;

/** 시험용 단색 사진. */
public final class Photos {

    private Photos() {
    }

    public static String jpeg(Color color) {
        BufferedImage img = new BufferedImage(320, 240, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        g.setColor(color);
        g.fillRect(0, 0, 320, 240);
        g.dispose();
        try (var out = new ByteArrayOutputStream()) {
            ImageIO.write(img, "jpg", out);
            return Base64.getEncoder().encodeToString(out.toByteArray());
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /** 바닥 색 위에 왼쪽 옷(left)과 오른쪽 옷(right)을 그린 사진. 박스는 시험의 옷 박스와 같음. */
    public static String halves(Color floor, Color left, Color right) {
        BufferedImage img = new BufferedImage(400, 400, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        g.setColor(floor);
        g.fillRect(0, 0, 400, 400);
        g.setColor(left);
        g.fillRect(40, 60, 120, 200);
        g.setColor(right);
        g.fillRect(240, 60, 120, 300);
        g.dispose();
        return encode(img);
    }

    private static String encode(BufferedImage img) {
        try (var out = new ByteArrayOutputStream()) {
            ImageIO.write(img, "jpg", out);
            return Base64.getEncoder().encodeToString(out.toByteArray());
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }
}
