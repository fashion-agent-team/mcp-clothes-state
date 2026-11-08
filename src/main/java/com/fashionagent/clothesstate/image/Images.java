package com.fashionagent.clothesstate.image;

import com.fashionagent.clothesstate.ClothesStateException;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.util.Base64;
import javax.imageio.ImageIO;

/** 사진 base64 풀기, 모델 입력 만들기, 밝기 계산. 사진은 메모리에서만 쓰고 저장하지 않습니다. */
public final class Images {

    private Images() {
    }

    public static BufferedImage decode(String imageBase64, long maxBytes) {
        if (imageBase64 == null || imageBase64.isBlank()) {
            throw new ClothesStateException("no_image", "사진이 없어요.");
        }
        String b64 = imageBase64.strip();
        int comma = b64.indexOf(',');
        if (b64.startsWith("data:") && comma > 0) {
            b64 = b64.substring(comma + 1);
        }
        if (b64.length() / 4L * 3 > maxBytes) {
            throw new ClothesStateException("bad_image", "사진이 너무 커요.");
        }
        byte[] bytes;
        try {
            bytes = Base64.getMimeDecoder().decode(b64);
        } catch (IllegalArgumentException e) {
            throw new ClothesStateException("bad_image", "사진을 읽지 못했어요.");
        }
        try {
            BufferedImage img = ImageIO.read(new ByteArrayInputStream(bytes));
            if (img == null || img.getWidth() < 32 || img.getHeight() < 32) {
                throw new ClothesStateException("bad_image", "사진을 읽지 못했어요.");
            }
            return img;
        } catch (IOException e) {
            throw new ClothesStateException("bad_image", "사진을 읽지 못했어요.");
        }
    }

    /**
     * 비율을 유지해 긴 변을 size로 줄이고, 남는 곳을 회색(114)으로 채운 뒤 RGB 0~1 CHW 배열로 만든다
     * (Ultralytics LetterBox와 같음). rect=true면 학습·평가 때처럼 짧은 변을 32의 배수까지만 채운 직사각형
     * (예: 512×384), false면 size×size 정사각형. 가운데 정렬.
     */
    public static Letterbox letterbox(BufferedImage img, int size, boolean rect) {
        double scale = Math.min((double) size / img.getWidth(), (double) size / img.getHeight());
        int w = Math.max(1, (int) Math.round(img.getWidth() * scale));
        int h = Math.max(1, (int) Math.round(img.getHeight() * scale));
        int dw = size - w;
        int dh = size - h;
        if (rect) {
            dw %= 32;
            dh %= 32;
        }
        int outW = w + dw;
        int outH = h + dh;
        int padX = (int) Math.round(dw / 2.0 - 0.1);
        int padY = (int) Math.round(dh / 2.0 - 0.1);
        BufferedImage canvas = new BufferedImage(outW, outH, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = canvas.createGraphics();
        g.setColor(new Color(114, 114, 114));
        g.fillRect(0, 0, outW, outH);
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.drawImage(img, padX, padY, w, h, null);
        g.dispose();
        int plane = outW * outH;
        float[] chw = new float[3 * plane];
        for (int y = 0; y < outH; y++) {
            for (int x = 0; x < outW; x++) {
                int rgb = canvas.getRGB(x, y);
                int i = y * outW + x;
                chw[i] = ((rgb >> 16) & 0xFF) / 255f;
                chw[plane + i] = ((rgb >> 8) & 0xFF) / 255f;
                chw[2 * plane + i] = (rgb & 0xFF) / 255f;
            }
        }
        return new Letterbox(chw, outW, outH, scale, padX, padY, img.getWidth(), img.getHeight());
    }

    /** 정사각형 입력(size×size). */
    public static Letterbox letterbox(BufferedImage img, int size) {
        return letterbox(img, size, false);
    }

    /** 모델 입력과, 모델 좌표(입력 픽셀)를 원본 사진 비율(0~1)로 되돌리는 값. */
    public record Letterbox(float[] chw, int inputW, int inputH, double scale, int padX, int padY, int width, int height) {

        public double toRatioX(double modelX) {
            return clamp((modelX - padX) / scale / width);
        }

        public double toRatioY(double modelY) {
            return clamp((modelY - padY) / scale / height);
        }

        private static double clamp(double v) {
            return Math.max(0, Math.min(1, v));
        }
    }

    /** 사진 평균 밝기(CIELAB L*, 0~100). 대략 64×64 점만 본다. */
    public static double meanLightness(BufferedImage img) {
        int stepX = Math.max(1, img.getWidth() / 64);
        int stepY = Math.max(1, img.getHeight() / 64);
        double sum = 0;
        int n = 0;
        for (int y = 0; y < img.getHeight(); y += stepY) {
            for (int x = 0; x < img.getWidth(); x += stepX) {
                int rgb = img.getRGB(x, y);
                double yLin = 0.2126729 * linear(((rgb >> 16) & 0xFF) / 255.0)
                        + 0.7151522 * linear(((rgb >> 8) & 0xFF) / 255.0)
                        + 0.0721750 * linear((rgb & 0xFF) / 255.0);
                sum += yLin > 216.0 / 24389 ? 116 * Math.cbrt(yLin) - 16 : 24389.0 / 27 * yLin;
                n++;
            }
        }
        return sum / n;
    }

    private static double linear(double c) {
        return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
    }
}
