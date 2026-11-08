package com.fashionagent.clothesstate.detect;

/** 모델이 찾은 결함 하나. kind는 stain | tear, 박스는 원본 사진을 1로 본 비율(왼쪽 위 기준). */
public record Detection(String kind, double confidence, double x, double y, double w, double h) {

    public double centerX() {
        return x + w / 2;
    }

    public double centerY() {
        return y + h / 2;
    }

    double iou(Detection o) {
        double ix = Math.max(0, Math.min(x + w, o.x + o.w) - Math.max(x, o.x));
        double iy = Math.max(0, Math.min(y + h, o.y + o.h) - Math.max(y, o.y));
        double inter = ix * iy;
        double union = w * h + o.w * o.h - inter;
        return union <= 0 ? 0 : inter / union;
    }
}
