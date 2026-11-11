# 색 조합 규칙 검증 스크립트

[색 조합 규칙](../../docs/color-rules.md)을 실제 사진으로 확인할 때 씁니다. 사진·라벨은 레포에 올리지 않습니다(AI Hub 약관).

1. `./gradlew bootJar` 후 jar를 풉니다: `unzip build/libs/mcp-clothes-state-0.1.0.jar -d bj`
2. 컴파일: `javac -encoding UTF-8 -cp "bj/BOOT-INF/classes:bj/BOOT-INF/lib/*" -d bj ColorBench.java`
3. 실행(입은 사진 폴더 → TSV): `java -cp "bj/BOOT-INF/classes:bj/BOOT-INF/lib/*:bj" ColorBench <사진 폴더> bj/BOOT-INF/classes/models/yolo26n_fashion12.onnx bench.tsv`
4. 라벨과 맞춰 보기: `python3 analyze.py bench.tsv labels.csv`
   - labels.csv 열: `img,style,gender,n,q5_1,q1_sum,q2_1` (AI Hub 라벨 JSON을 사진별로 모은 것)

얼룩 모델·GPT는 쓰지 않고 옷 종류 모델(fashion12)과 색 인식만 씁니다. 2026-11-11 결과는 docs/color-rules.md "검증"에 있습니다.
