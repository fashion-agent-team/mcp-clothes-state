# 색 조합 규칙 검증 스크립트

[색 조합 규칙](../../docs/color-rules.md)을 실제 사진으로 확인할 때 씁니다. 사진·라벨은 레포에 올리지 않습니다(AI Hub 약관).

1. `./gradlew bootJar` 후 jar를 풉니다: `unzip build/libs/mcp-clothes-state-0.1.0.jar -d bj`
2. 컴파일: `javac -encoding UTF-8 -cp "bj/BOOT-INF/classes:bj/BOOT-INF/lib/*" -d bj ColorBench.java`
3. 실행(입은 사진 폴더 → TSV): `java -cp "bj/BOOT-INF/classes:bj/BOOT-INF/lib/*:bj" ColorBench <사진 폴더> bj/BOOT-INF/classes/models/yolo26n_fashion12.onnx bench.tsv [bj/BOOT-INF/classes/models/color20_cnn.onnx|none] [worn|laid]`
   - 4번째 인자로 색 인식 모델을 주면 서버처럼 모델로 색을 정하고, `none`이거나 빼면 예전 픽셀 규칙으로 정합니다(두 방식 비교용).
   - TSV의 옷마다 `종류:이름:colorFamily:tone:colorName:x,y,w,h`(박스는 사진 대비 비율)를 씁니다.
4. 라벨과 맞춰 보기: `python3 analyze.py bench.tsv labels.csv`
   - labels.csv 열: `img,style,gender,n,q5_1,q1_sum,q2_1` (AI Hub 라벨 JSON을 사진별로 모은 것)

얼룩 모델·GPT는 쓰지 않고 옷 종류 모델(fashion12)과 색 인식만 씁니다. 2026-11-11 결과(픽셀 규칙, 색 인식 모델)는 docs/color-rules.md "검증"에 있습니다.
