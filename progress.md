# ROS2 + TurtleBot4 + YOLOv8n 기반 우체국 박스 4호 탐사 프로젝트 진행 계획

## 1. 프로젝트 개요
- ROS2 (권장: Jazzy, Ubuntu 24.04) + TurtleBot4 + RGB 카메라를 사용
- SLAM 을 통해 환경 지도(map)는 이미 생성·저장되어 있다고 가정
- Nav2 를 이용해 지도 기반 자율 주행을 수행하면서, YOLOv8n 으로 목표 Object(대한민국 우체국 박스 4호)를 탐지
- 탐지된 목표 Object를 여러 각도/거리에서 촬영해 YOLO 커스텀 학습용 데이터셋을 자동으로 수집

## 2. 시스템 목표
1. 정적 맵을 기반으로 한 자율 주행으로 실내 환경을 순회
2. 카메라 토픽에서 YOLOv8n 추론을 통해 “우체국 박스 4호” Object 를 실시간 탐지
3. 탐지 신뢰도가 일정 기준(threshold) 이상일 때, 해당 프레임 이미지를 저장
4. 서로 다른 위치/각도/거리에서 여러 장의 이미지를 확보하도록 로봇의 이동 전략 구성
5. 저장된 이미지들을 YOLO 포맷에 맞춰 정리해 향후 재학습/미세조정에 활용

## 3. 하드웨어 / 소프트웨어 환경
- 하드웨어
  - TurtleBot4 (Burger / Lite / Standard 등)
  - 온보드 PC 또는 외부 노트북 (GPU 사용 가능하면 더욱 좋음)
  - RGB 카메라 (예: Intel RealSense, USB WebCam 등)
- 소프트웨어
  - Ubuntu 24.04 LTS
  - ROS2 Jazzy (nav2, slam_toolbox 등 기본 탐사 스택 설치)
  - Python 3.10 이상
  - YOLOv8n (ultralytics 패키지)
  - OpenCV, NumPy 등 기본 이미지 처리 라이브러리

## 4. 데이터셋 폴더 구조 제안
SLAM 및 탐사 과정에서 촬영한 이미지들을 다음과 같은 구조로 저장하는 것을 목표로 한다.

```text
datasets/
  postbox_4/
    raw_capture/          # 로봇이 자동으로 찍은 원본 이미지
    images/
      train/
      val/
      test/
    labels/               # YOLO txt 라벨 (x_center y_center w h, class_id)
      train/
      val/
      test/
```

- raw_capture/ 에는 로봇이 자동으로 저장한 이미지들이 들어가며, 이후 수동 라벨링 도구(labelImg, Roboflow 등)를 사용해 labels/ 를 생성
- images/ 와 labels/ 는 YOLO 학습 시 사용되는 최종 구조
- class_id 0 을 “우체국 박스 4호” 한 클래스로 두고 시작한 뒤, 필요 시 다른 박스/객체 클래스를 확장

## 5. ROS2 노드 및 기능 단위 설계(개념)
1. 자율 주행 / 경로 계획
   - Nav2 를 이용해 global costmap, local costmap 기반 목표 지점 이동
   - 맵 상의 여러 waypoint 를 미리 정의하거나, Frontier 기반 탐사 알고리즘을 사용해 미방문 지역을 순회
2. 카메라 캡처
   - /camera/color/image_raw (또는 사용 중인 카메라 토픽)을 구독
   - Nav2 가 정지 상태이거나 속도가 낮을 때, 이미지 캡처 타이밍을 조절해 모션 블러 최소화
3. YOLOv8n 추론
   - Python 노드에서 YOLOv8n 모델을 로드하고 카메라 이미지를 입력
   - “postbox_4” 클래스에 해당하는 bounding box 가 감지되면, 신뢰도(score)와 함께 결과를 ROS 토픽 또는 내부 큐로 전달
4. 목표 Object 데이터 수집
   - 감지 결과가 기준 신뢰도 이상일 때, 현재 RGB 프레임을 datasets/postbox_4/raw_capture/ 아래에 저장
   - 파일명에 시간, 로봇 위치(좌표), 탐사 회차 등 메타정보를 포함 (예: yyyyMMdd_HHmmss_x_y_theta.png)
   - 별도의 CSV 또는 JSON 로그에 이미지 파일명, 로봇 pose, 감지 박스 정보(bbox, score)를 기록

## 6. 탐사 및 촬영 로직 예시(개념 흐름)
1. 초기화
   - 생성된 맵과 함께 Nav2 bringup
   - YOLOv8n 모델 로드 (사전 학습된 COCO 가중치 또는 우체국 박스 4호로 미리 미세조정한 가중치)
2. 탐사 경로 설정
   - 미리 정의한 waypoint 리스트를 순회하거나, Frontier 기반 탐사 노드 사용
3. 주행 중 탐지
   - 주행 중 카메라 이미지를 실시간으로 YOLOv8n 에 입력
   - postbox_4 가 감지되면, 로봇 속도를 줄이거나 정지한 뒤 다수의 프레임을 촬영
4. 데이터 수집
   - 조건(신뢰도, 최소 간격, 시야각 변화 등)을 만족하는 프레임만 저장
   - 동일한 장소에서 너무 많은 중복 이미지가 쌓이지 않도록, pose 변화량 또는 탐사 단계별로 샘플링
5. 종료 및 백업
   - 수집된 datasets/postbox_4 디렉터리를 외장 저장소나 클라우드에 백업

## 7. SLAM/맵과의 연계
- SLAM 으로 생성된 맵은 이미 존재한다고 가정하되, 실제 실행 시에는 다음을 점검
  - 맵 파일(.yaml, .pgm)이 Nav2 bringup 시 정상적으로 로드되는지
  - 로봇의 초기 위치(pose) 설정이 올바른지 (AMCL 초기화)
  - 전역/지역 costmap 에서 카메라 시야를 가리는 장애물이 올바르게 반영되는지
- 목표 Object 촬영 시, 로봇의 현재 좌표를 함께 기록해 “어느 위치에서 어떤 각도의 사진을 찍었는지” 를 추후 분석 가능하게 함

## 8. YOLO 커스텀 학습을 위한 준비
1. 우체국 박스 4호 데이터 수집
   - 다양한 거리(근거리, 중거리, 원거리), 각도(정면, 측면, 대각선), 조명(밝음/어두움) 조건에서 충분한 이미지를 확보
   - 배경이 다른 장소(복도, 실험실, 창고 등)에서도 촬영해 일반화 성능 확보
2. 라벨링
   - datasets/postbox_4/raw_capture/ 의 이미지를 라벨링 도구로 열어 bounding box 와 class_id(0: postbox_4)를 지정
   - 라벨링 결과를 YOLO 포맷(txt)으로 export 하여 images/·labels/ 구조로 정리
3. 학습 스크립트
   - ultralytics YOLO 에서 data.yaml 을 작성하고 train 명령으로 커스텀 학습
   - 학습이 끝난 best.pt 가 실제 로봇에서 사용하는 모델 가중치가 됨

## 9. 향후 확장 아이디어
- 우체국 박스 4호 외에 다른 크기/종류의 박스를 클래스 추가
- 탐지된 Object 위치를 맵에 마킹하여 “우체국 박스 위치 지도” 생성
- 목표 Object 주변의 장애물을 고려한 접근·후퇴 동작 자동화
- 다중 로봇 협업을 통한 더 빠른 데이터 수집

## 10. 오늘 이후 실제 구현 순서(요약 체크리스트)
1. ROS2 + TurtleBot4 + 카메라 환경에서 Nav2 bringup 및 맵 기반 이동 확인
2. 카메라 토픽(/camera/color/image_raw 등) 확인 및 이미지 subscribe 테스트
3. requirements.txt 기반 Python 가상환경 구성 및 YOLOv8n 추론 단일 이미지/영상 테스트
4. ROS2 노드에서 카메라 이미지를 YOLOv8n 에 전달하고, postbox_4 클래스 감지 결과를 로그로 출력
5. 감지 결과를 기준으로 datasets/postbox_4/raw_capture/ 에 이미지와 메타데이터 저장
6. 수집된 이미지에 대해 수동 라벨링 후 YOLO 학습용 구조로 재정리
7. 커스텀 학습된 모델을 로봇에 탑재해 실시간 탐지 성능 검증

## 11. tb4_target_selector ROS2 패키지 및 map_box_selector_node 연동

1. tb4_target_selector 패키지 생성 및 메시지 정의
   - msg:
     - Detection.msg
     - YoloDetections.msg
     - BoxCandidate.msg
     - BoxCandidateArray.msg
     - BestTargetBox.msg
   - 기능:
     - /yolo/detections, /target_box_candidates, /best_target_box 토픽 인터페이스 표준화
2. YOLO 노드 및 map_box_selector_node 구현/통합
   - YOLO 노드:
     - /camera/image_raw, /camera/camera_info, (선택) /camera/depth/image_raw 구독
     - YOLOv5n 커스텀 모델로 타겟 의심 Detection 생성 → /yolo/detections publish
   - map_box_selector_node:
     - /yolo/detections, /map, TF(map-odom-base_link-camera_link) 사용
     - 타겟 의심 발생 시, 맵 ROI 잘라 OpenCV로 장애물/구조물 박스 추정
     - 스코어링 후 /target_box_candidates 및 /best_target_box publish
3. TurtleBot4 실제 환경 연동
   - slam_toolbox + Nav2 bringup
   - tb4_target_selector 패키지 빌드 및 map_box_selector_node 실행
   - /best_target_box 결과를 이용한 후속 행동(접근, 재탐색, 데이터 수집 전략) 설계
4. 성능/파라미터 튜닝
   - target_classes, yolo_conf_threshold, roi_size_m, occ_threshold, min_area_pix, trigger_cooldown_sec 등의 파라미터 실험
   - 수 Hz 수준의 전체 파이프라인 유지 여부 확인 (YOLO + ROI + OpenCV + 스코어링)
5. 최종 통합
   - 데이터 수집 노드, 네비게이션, tb4_target_selector 를 하나의 런치 파일로 묶어 자동 실행 플로우 구성