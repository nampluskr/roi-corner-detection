# ROI Corner Detection 방법론 비교

이 문서는 `roi-corner-detection` 프로젝트에서 평가하는 10개 방법론의 상세 비교 지침이다.
모든 방법론은 공통 CSV 스키마(`image_dir,image_name,x1,y1,...,x4,y4`, 정규화 [0,1],
TL->TR->BR->BL)를 입출력 기준으로 삼고, 공통 `Dataloader`로부터 입력을 받아 방법론별
전용 `preprocessor`가 학습 타깃을 생성한다. 우선순위 1-3은 고정이고, 4-10은 프로젝트
목적(fringe 패널 도메인, F1-F8 제약) 적합도 순으로 재정렬한 결과이다.
제약 F1-F8의 정의와 출처는 `README.md` 2절을 참조한다.

## 1. 전체 비교 표

| 우선순위 | 코드 | 방법론 | 출처 | raw 출력 | 후처리 | 학습 필요 | 비용 |
|---:|---|---|---|---|---|---|---|
| 1 | `direct` | Direct Coordinate Regression | 카탈로그 A | (N, 8) 좌표 logits | sigmoid + reshape | 필요 | 경량 |
| 2 | `seg` | Segmentation Corner | 카탈로그 C | (N, 1, H, W) quad 마스크 | findContours + approxPolyDP | 필요 | 중간 |
| 3 | `detect` | BBox Keypoint Detection | 신규 설계 | (N, A, 5+4) 그리드 박스 예측 | 클래스별 top-1 박스 중심 | 필요 | 중간 |
| 4 | `heatmap` | Heatmap Keypoint Detection | 카탈로그 B | (N, 4, H, W) heatmap | soft-argmax | 필요 | 중간 |
| 5 | `hybrid` | DL + Classical CV Hybrid | 카탈로그 I | (N, 1, H, W) 마스크 | Canny + Hough + cornerSubPix | 필요(seg 단계만) | 경량 |
| 6 | `line` | Line Intersection | 카탈로그 D | 직선 세그먼트 | 직선 그룹화 + 교점 계산 | 부분(M-LSD 사전학습) | 경량 |
| 7 | `doc` | Document Pretrained | 카탈로그 J | (N, 8) 좌표 logits | sigmoid + reshape | 파인튜닝만 | 중량 |
| 8 | `homography` | Homography Regression | 카탈로그 E | (N, 8) offset | 정준 좌표 + offset | 필요 | 경량 |
| 9 | `foundation` | Foundation Adapter | 카탈로그 H | (N, 8) 좌표 logits | sigmoid + reshape | 최소(head만) | 중량 |
| 10 | `gcn` | Polygon GCN | 카탈로그 F | 코너 + GCN 반복 정제 | 최종 반복 출력 | 필요 | 중량 |

## 2. 방법론별 상세

### 2.1 direct (우선순위 1, 카탈로그 A)

- **접근 방식**: CNN backbone(ResNet18/34/50) -> FC(8) -> Sigmoid로 8개 좌표를 직접 회귀
- **raw 출력**: (N, 8) 좌표 logits
- **preprocessor 타깃**: 정규화 좌표 그대로 (N, 4, 2) -> (N, 8)
- **loss**: SmoothL1 또는 Wing Loss (logits vs 좌표 타깃)
- **후처리**: sigmoid + reshape -> (N, 4, 2)
- **장점**: 구조 단순, 베이스라인으로 최적, 추론 빠름
- **단점**: 공간 정보 손실로 서브픽셀 정밀도 한계 (F5 취약)
- **F1-F8 적합성**: F1-F4 우수, F5 보통, F6 경량(CPU 80-150ms 수준)

### 2.2 seg (우선순위 2, 카탈로그 C)

- **접근 방식**: quad 영역 세그멘테이션 -> 컨투어 기반 코너 추출
- **raw 출력**: (N, 1, H, W) quad 마스크
- **preprocessor 타깃**: corners -> 채운 폴리곤 마스크 (`src/utils/geometry.py`의
  `corners_to_mask` 재사용)
- **loss**: BCE/Dice (mask vs 마스크 타깃)
- **후처리**: findContours + approxPolyDP -> 4 코너 -> (N, 4, 2)
- **장점**: 경계가 선명한 fringe 패널에 적합, 마스크 supervision이 안정적
- **단점**: approxPolyDP가 4점을 보장하지 않아 후처리 예외 처리 필요
- **F1-F8 적합성**: F1-F2 우수, F5는 컨투어 해상도에 의존(cornerSubPix 후정제 옵션),
  F8(가림 없음)이 단일 컨투어 가정의 근거

### 2.3 detect (우선순위 3, 신규 설계)

- **접근 방식**: YOLO/CenterNet 스타일 바운딩박스 기반 keypoint detector.
  4개 코너를 작은 박스 객체로 검출하고 클래스(TL/TR/BR/BL)로 구분
- **raw 출력**: (N, A, 5+4) 그리드 박스 예측 (objectness, dx, dy, w, h + 4-class)
- **preprocessor 타깃**: corners -> 코너별 고정 크기 박스, 클래스 = 코너 인덱스
- **loss**: box 회귀(IoU/L1) + objectness/class BCE
- **후처리**: 클래스별 top-1 박스 선택 -> 박스 중심 -> (N, 4, 2)
- **장점**: 검출 프레임워크의 강건성, 코너 순서가 클래스로 자동 결정
- **단점**: 카탈로그에 없는 신규 설계라 하이퍼파라미터 탐색 비용 존재
- **F1-F8 적합성**: F1-F3 우수, F5는 박스 중심 해상도에 의존(cornerSubPix 후정제 옵션),
  F4는 합성 단계 의존이 큰 편, F8(가림 없음)이 클래스별 정확히 1개 검출 가정의 근거
- **비고**: 카탈로그 G(DETR-style set prediction)와 다른 접근. G는 대규모 합성
  데이터 요구(F4-F5 충돌)로 제외하고 bbox 기반으로 새로 설계. 코너별 소형 박스는
  quad 형상을 box로 근사하는 것이 아니므로 F1(OBB 배제)에 저촉되지 않는다.
  코너가 경계에 가까우면 박스가 이미지 밖으로 나갈 수 있어 preprocessor에서
  박스 경계 클리핑 처리가 필요하다 (F3)

### 2.4 heatmap (우선순위 4, 카탈로그 B)

- **접근 방식**: 코너별 4채널 gaussian heatmap 예측 + soft-argmax
- **raw 출력**: (N, 4, H, W) heatmap
- **preprocessor 타깃**: corners -> gaussian heatmap
- **loss**: MSE (heatmap vs heatmap 타깃)
- **후처리**: soft-argmax -> (N, 4, 2) (서브픽셀)
- **장점**: 서브픽셀 정밀도 우수 (F5 강점), keypoint 검출의 표준 접근
- **단점**: 해상도-메모리 트레이드오프, 후처리 미분 불가 구간 주의
- **F1-F8 적합성**: F5 최상위(태생적 서브픽셀), F1-F4 우수, F6은 heatmap 해상도에 따라 중간

### 2.5 hybrid (우선순위 5, 카탈로그 I)

- **접근 방식**: 경량 세그멘테이션(MobileNetV3-UNet) + 고전 CV 정밀화
- **raw 출력**: (N, 1, H, W) 세그멘테이션 마스크
- **preprocessor 타깃**: corners -> 마스크
- **loss**: BCE/Dice (seg 단계만 학습)
- **후처리**: Canny + HoughLinesP + cornerSubPix -> (N, 4, 2)
- **장점**: CPU 배포 최적 후보, cornerSubPix로 서브픽셀 확보
- **단점**: 후처리 단계가 많아 실패 모드 다양
- **F1-F8 적합성**: F5 우수(cornerSubPix, 태생적 서브픽셀), F6 최적(CPU 30-60ms 수준),
  F8이 후처리 단순화의 근거

### 2.6 line (우선순위 6, 카탈로그 D)

- **접근 방식**: 직선 검출(M-LSD 또는 Canny+Hough) 후 4변 그룹화 + 교점 계산
- **raw 출력**: 직선 세그먼트 목록
- **preprocessor 타깃**: corners -> 직선/엣지 타깃
- **loss**: 직선 표현 loss (M-LSD 파인튜닝 시)
- **후처리**: 직선 그룹화 + 교점 계산 -> (N, 4, 2)
- **장점**: 경량, 직선 경계가 뚜렷한 fringe 도메인에 적합
- **단점**: 곡률/부분 가림에 취약, 그룹화 휴리스틱 필요
- **F1-F8 적합성**: F2(단일 대형 객체) 전제에서 안정적, F5 우수, F6 우수(CPU 25-60ms 수준),
  F8(가림 없음)이 4변 그룹화 휴리스틱의 근거

### 2.7 doc (우선순위 7, 카탈로그 J)

- **접근 방식**: DocTr/DocScanner 등 문서 코너 사전학습 모델 파인튜닝
- **raw 출력**: (N, 8) 좌표 logits (사전학습 기반)
- **preprocessor 타깃**: 정규화 좌표 그대로
- **loss**: logits vs 좌표 타깃
- **후처리**: sigmoid + reshape -> (N, 4, 2)
- **장점**: 공개 가중치로 빠른 실험, few-shot 적응 가능 (F4 대응)
- **단점**: 외부 패키지 의존, 문서 도메인과 fringe 도메인 간극
- **F1-F8 적합성**: F4 강점, 도메인 간극이 변수, F6은 무거운 편

### 2.8 homography (우선순위 8, 카탈로그 E)

- **접근 방식**: 정준 사각형(canonical square) 기준 8개 offset 회귀
- **raw 출력**: (N, 8) offset
- **preprocessor 타깃**: corners -> 정준 사각형 기준 offset
- **loss**: offset vs offset 타깃 (SmoothL1)
- **후처리**: 정준 좌표 + offset -> (N, 4, 2)
- **장점**: direct와 유사한 경량 구조, 기하 prior 내장
- **단점**: direct 대비 이득이 제한적일 수 있음 (비교 실험 가치)
- **F1-F8 적합성**: direct와 유사, F1 기하 prior로 소폭 유리, F6 경량

### 2.9 foundation (우선순위 9, 카탈로그 H)

- **접근 방식**: frozen DINOv2/SAM backbone + 경량 head만 학습 (few-shot)
- **raw 출력**: (N, 8) 좌표 logits
- **preprocessor 타깃**: 정규화 좌표 그대로 (direct와 동일)
- **loss**: logits vs 좌표 타깃
- **후처리**: sigmoid + reshape -> (N, 4, 2)
- **장점**: 실측 데이터 극소량으로 상한 성능 탐색 (F4 극단 대응)
- **단점**: backbone 무거움, 실시간 배포 부적합
- **F1-F8 적합성**: F4 최강(few-shot), F6 최하위(frozen backbone 무거움)
- **비고**: 성능 상한(upper bound) 탐색용 보험 성격

### 2.10 gcn (우선순위 10, 카탈로그 F)

- **접근 방식**: 초기 코너 추정 -> GCN(Graph Convolutional Network)으로 1-3회 반복 정제
- **raw 출력**: 초기 코너 + 반복별 정제 출력
- **preprocessor 타깃**: corners (반복 단계별 supervision)
- **loss**: 반복 단계별 좌표 loss 합산
- **후처리**: 최종 반복 출력 -> (N, 4, 2)
- **장점**: 반복 정제로 정밀도 향상 여지 (F5 잠재력)
- **단점**: 구조 복잡, 학습 무거움, 초기 추정 품질에 의존
- **F1-F8 적합성**: F5 잠재력 있으나 비용 대비 효율 검증 필요, F4 합성 의존 큼, F6 무거움

## 3. F1-F8 제약과 방법론별 설계 영향

| 제약 | 내용 | 주요 영향 방법론 |
|---|---|---|
| F1 | 임의의 볼록 사각형 (OBB 아님) | 전체: quad를 box로 근사하는 접근 배제. detect의 코너별 소형 박스는 무관. homography는 기하 prior로 소폭 유리 |
| F2 | 단일 객체, 이미지의 50% 이상 차지 | 전체: 패널 검출 stage 불필요. detect: 코너 간 거리가 충분해 클래스별 top-1 선택 안정. line: 4변이 화면 대부분을 가로질러 검출 용이 |
| F3 | 4개 코너는 항상 이미지 경계 내부 | 전체: 회전 증강 +-5도, clipping 검증 (transforms.py 구현 완료). detect: 코너 박스 경계 클리핑 처리 필요 |
| F4 | 실측 소량, 합성 대량 | doc/foundation: F4 대응책 (사전학습/few-shot). gcn/detect: 데이터 요구가 커 합성 단계 의존 큼. 카탈로그 G(DETR) 제외 사유 |
| F5 | 서브픽셀 정밀도 (위상 복원) | heatmap(soft-argmax)/hybrid(cornerSubPix): 태생적 서브픽셀. seg/detect: 픽셀 단위라 cornerSubPix 후정제 옵션. 전체: Reprojection Error 필수 추적 |
| F6 | CPU 배포 지연/모델 크기 (예산 TBD) | hybrid/line/homography: 경량 강점. foundation/gcn/doc: 무거움. 전체: benchmark.py가 latency/모델 크기 측정 (latency는 전처리 -> 추론 -> 후처리 end-to-end 기준) |
| F7 | 조명/글레어/비네팅 변동 | 전체: ColorJitter/GaussianBlur/GaussianNoise 광학 증강 필수 (transforms.py 구현 완료), 합성 fringe에 반사/글레어 시뮬레이션 포함 |
| F8 | 패널 가림 없음 | line: 4변 그룹화 휴리스틱 근거. seg: 단일 컨투어 가정 근거. detect: 클래스별 정확히 1개 검출 가정 근거 |

## 4. 공통 아키텍처 규약

- 모든 방법론은 `src/models/base/`의 5종 추상 클래스(`BaseModel`, `BasePreprocessor`,
  `BasePostprocessor`, `BaseLoss`, `BaseWrapper`)를 상속한다.
- 모든 모델은 공통 `Dataloader`(`src/data/dataloader.py`)로부터 (image, corners) 배치를
  받고, 방법론별 전용 `preprocessor.py`가 표준 코너 (N, 4, 2)를 학습 타깃으로 변환한다.
- 모든 `postprocessor.py`는 raw 출력을 표준 코너 (N, 4, 2)로 변환하며, 이후
  `pred_corners.csv` 기록과 `src/metrics/metrics.py` 평가에는 방법론 이름이 등장하지
  않는다.
- 구현 순서와 Phase 배치는 `PLAN.md`, 모듈 시그니처는 `README.md`(SSOT)를 따른다.
