# ROI Corner Detection

평면 사각형 객체(문서, ID 카드, 검사 대상 패널 등)의 4개 코너 좌표를 검출하는 여러 방법론을
하나의 공유 데이터/평가 파이프라인 위에서 구현하고 비교하는 프로젝트이다.

## 1. 방법론

| 코드 | 방법론 | 구현 순서 | 핵심 아이디어 | 추가 의존성 | 복잡도 |
|---|---|---:|---|---|---|
| `direct` | Direct Coordinate Regression (카탈로그 A) | 1 | CNN backbone $\to$ FC(8), Sigmoid + SmoothL1/Wing Loss | 없음 | 낮음 |
| `homography` | Homography Regression (카탈로그 E) | 2 | 정준 사각형 기준 8개 offset 회귀 | 없음 | 낮음 |
| `heatmap` | Heatmap Keypoint Detection (카탈로그 B) | 3 | 4채널 heatmap + soft-argmax, 서브픽셀 정밀도 | 없음 | 중간 |
| `seg` | Segmentation Corner (카탈로그 C) | 4 | quad 마스크 세그멘테이션 $\to$ findContours + approxPolyDP | OpenCV(기존 사용) | 중간 |
| `hybrid` | DL + Classical CV Hybrid (카탈로그 I) | 5 | Segmentation(MobileNetV3-UNet) + Canny/Hough/cornerSubPix | OpenCV(기존 사용) | 중간 |
| `det` | BBox Keypoint Detection (신규 설계) | 6 | 코너 4개를 작은 박스 객체로 검출 (클래스 = 코너 id), 박스 중심 $\to$ 코너 | 없음 | 중간 |
| `gcn` | Polygon GCN (카탈로그 F) | 7 | 초기 코너 $\to$ GCN 반복 정제 (1-3회) | 없음 | 높음 |
| `doc` | Document Pretrained (카탈로그 J) | 8 | DocTr/DocScanner 파인튜닝, few-shot 적응 | DocTr 패키지(외부) | 높음 |
| `foundation` | Foundation Adapter (카탈로그 H) | 9 | frozen DINOv2/SAM backbone + 경량 head, few-shot 상한 탐색 | DINOv2/SAM 가중치(외부) | 높음 |
| `line` | Line Intersection (카탈로그 D) | 10 | 직선 검출(M-LSD 또는 Canny+Hough) + 교점 계산 | M-LSD 가중치(외부) | 중간 |
| `torchseg` | Torchvision Segmentation Corner (신규, torchvision 재사용) | 11 | torchvision 세그멘테이션 모델(DeepLabV3/FCN/LR-ASPP) end-to-end + 독립 contour 후처리 | OpenCV(기존 사용), torchvision seg 가중치 | 낮음 |
| `torchdet` | Torchvision Detection (신규, torchvision 재사용) | 12 | torchvision 검출 모델(Faster R-CNN/RetinaNet/SSD) end-to-end, 코너별 소형 박스 $\to$ 박스 중심 | torchvision det 가중치(외부) | 중간 |

구현 순서는 구현 난이도와 속도 기준으로 정렬한 착수 순서이며, 상세 근거는
`docs/common/roi-corner-detection-implementation-order.md`를 참조한다. `direct`를 베이스라인으로
두고, 그 사촌인 `homography`(동일 FC head)와 외부 의존이 없는 `heatmap`을 앞에 둔다. `seg`는
기존에 `hybrid`(I)의 세그멘테이션 단계로 흡수되어 있던 카탈로그 C를 독립 방법론으로 승격한
것이고(그래서 `seg` 직후 `hybrid`가 동일한 마스크 model을 독립 구현한다), `det`는 카탈로그 G
(DETR-style)가 아닌 바운딩박스 기반 keypoint detector로 새로 설계한 방법론이다. 외부
가중치/패키지 의존이 있는 `doc`, `foundation`, `line`은 뒤로 둔다. 마지막의 `torchseg`,
`torchdet`은 커스텀 model을 직접 설계하는 `seg`/`det`와 달리 torchvision의 세그멘테이션/
검출 모델을 head까지 그대로(end-to-end) 재사용하는 라이브러리 기반 방법론으로, 커스텀 설계
대비 성능/비용을 비교하기 위한 대조군이다 (`torchseg`는 `seg`, `torchdet`은 `det`와 대비).

모든 방법론은 `src/models/base/`의 4종 추상 클래스(`BaseModel`, `BasePreprocessor`,
`BasePostprocessor`, `BaseWrapper`)를 직접 상속하고, 공유 `Trainer`/`Evaluator`/
`Predictor`(`src/core/`), 공통 loss(`src/losses/`, `BaseLoss` 상속), 공통 메트릭(`src/metrics/`)을
그대로 사용한다. `BaseWrapper`가 공통 `train_step`/`eval_step`과 단순 대입 세터를 제공하고
각 wrapper가 `__init__`에서 `set_optimizer` 다음 `set_scheduler(self.scheduler or ...)`로
optimizer와 대칭되게 scheduler를 확정하며, 방법론별 loss 계산 차이는 `compute_losses()`
오버라이드로 처리한다. torchvision detector처럼
모델 호출 규약 자체가 다른 방법론만 전체 step을 오버라이드한다. 모든 모델은
공통 `Dataloader`(`src/data/dataloader.py`)로부터 입력을 받고, 방법론별 전용
`preprocessor.py`가 표준 코너 `(N, 4, 2)`를 해당 방법론의 학습 타깃으로 변환한다.
방법론 간 성능 비교는 `experiments/benchmark.py`가 담당한다.
12개 방법론의 상세 비교는 `docs/common/roi-corner-detection-models.md`를 참조한다.

## 2. 핵심 제약 (F1-F8)

| 제약 | 내용 | 설계 영향 |
|---|---|---|
| F1 | 임의의 볼록 사각형 (OBB 아님, 원근 투영 결과) | 8개 자유 좌표 직접 회귀/검출. quad 전체를 box로 근사하는 접근 배제 (detect의 코너별 소형 박스는 무관) |
| F2 | 단일 객체, 이미지의 50% 이상 차지 | 패널 검출 stage 불필요. 코너 간 거리가 충분해 detect의 클래스별 top-1 선택 안정 |
| F3 | 4개 코너는 항상 이미지 경계 내부 | 회전 증강 +-5도 제한, clipping 검증 필수. det 코너 박스는 경계 클리핑 처리 |
| F4 | 실측 데이터 적음, 합성 데이터 다수 | 3단계 학습 전략 (공개 데이터 $\to$ 합성 $\to$ 실측 파인튜닝). doc/foundation이 F4 대응책, gcn/detect는 합성 단계 의존 큼 |
| F5 | 서브픽셀 정밀도가 중요 (위상 복원용) | Reprojection Error를 IoU와 함께 필수 메트릭으로 추적. heatmap/hybrid는 태생적 서브픽셀, seg/detect는 cornerSubPix 후정제 옵션 |
| F6 | CPU 배포 지연/모델 크기 제약 (예산 TBD) | `experiments/benchmark.py`가 GPU/CPU latency, 모델 크기 필수 측정. 경량 방법론(hybrid, line, homography) 우선순위 근거 |
| F7 | 조명/글레어/비네팅 변동 존재 | ColorJitter, GaussianBlur, GaussianNoise 광학 증강 필수. 합성 fringe에 반사/글레어 시뮬레이션 포함 |
| F8 | 패널 가림 없음 (검사 환경 보장) | 후처리 단순화 근거 - line 4변 그룹화, seg 단일 컨투어 가정, det 클래스별 정확히 1개 검출 가정 |

F1-F5는 레거시 프로젝트 `_project/sources/P1-project-overview.md` 3절(PMD fringe 패턴
패널 검사 도메인 사실)에서 유래했고, F6-F8은 레거시에서 미결/비공식 항목으로 남아 있던
것을 방법론 확장 시점에 정식 제약으로 승격한 것이다.

## 3. 데이터 전략 (3단계)

1. **Pre-training**: SmartDoc + MIDV-2020 (공개 데이터, 일반 코너 검출 학습)
2. **Domain Adaptation**: 합성 Fringe 패턴 이미지 (검사 도메인 적응, `scripts/generate_synthetic_labelme.py`로 생성)
3. **Fine-tuning**: 실측 PMD 데이터 (소량, 최종 정밀도 확보)

데이터 레이블링 방법론, 3단계 학습 전략의 근거, 합성 Fringe 패턴 생성 수식은
`docs/common/roi-corner-detection-data-pipeline.md`를 참조한다.

## 4. 평가 메트릭

| 메트릭 | 관점 | 값의 범위 | 좋은 방향 | 설명 |
|---|---|---|---|---|
| Polygon IoU | 면적 겹침 | [0, 1] | 클수록 좋음 | 예측/정답 사각형의 Shoelace 공식 기반 면적 중첩도. 스케일 불변이지만 코너가 변을 따라 미끄러지는 오차에는 둔감 |
| MCD (Mean Corner Distance) | 평균 좌표 오차 | [0, inf) | 작을수록 좋음 | 같은 순번 코너 쌍의 유클리드 거리를 4개 코너에 대해 평균 |
| MaxCD (Max Corner Distance) | 최악 좌표 오차 | [0, inf) | 작을수록 좋음 | 4개 코너 중 최대 거리. 코너 하나의 큰 오차(worst-case)를 탐지 |
| Reprojection Error | 기하 변환 품질 | [0, inf) | 작을수록 좋음 | 기준점 집합을 예측 코너 기반 호모그래피 $H_P$와 정답 코너 기반 호모그래피 $H_G$로 각각 투영한 뒤 그 차이를 평균. 위상 복원(원근 보정) 품질과 직결되는 핵심 지표 |
| SR (Success Rate) | 검출 성공 비율 | [0, 1] | 클수록 좋음 | 표준 코너 (4,2)를 반환한 표본 비율. 후처리 실패 모드가 있는 방법론(seg/det/line/hybrid)의 공정 비교 필수 지표 |
| PCK@tau | 임계값 내 성공 비율 | [0, 1] | 클수록 좋음 | MaxCD가 임계값 tau 이하인 표본 비율 (tau는 픽셀 기준, 기본 2px/5px을 정규화 좌표로 환산해 적용) |
| GPU/CPU latency, 모델 크기 | 배포 적합성 | - | 작을수록 좋음 | `experiments/benchmark.py`로 측정. latency는 전처리 $\to$ 추론 $\to$ 후처리 end-to-end 기준, warm-up 후 반복 측정 평균 |

Polygon IoU/MCD/MaxCD/Reprojection Error는 표본(이미지) 단위 지표이고, SR/PCK@tau는
테스트셋 전체에 대한 집계 단위 지표이다. 모든 표본 단위 메트릭은 `src/metrics/`에
`BaseMetric`을 상속하는 클래스로 구현하며(`PolygonIoU`, `MCD`, `MaxCD`,
`ReprojectionError`, `PCK`), 각 방법론의 `postprocessor.py`가 만든 표준 코너
좌표(`(N, 4, 2)`, 정규화 [0, 1])에 대해 방법론 이름과 무관하게 동일하게 적용된다.
호모그래피 추정/점 투영은 `src/utils/homography.py`의
`estimate_homography`/`reproject_points`를 사용한다.

정확도 메트릭(IoU, MCD, MaxCD, RE, PCK)은 검출 성공 표본에 대해서만 계산하고 항상 SR과
함께 보고한다(실패 표본만 빼고 평균하면 실제 실력보다 좋게 보이는 편향이 생기기 때문).
SR은 `src/metrics/`의 클래스가 아니라 postprocessor의 성공 플래그를 Evaluator가
집계하는 지표이다. PCK@tau의 분모는 실패 표본을 포함한 전체 표본 수이며, 실패 표본은 자동으로
PCK 실패(0)로 집계된다. 호모그래피 추정이 불가능한 퇴화 표본(코너가 일직선에 가까운 경우
등)은 Reprojection Error 집계에서 제외하고 실패 건수로 별도 집계한다.

6개 메트릭 각각의 수식, 계산 절차, 해석 방법과 조합 해석 패턴은
`docs/common/roi-corner-detection-metrics.md`를 참조한다.

## 5. 폴더 구조

```text
roi-corner-detection/
├── data/
├── docs/
├── experiments/
│   ├── benchmark.py
│   ├── configs.py
│   └── run.py
├── notebooks/
│   ├── data/
│   ├── models/
│   └── results/
├── outputs/
│   ├── det/
│   ├── direct/
│   ├── doc/
│   ├── foundation/
│   ├── gcn/
│   ├── heatmap/
│   ├── homography/
│   ├── hybrid/
│   ├── line/
│   ├── seg/
│   ├── torchdet/
│   ├── torchseg/
│   └── comparison/
├── scripts/
│   ├── config.py
│   ├── create_data.py
│   ├── evaluate.py
│   ├── fix_data.py
│   ├── predict.py
│   └── train.py
└── src/
    ├── core/
    ├── data/
    ├── metrics/
    ├── models/
    └── utils/
```

| 폴더 | 역할 |
|---|---|
| `data/` | 정답/예측 CSV 등 데이터 산출물 (git-ignore, 예: `gt_corners.csv`) |
| `docs/` | 방법론 카탈로그, 데이터셋 설계, F1-F8 제약 등 설계 문서 |
| `experiments/` | 배치 실행 스크립트 |
| `notebooks/` | EDA 및 방법론 비교 노트북 |
| `outputs/` | 방법론별 체크포인트, 학습 결과, 비교 집계 |
| `scripts/` | CLI 실행 스크립트 - 데이터 준비, 학습/평가/추론, 방법론 비교 |
| `src/` | 공유 유틸/데이터/메트릭/코어 + 방법론별 구현 (아래 6절 참조) |

`configs/` 폴더는 두지 않는다. 하이퍼파라미터는 MNIST 프로젝트와 동일하게
`scripts/config.py`의 `DEFAULTS` 딕셔너리로 관리한다.

## 6. `src`

```text
src/
├── core/
│   ├── evaluator.py
│   ├── factory.py
│   ├── predictor.py
│   └── trainer.py
├── data/
│   ├── dataloader.py
│   ├── dataset.py
│   ├── images.py
│   ├── midv2020.py
│   ├── smartdoc.py
│   └── transforms.py
├── losses/
│   ├── base_loss.py
│   ├── bce_loss.py
│   ├── cross_entropy_loss.py
│   ├── dice_loss.py
│   ├── focal_loss.py
│   ├── mse_loss.py
│   ├── smooth_l1_loss.py
│   └── wing_loss.py
├── metrics/
│   ├── base_metric.py
│   ├── max_cd.py
│   ├── mcd.py
│   ├── pck.py
│   ├── polygon_iou.py
│   ├── reprojection_error.py
│   └── success_rate.py
├── models/
│   ├── base/
│   │   ├── base_model.py
│   │   ├── base_postprocessor.py
│   │   ├── base_preprocessor.py
│   │   └── base_wrapper.py
│   ├── det/
│   │   ├── model.py
│   │   ├── postprocessor.py
│   │   ├── preprocessor.py
│   │   └── wrapper.py
│   ├── direct/      (동일 4개 파일: model.py, postprocessor.py, preprocessor.py, wrapper.py)
│   ├── doc/         (동일 4개 파일)
│   ├── foundation/  (동일 4개 파일)
│   ├── gcn/         (동일 4개 파일)
│   ├── heatmap/     (동일 4개 파일)
│   ├── homography/  (동일 4개 파일)
│   ├── hybrid/      (동일 4개 파일)
│   ├── line/        (동일 4개 파일)
│   ├── seg/         (동일 4개 파일)
│   ├── torchdet/    (동일 4개 파일)
│   └── torchseg/    (동일 4개 파일)
└── utils/
    ├── geometry.py
    ├── homography.py
    ├── io.py
    ├── measure.py
    └── plot.py
```

각 방법론 폴더(`det/direct/doc/foundation/gcn/heatmap/homography/hybrid/line/seg/torchdet/torchseg`)는 항상
**같은 이름의 파일 4개**(`model.py`, `preprocessor.py`, `postprocessor.py`, `wrapper.py`)를 갖고,
내용만 방법론별로 다르다. loss는 `src/losses`의 재사용 클래스를 `wrapper.py`가 조합해 쓴다.
각 파일의 대표 클래스는 대응하는 `BaseModel`/`BasePreprocessor`/`BasePostprocessor`/
`BaseWrapper`를 직접 상속하며, 다른 방법론 폴더를 import하거나 상속하지 않는다.
방법론 간 raw 출력 형태 차이는 다음과 같다 (구현 순서 순).

| 방법론 | raw 출력 (`model.py`) | 학습 타깃 (`preprocessor.py`) | loss (`src/losses`, wrapper가 조합) | 후처리 (`postprocessor.py`) |
|---|---|---|---|---|
| `direct` | (N, 8) 좌표 logits | 정규화 좌표 그대로 | logits vs 좌표 타깃 | sigmoid + reshape (N,4,2) |
| `homography` | (N, 8) offset | corners $\to$ 정준 사각형 기준 offset | offset vs offset 타깃 | 정준 좌표 + offset $\to$ (N,4,2) |
| `heatmap` | (N, 4, H, W) heatmap | corners $\to$ gaussian heatmap | heatmap vs heatmap 타깃 | soft-argmax $\to$ (N,4,2) |
| `seg` | (N, 1, H, W) quad 마스크 | corners $\to$ 채운 폴리곤 마스크 | mask vs 마스크 타깃 | findContours + approxPolyDP $\to$ (N,4,2) |
| `hybrid` | (N, 1, H, W) 세그멘테이션 마스크 | corners $\to$ 마스크 | mask vs 마스크 타깃 | Canny+Hough+cornerSubPix $\to$ (N,4,2) |
| `det` | (N, A, 5+4) 그리드 박스 예측 (obj, dx, dy, w, h, 4-class) | corners $\to$ 코너별 고정 크기 박스 (클래스 = 코너 인덱스) | box 회귀 + objectness/class | 클래스별 top-1 박스 중심 $\to$ (N,4,2) |
| `gcn` | 초기 코너 + GCN 반복 정제 출력 | corners (반복 단계별 supervision) | 반복 단계별 좌표 loss | 최종 반복 출력 $\to$ (N,4,2) |
| `doc` | (N, 8) 좌표 logits (사전학습 기반) | 정규화 좌표 그대로 | logits vs 좌표 타깃 | sigmoid + reshape (N,4,2) |
| `foundation` | (N, 8) 좌표 logits (frozen backbone + 경량 head) | 정규화 좌표 그대로 | logits vs 좌표 타깃 | sigmoid + reshape (N,4,2) |
| `line` | 직선 세그먼트 raw 출력 | corners $\to$ 직선/엣지 타깃 | 직선 표현 loss | 직선 그룹화 + 교점 계산 $\to$ (N,4,2) |
| `torchseg` | (N, 1, H, W) quad 마스크 (torchvision seg 모델) | corners $\to$ 채운 폴리곤 마스크 | mask vs 마스크 타깃 | findContours + approxPolyDP $\to$ (N,4,2) (독립 구현) |
| `torchdet` | torchvision 검출 출력 (학습 시 loss dict, 추론 시 박스/라벨/점수) | corners $\to$ 코너별 소형 박스 + 클래스(코너 id) | torchvision 내장 검출 loss | 클래스별 top-1 박스 중심 $\to$ (N,4,2) |

### 6.1 `src/utils`

**`geometry.py`**

```python
def order_corners(corners): ...        # $\to$ TL, TR, BR, BL 순서로 정렬된 (4, 2) 배열
def is_invalid_corners(corners, min_dist=0.02): ...  # 코너 간 거리가 min_dist 미만이면 True (퇴화 판정)
def mask_to_corners(mask): ...         # 이진 마스크 $\to$ (4, 2) 코너 좌표 (contour 기반)
def polygon_area(corners): ...         # Shoelace 공식 면적 계산
```

**`homography.py`**

```python
def estimate_homography(src_corners, dst_corners): ...  # 코너 4쌍 $\to$ 호모그래피 행렬
def reproject_points(points, homography): ...            # 호모그래피로 점 역투영
```

**`plot.py`**

```python
def show_samples(images, corners=None, ncols=5, title=None, denormalize=False, cell_size=(2, 2)): ...
# images: (N, 3, H, W) 텐서 (get_samples/get_dataset 등 transform이 이미 적용된 결과)
# corners: (N, 4, 2) 텐서 또는 None. None이면 오버레이 없이 이미지만 표시
# ncols: 그리드 열 개수, nrows는 len(images)와 ncols로 자동 계산
# denormalize: True면 Denormalize()(IMAGENET_MEAN/STD 기본값)를 적용 후 clamp(0,1)
# cell_size: 개별 subplot 한 칸의 (width, height) 크기(inch). figsize = (ncols*w, nrows*h)
# 코너가 주어지면 TL/TR/BR/BL 순서로 점 + 라벨(고정 오프셋) + Polygon(fill=False) 외곽선을 그린다

def show_history(history, title=None): ...
# history["train"]["key"] = [val0, val1, ...] 형태의 에폭별 메트릭 스칼라
# history["valid"]["key"]는 선택사항. 각 key마다 subplot 하나씩 train/valid 곡선을 겹쳐 그림
# title: 선택사항, 그래프 최상단에 표시할 제목 문자열
```

**`io.py`**

```python
def save_model(model, checkpoint): ...   # model.state_dict()를 checkpoint(.pth 경로)에 저장
def load_model(model, checkpoint): ...   # checkpoint(.pth 경로)에서 state_dict를 로드해 model에 적용
```

**`measure.py`**

```python
def measure_parameters(model): ...       # $\to$ 모델 전체 파라미터 개수
def measure_size_mb(model): ...          # $\to$ state_dict 기준 모델 크기 (MB)
def measure_latency(model, device, image_size=224, batch_size=1, warmup=5, iters=20): ...  # $\to$ 배치당 평균 추론 지연 (ms)
```

### 6.2 `src/data`

**`dataset.py`**

```python
class Dataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, transform=None): ...
    # csv_path: str 또는 str 리스트 (여러 CSV 병합 로드)
    # transform 생략 시 ToTensor()가 기본 적용
    def _load_csv(self, csv_path): ...  # 공통 CSV 오픈/행 순회, 각 행은 _parse_row에 위임
    def _parse_row(self, row): ...      # 서브클래스가 오버라이드 (NotImplementedError)
    def split(self, split_ratio=0.8, seed=42): ...
    # $\to$ (train_subset, valid_subset), 각각 Subset 인스턴스
    def subset(self, num_samples, seed=42): ...  # $\to$ Subset, 무작위 num_samples개
    def set_transform(self, transform): ...  # $\to$ transform만 교체한 새 Dataset 인스턴스(불변)
    def __len__(self): ...
    def __getitem__(self, idx): ...  # 서브클래스가 오버라이드 (NotImplementedError)

class Subset(torch.utils.data.Dataset):
    # torch.utils.data.Subset을 감싸 set_transform/subset을 지원하는 래퍼
    def __init__(self, dataset, indices): ...
    def set_transform(self, transform): ...  # $\to$ 같은 인덱스, 새 dataset을 가진 새 Subset(불변)
    def subset(self, num_samples, seed=42): ...  # $\to$ Subset, 무작위 num_samples개 (split 결과에서도 재사용 가능)
    def __len__(self): ...
    def __getitem__(self, idx): ...

class CornerDataset(Dataset):
    # x1..y4 코너 컬럼이 있는 CSV용. _parse_row $\to$ (image_path, corners), corners: (4, 2) np.float32
    def __getitem__(self, idx): ...  # $\to$ (image, corners), self.transform(image, corners) 적용 결과

class ImageDataset(Dataset):
    # 코너 컬럼이 없는 CSV용. _parse_row $\to$ image_path
    def __getitem__(self, idx): ...  # $\to$ image, self.transform(image) 적용 결과
```

**`dataloader.py`**

```python
class Dataloader(torch.utils.data.DataLoader):
    def __init__(self, split, dataset, batch_size=16, seed=42): ...
    # split == "train": shuffle=True, drop_last=True, num_workers=4, persistent_workers=True
    # split != "train": shuffle=False, drop_last=False, num_workers=0
    # __len__, __iter__는 DataLoader 상속으로 제공 (별도 구현 없음)
```

**`transforms.py`**

```python
class Compose:
    def __init__(self, transforms): ...
    def __call__(self, image, corners=None): ...
    # corners 생략(None) 시 image만 반환, corners 전달 시 (image, corners) 반환

# 기하 변환 - (image, corners) 동시 변환, corners: (4, 2) normalized [0, 1], TL/TR/BR/BL 순서 보존
# 모든 변환은 __call__(self, image, corners=None)이며, corners=None이면 image만 반환
class Resize:
    def __init__(self, size): ...   # int 또는 (H, W)
class RandomHorizontalFlip:
    def __init__(self, p=0.5): ...    # 코너 재정렬 포함
class RandomVerticalFlip:
    def __init__(self, p=0.5): ...    # 코너 재정렬 포함
class RandomRotation:
    def __init__(self, degrees=5): ...   # +-degrees, clipping 시 원본 유지 (F3)
class RandomPerspective:
    def __init__(self, distortion_scale=0.1, p=0.5): ...
    # 4점 대응 호모그래피를 numpy로 직접 계산해 corners에 적용, clipping 시 원본 유지
class RandomScale:
    def __init__(self, scale_range=(0.9, 1.1)): ...
    # 중심 기준 확대/축소 후 resize+center_crop으로 원본 크기 유지, clipping 시 원본 유지
class RandomAffine:
    def __init__(self, degrees=5, translate=(0.05, 0.05), scale_range=(0.9, 1.1), shear=5): ...
    # 회전+이동+스케일+전단 통합 변환, clipping 시 원본 유지 (F3)

# 이미지 전용 변환 - corners는 그대로 통과 (corners=None이면 image만 반환)
class ColorJitter:
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0): ...
class GaussianBlur:
    def __init__(self, kernel_size, sigma=(0.1, 2.0)): ...
class ToTensor:
    def __call__(self, image, corners=None): ...
    # $\to$ corners 없으면 image_tensor만, 있으면 (image_tensor, corners_tensor) 둘 다 torch.float32
class Normalize:
    def __init__(self, mean=IMAGENET_MEAN, std=IMAGENET_STD): ...
class Denormalize:
    def __init__(self, mean=IMAGENET_MEAN, std=IMAGENET_STD): ...
    # Normalize의 역변환: x * std + mean으로 원본 픽셀 스케일 복원
class ToNumpy:
    def __call__(self, image, corners=None): ...
    # image: (3, H, W) 텐서 $\to$ clamp(0,1) 후 (H, W, 3) numpy. corners는 주어지면 .numpy()로 변환
class GaussianNoise:
    def __init__(self, std=0.05): ...
    # image_tensor에 가우시안 노이즈 추가 후 clamp(0, 1). ToTensor 이후에만 적용 가능
```

`IMAGENET_MEAN`/`IMAGENET_STD`는 `transforms.py` 모듈 상단에 정의된 상수
(`[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`)이며 `Normalize`/`Denormalize`의
기본값으로 쓰인다.

**`smartdoc.py` / `midv2020.py`**

```python
def create_data(data_dir, output_path): ...   # smartdoc.py: data_dir 아래 frame_data.csv + images/ 구조 기대,
                                               # output_path에 gt_corners.csv 저장 (반환값 없음, pandas 미사용)
def create_data(data_dir, output_path): ...   # midv2020.py: data_dir 아래 masks/ + images/ 구조 기대,
                                               # output_path에 gt_corners.csv 저장 (mask $\to$ corners, 반환값 없음)
```

> **CSV 스키마 (공통)**: `image_dir,image_name,x1,y1,x2,y2,x3,y3,x4,y4` (정규화 [0,1], 소수점 6자리,
> TL$\to$TR$\to$BR$\to$BL 시계방향)

**`images.py`**

```python
def create_data(data_dir, output_path): ...
# data_dir 바로 아래 .jpg/.jpeg/.png 이미지 파일만 스캔(대소문자 무관),
# output_path에 코너 컬럼 없는 image_dir,image_name CSV 저장 (ImageDataset용)
```

합성 Fringe 패턴 이미지와 LabelMe 형식 어노테이션 생성은 `src/data/`가 아니라
`scripts/generate_synthetic_labelme.py`가 담당한다 (Domain Adaptation 단계용, F4 제약).

### 6.3 `src/metrics`

메트릭마다 `BaseMetric`을 상속하는 클래스 하나씩, 파일도 하나씩 분리한다. 계산 시그니처는
방법론/도메인 이름을 넣지 않고 `__call__(preds, targets)`로 일반화하며, 함수처럼 호출한다.
메트릭 고유의 설정값(`ReprojectionError`의 `ref_corners`, `PCK`의 `tau`)은 `__call__` 인자가
아니라 `__init__` 인자로 받는다.

**`base_metric.py`**

```python
class BaseMetric:
    def update(self, preds, targets): ...               # 배치를 순회하며 표본별 값을 누적
    def __call__(self, preds, targets): ...             # 서브클래스가 오버라이드 (NotImplementedError)
```

**`polygon_iou.py` / `mcd.py` / `max_cd.py` / `reprojection_error.py` / `pck.py` / `success_rate.py`**

```python
class PolygonIoU(BaseMetric):
    def __call__(self, preds, targets): ...
    # (4,2), (4,2) $\to$ float, Sutherland-Hodgman 클리핑 + utils.geometry.polygon_area 사용

class MCD(BaseMetric):
    def __call__(self, preds, targets): ...             # $\to$ float, Mean Corner Distance

class MaxCD(BaseMetric):
    def __call__(self, preds, targets): ...             # $\to$ float, Max Corner Distance

class ReprojectionError(BaseMetric):
    def __init__(self, ref_corners=None): ...           # ref_corners: 정규 사각형 기준점
    def __call__(self, preds, targets): ...             # $\to$ float, utils.homography 사용

class PCK(BaseMetric):
    def __init__(self, tau=0.02): ...
    def __call__(self, preds, targets): ...             # $\to$ bool, MaxCD <= tau 성공 여부 (MaxCD 재사용)

class SuccessRate(BaseMetric):
    def __call__(self, preds, targets): ...             # $\to$ float, 후처리가 유효한(non-NaN) 좌표를 냈는지
```

### 6.4 `src/losses`

loss마다 `BaseLoss`를 상속하는 클래스 하나씩, 파일도 하나씩 분리한다. metrics와 대칭으로
방법론/모델 이름을 넣지 않고 의미 기반 이름(`WingLoss` 등)으로 정의하며, 각 방법론 wrapper가
`losses` dict로 조합해 쓴다. loss 고유 설정값은 `__init__` 인자로 받는다. `BaseLoss`는
metrics처럼 reset/update/compute 상태를 갖고, 배치 평균 loss를 표본 가중으로 누적한다.
또한 `weight`(기본 1.0)를 가지며, `train_step`이 loss별 `weight`를 곱해 여러 loss를 가중합한다
(`__call__`/`update`는 가중 전 raw loss 기준이라 로그는 원래 스케일을 유지한다). `compute_losses`를
오버라이드하는 wrapper는 반환 dict의 key를 `losses` dict의 key와 일치시킨다.

**`base_loss.py`**

```python
class BaseLoss:
    def __init__(self, weight=1.0): ...           # weight: train_step 가중합 계수
    def reset(self): ...                          # loss 누적 초기화 (total/count)
    def update(self, value, count): ...           # 배치 평균 loss를 표본 가중 누적 (raw)
    def compute(self): ...                        # $\to$ 누적 평균 loss
    def __call__(self, raw_output, target): ...   # forward 호출 + 자동 누적, $\to$ raw scalar loss
    def forward(self, raw_output, target): ...    # 서브클래스가 오버라이드 (NotImplementedError)
```

**`wing_loss.py`**

```python
class WingLoss(BaseLoss):
    def __init__(self, apply_sigmoid=False, w=10.0, epsilon=2.0, weight=1.0): ...
    def forward(self, raw_output, target): ...    # apply_sigmoid=True면 sigmoid 후 Wing loss, $\to$ scalar
```

### 6.5 `src/models/base`

**`base_model.py`**

```python
class BaseModel(nn.Module):
    def forward(self, images): ...   # images: (N, C, H, W) $\to$ raw 출력 (방법론별 형태 자유)
```

**`base_preprocessor.py`**

```python
class BasePreprocessor:
    def __call__(self, corners): ...   # corners: (N, 4, 2) $\to$ 학습 타깃 (방법론별 형태)
```

**`base_postprocessor.py`**

```python
class BasePostprocessor:
    def __call__(self, raw_output): ...   # raw_output $\to$ corners: (N, 4, 2) 표준 포맷
```

**`base_wrapper.py`**

```python
class BaseWrapper:
    def __init__(self, model, preprocessor, postprocessor, optimizer=None, scheduler=None): ...
    def set_scheduler(self, scheduler): ...         # 단순 대입, 기본 생성은 각 wrapper 책임(optimizer와 대칭)
    def compute_losses(self, raw_output, targets): ...  # $\to$ 배치별 {name: loss tensor}
    def compute_metrics(self, raw_output, targets): ... # 후처리 후 metric 누적
    def get_loss_results(self): ...                 # $\to$ 누적 loss 평균 dict
    def get_metric_results(self): ...               # $\to$ 누적 metric 평균 dict
    def train_step(self, images, targets): ...     # 공통 학습 step, $\to$ dict(loss/메트릭 누적 평균)
    def eval_step(self, images, targets): ...      # 공통 평가 step, $\to$ dict(loss/메트릭 누적 평균)
    def predict_step(self, images): ...            # $\to$ preds: (N, 4, 2)
```

`direct`와 `heatmap`은 기본 `compute_losses()`를 사용한다. `homography`, `seg`, `hybrid`,
`det`, `torchseg`은 방법론별 `compute_losses()`만 오버라이드하고 공통 step을 사용한다.
`torchdet`만 torchvision detector의 호출 규약 때문에 `train_step`/`eval_step`/`predict_step`을
오버라이드한다. 누적 결과 조회는 `get_loss_results()`/`get_metric_results()`로 분리한다.
각 방법론 Wrapper는 같은 폴더의 preprocessor와 postprocessor를 생성해 `BaseWrapper`에 필수
인자로 전달하며, 생성 이후에는 교체하지 않는다.

### 6.6 `src/models/<name>`

각 방법론 폴더는 `base/`의 4종 클래스(`BaseModel`/`BasePreprocessor`/`BasePostprocessor`/
`BaseWrapper`)를 직접 상속하고, 다른 방법론 폴더를 import하거나 상속하지 않는다.
`wrapper.py`는 생성자에서 같은 폴더의 `model`/`preprocessor`/`postprocessor`(+옵티마이저)와
`src/losses`의 loss를 직접 구성해 감싼다. preprocessor와 postprocessor는 선택적으로 주입하거나
런타임에 교체하지 않는다.

**예: `direct/`** (Method A)

```python
# model.py
class DirectModel(BaseModel):
    def __init__(self, backbone="resnet18", pretrained=True): ...
    def forward(self, images): ...   # $\to$ (N, 8) raw logits

# preprocessor.py
class DirectPreprocessor(BasePreprocessor):
    def __call__(self, corners): ...   # (N,4,2) $\to$ (N,8) 정규화 좌표

# postprocessor.py
class DirectPostprocessor(BasePostprocessor):
    def __call__(self, raw_output): ...   # (N,8) $\to$ sigmoid $\to$ (N,4,2)

# wrapper.py
class DirectWrapper(BaseWrapper):
    def __init__(self, backbone="resnet18", optimizer=None, preprocessor=None,
                 postprocessor=None, losses=None, metrics=None, device=None): ...
    # super() 앞에서 preprocessor = preprocessor or DirectPreprocessor() 형태로 지정(주입 허용),
    # 내부에서 DirectModel/Adam optimizer 구성, losses={"loss": WingLoss(apply_sigmoid=True)} (src/losses)
```

**나머지 방법론**은 동일한 4개 파일 패턴을 따르되 내용만 다르다 (6절 상단 표 참조).
loss는 각 wrapper가 `src/losses`의 재사용 클래스를 `losses` dict로 조합해 쓴다
(MSE/BCE/Dice 등은 `src/losses`에 필요 시점에 추가).
- `heatmap/`: `HeatmapModel`(deconv head), `HeatmapPreprocessor`(corners$\to$gaussian heatmap),
  `HeatmapPostprocessor`(soft-argmax), `HeatmapWrapper`(losses: heatmap MSE)
- `hybrid/`: `HybridModel`(MobileNetV3-UNet), `HybridPreprocessor`(corners$\to$mask),
  `HybridPostprocessor`(Canny+Hough+cornerSubPix), `HybridWrapper`(losses: BCE/Dice)
- `line/`: `LineModel`(M-LSD 또는 Canny/Hough 래퍼), `LinePreprocessor`(corners$\to$직선 타깃),
  `LinePostprocessor`(직선 그룹화+교점 계산, 직선 교점 연산은 이 폴더 내부에 둠), `LineWrapper`(losses: 직선 표현)
- `doc/`: `DocModel`(DocTr/DocScanner 파인튜닝 + 어댑터), `DocPreprocessor`(direct와 동일 방식),
  `DocPostprocessor`(direct와 동일 방식), `DocWrapper`(losses: direct와 동일)

### 6.7 `src/core`

**`factory.py`**

```python
def get_transform(split, image_size=512): ...
def get_dataset(split, csv_path, image_size=512, has_corners=True): ...
# has_corners=True $\to$ CornerDataset, False $\to$ ImageDataset
def get_dataloader(split, csv_path, image_size=512, batch_size=16, seed=42, has_corners=True): ...
def get_samples(split, csv_path, image_size=512, indices=None, num_samples=None,
                shuffle=False, seed=42, has_corners=True): ...
# $\to$ has_corners=True: (images, corners) 스택된 텐서 튜플, False: images 스택된 텐서 하나
def get_wrapper(method, device=None, **kwargs): ...   # method: direct|heatmap|hybrid|line|doc
def get_logger(name, output_dir=None): ...   # 터미널 + output_dir 지정 시 run.log 파일
```

**`trainer.py`**

```python
def format_result(result): ...

class Trainer:
    def __init__(self, wrapper, output_dir=None): ...
    def train(self, dataloader): ...
    def evaluate(self, dataloader): ...
    def fit(self, train_loader, valid_loader=None, max_epochs=10): ...
    def fit_early_stop(self, train_loader, valid_loader, max_epochs=100, patience=10): ...
    def save(self, history, output_dir=None): ...
```

**`evaluator.py`**

```python
class Evaluator:
    def __init__(self, wrapper, output_dir=None): ...
    def evaluate(self, dataloader): ...   # $\to$ dict(iou=..., mcd=..., max_cd=..., reproj_error=...)
    def save(self, result, output_dir=None): ...
```

**`predictor.py`**

```python
class Predictor:
    def __init__(self, wrapper, output_dir=None): ...
    def predict(self, dataloader): ...    # $\to$ pred_corners.csv 형식 DataFrame
    def save(self, preds, output_dir=None): ...
```

## 7. `scripts`

```text
scripts/
├── config.py
├── create_data.py
├── evaluate.py
├── fix_data.py
├── predict.py
└── train.py
```

### 7.1 `config.py`

```python
DEFAULTS = {
    "data_dir": "/mnt/d/datasets/roi-corner",
    "csv_path": ["data/smartdoc/gt_corners.csv", "data/midv2020/gt_corners.csv"],
    "seed": 42,
    "method": "direct",
    "image_size": 224,
    "batch_size": 4,
    "max_epochs": 50,
    "patience": 10,
    "num_workers": 4,
    "train_size": 20000,
    "valid_size": 1000,
    "test_size": 1000,
}
def get_experiment(cfg): ...   # $\to$ "{method}_bs{batch_size}_ep{max_epochs}"
def get_output_dir(cfg): ...   # $\to$ "outputs/{method}/{exp_name}"
def parse_args(): ...          # 모든 스크립트가 공유하는 ArgumentParser
```

### 7.2 인수 (`parse_args`)

| 인수 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `--method` | str | `direct` | 방법론 코드 (`direct`, `seg`, `det`, ...) |
| `--device` | str | `None` | 생략 시 자동선택, `cpu`/`cuda` 강제 지정 가능 |
| `--batch_size` | int | `4` | 배치 크기 |
| `--max_epochs` | int | `50` | 학습 에폭 수 (early stopping 시 상한) |
| `--patience` | int | `10` | early stopping patience (valid iou 미개선 허용 epoch 수, `0` 이하이면 비활성) |
| `--num_workers` | int | `4` | DataLoader 워커 프로세스 수 (train 기준) |
| `--train_size` | int | `20000` | train 표본 수로 서브샘플링 (`None`이면 전체) |
| `--valid_size` | int | `1000` | valid 표본 수로 서브샘플링 (`None`이면 전체) |
| `--test_size` | int | `1000` | test 표본 수로 서브샘플링 (`None`이면 전체) |
| `--save` | flag | `False` | 결과 저장 여부 |
| `--checkpoint` | str | `None` | 체크포인트 경로 (None 이면 자동 계산) |
| `--output_dir` | str | `None` | 결과 저장 경로 (None 이면 `outputs/{method}/{exp_name}`) |

> `data_dir`, `csv_path`, `seed`, `image_size`는 CLI 인수가 아닌 `DEFAULTS`에서 직접 읽는다.

### 7.3 스크립트별 사용법

```
python scripts/create_data.py --dataset smartdoc --data_dir <raw_dir> --output_path data/smartdoc/gt_corners.csv
python scripts/fix_data.py data/smartdoc/gt_corners.csv data/midv2020/gt_corners.csv
python scripts/train.py --method direct --max_epochs 50
python scripts/evaluate.py --method direct --checkpoint outputs/direct/direct_bs16_ep50/model.pth
python scripts/predict.py --method direct --checkpoint outputs/direct/direct_bs16_ep50/model.pth
```

## 8. `experiments`

```text
experiments/
├── benchmark.py
├── configs.py
└── run.py
```

`experiments/configs.py`의 `CONFIGS` 리스트에 정의된 방법론/하이퍼파라미터 조합을 순서대로
`run.py`가 `scripts/`를 subprocess로 호출하여 일괄 실행한다. 같은 `CONFIGS`를
`benchmark.py`도 참조하여 학습된 체크포인트를 동일 테스트셋으로 비교한다.

```
python experiments/run.py --mode train
python experiments/run.py --mode evaluate
python experiments/run.py --mode predict
python experiments/run.py --mode all
python experiments/benchmark.py   # configs.py CONFIGS 평가 $\to$ outputs/comparison/results.csv
```

## 9. 참조 프로젝트

이 프로젝트는 **레거시 프로젝트**(`../../99_deprecated/260622_roi-direct-regression`)의
데이터 파이프라인 설계와 `docs/plans/roi-detection-calm-papert.md`의 통합 계획, 그리고
**MNIST 프로젝트**(`../260626_mnist-from-scratch`)의 구조/문서 컨벤션을 참조하여 작성한다.
두 프로젝트 명칭과 세부 참조 범위는 [CLAUDE.md](./CLAUDE.md) 7절을 참조한다.
단계별 작업 계획은 [PLAN.md](./PLAN.md)를 참조한다.
