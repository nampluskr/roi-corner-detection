# ROI Corner Detection

평면 사각형 객체(문서, ID 카드, 검사 대상 패널 등)의 4개 코너 좌표를 검출하는 여러 방법론을
하나의 공유 데이터/평가 파이프라인 위에서 구현하고 비교하는 프로젝트이다.

## 1. 방법론

| 코드 | 방법론 | 우선순위 | 핵심 아이디어 | 추가 의존성 | 복잡도 |
|---|---|---:|---|---|---|
| `direct` | Direct Coordinate Regression | 1 | CNN backbone -> FC(8), Sigmoid + SmoothL1/Wing Loss | 없음 | 낮음 |
| `heatmap` | Heatmap Keypoint Detection | 2 | 4채널 heatmap + soft-argmax, 서브픽셀 정밀도 | 없음 | 중간 |
| `hybrid` | DL + Classical CV Hybrid | 3 | Segmentation(MobileNetV3-UNet) + Canny/Hough/cornerSubPix | OpenCV(기존 사용) | 중간 |
| `line` | Line Intersection | 4 | 직선 검출(M-LSD 또는 Canny+Hough) + 교점 계산 | M-LSD 가중치(외부) | 중간 |
| `doc` | Document Pretrained | 5 | DocTr/DocScanner 파인튜닝, few-shot 적응 | DocTr 패키지(외부) | 높음 |

모든 방법론은 `src/models/base/`의 5종 추상 클래스(`BaseModel`, `BasePreprocessor`,
`BasePostprocessor`, `BaseLoss`, `BaseWrapper`)를 구현하고, 공유 `Trainer`/`Evaluator`/
`Predictor`(`src/core/`)와 공통 메트릭(`src/metrics/`)을 그대로 사용한다. 방법론 간 성능 비교는
`scripts/benchmark.py`가 담당한다.

## 2. 핵심 제약 (F1-F5)

| 제약 | 내용 | 설계 영향 |
|---|---|---|
| F1 | 임의의 볼록 사각형 (OBB 아님) | 8개 자유 좌표 직접 회귀/검출 |
| F2 | 단일 객체, 이미지의 50% 이상 차지 | 별도 detection stage 불필요 |
| F3 | 4개 코너는 항상 이미지 경계 내부 | 회전 증강 +-5도 제한, clipping 검증 필수 |
| F4 | 실측 데이터 적음, 합성 데이터 다수 | 3단계 학습 전략 (공개 데이터 -> 합성 -> 실측 파인튜닝) |
| F5 | 서브픽셀 정밀도가 중요 (위상 복원용) | Reprojection Error를 IoU와 함께 필수 메트릭으로 추적 |

## 3. 데이터 전략 (3단계)

1. **Pre-training**: SmartDoc + MIDV-2020 (공개 데이터, 일반 코너 검출 학습)
2. **Domain Adaptation**: 합성 Fringe 패턴 이미지 (검사 도메인 적응, `src/data/synthetic.py`로 생성)
3. **Fine-tuning**: 실측 PMD 데이터 (소량, 최종 정밀도 확보)

## 4. 평가 메트릭

| 메트릭 | 설명 |
|---|---|
| Polygon IoU | 예측/정답 사각형의 면적 기반 중첩도 |
| MCD (Mean Corner Distance) | 4개 코너 평균 유클리드 거리 |
| MaxCD | 최대 코너 오차 (worst-case) |
| Reprojection Error | 호모그래피 역투영 오차 (위상 복원 품질과 직결, 핵심 지표) |
| GPU/CPU latency, 모델 크기 | 배포 적합성 비교 (`scripts/benchmark.py`) |

모든 메트릭은 `src/metrics/metrics.py`에 위치하며, 각 방법론의 `postprocessor.py`가 만든
표준 코너 좌표(`(N, 4, 2)`)에 대해 방법론 이름과 무관하게 동일하게 적용된다.

## 5. 폴더 구조

```text
roi-corner-detection/
├── data/
├── docs/
├── experiments/
│   └── run.py
├── notebooks/
│   ├── data/
│   └── results/
├── outputs/
│   ├── direct/
│   ├── heatmap/
│   ├── hybrid/
│   ├── line/
│   ├── doc/
│   └── comparison/
├── scripts/
│   ├── benchmark.py
│   ├── config.py
│   ├── evaluate.py
│   ├── fix_corner_order.py
│   ├── predict.py
│   ├── prepare_data.py
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
| `docs/` | 방법론 카탈로그, 데이터셋 설계, F1-F5 제약 등 설계 문서 |
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
│   ├── labelme.py
│   ├── midv2020.py
│   ├── smartdoc.py
│   ├── synthetic.py
│   └── transforms.py
├── metrics/
│   └── metrics.py
├── models/
│   ├── base/
│   │   ├── base_loss.py
│   │   ├── base_model.py
│   │   ├── base_postprocessor.py
│   │   ├── base_preprocessor.py
│   │   └── base_wrapper.py
│   ├── direct/
│   │   ├── loss.py
│   │   ├── model.py
│   │   ├── postprocessor.py
│   │   ├── preprocessor.py
│   │   └── wrapper.py
│   ├── doc/        (동일 5개 파일: loss.py, model.py, postprocessor.py, preprocessor.py, wrapper.py)
│   ├── heatmap/    (동일 5개 파일)
│   ├── hybrid/     (동일 5개 파일)
│   └── line/       (동일 5개 파일)
└── utils/
    ├── geometry.py
    └── homography.py
```

각 방법론 폴더(`direct/doc/heatmap/hybrid/line`)는 항상 **같은 이름의 파일 5개**를 갖고,
내용만 방법론별로 다르다. 방법론 간 raw 출력 형태 차이는 다음과 같다.

| 방법론 | raw 출력 (`model.py`) | 학습 타깃 (`preprocessor.py`) | loss 대상 (`loss.py`) | 후처리 (`postprocessor.py`) |
|---|---|---|---|---|
| `direct` | (N, 8) 좌표 logits | 정규화 좌표 그대로 | logits vs 좌표 타깃 | sigmoid + reshape (N,4,2) |
| `heatmap` | (N, 4, H, W) heatmap | corners -> gaussian heatmap | heatmap vs heatmap 타깃 | soft-argmax -> (N,4,2) |
| `hybrid` | (N, 1, H, W) 세그멘테이션 마스크 | corners -> 마스크 | mask vs 마스크 타깃 | Canny+Hough+cornerSubPix -> (N,4,2) |
| `line` | 직선 세그먼트 raw 출력 | corners -> 직선/엣지 타깃 | 직선 표현 loss | 직선 그룹화 + 교점 계산 -> (N,4,2) |
| `doc` | (N, 8) 좌표 logits (사전학습 기반) | 정규화 좌표 그대로 | logits vs 좌표 타깃 | sigmoid + reshape (N,4,2) |

### 6.1 `src/utils`

**`geometry.py`**

```python
def order_corners(corners): ...        # -> TL, TR, BR, BL 순서로 정렬된 (4, 2) 배열
def mask_to_corners(mask): ...         # 이진 마스크 -> (4, 2) 코너 좌표 (contour 기반)
def corners_to_mask(corners, shape): ...  # (4, 2) 코너 좌표 -> 이진 마스크
def polygon_area(corners): ...         # Shoelace 공식 면적 계산
def clip_corners_to_bounds(corners, width, height): ...  # 경계 내부 검증/클리핑 (F3 제약)
```

**`homography.py`**

```python
def estimate_homography(src_corners, dst_corners): ...  # 코너 4쌍 -> 호모그래피 행렬
def reproject_points(points, homography): ...            # 호모그래피로 점 역투영
```

### 6.2 `src/data`

**`dataset.py`**

```python
class CornerDataset:
    def __init__(self, csv_path, transform=None): ...
    def split(self, split_ratio=0.8, seed=42): ...   # -> (train_dataset, valid_dataset)
    def __len__(self): ...
    def __getitem__(self, idx): ...   # -> (image, corners)  corners: (4, 2) normalized [0, 1]
```

**`dataloader.py`**

```python
class Dataloader:
    def __init__(self, split, dataset, batch_size=16, seed=42): ...
    def __len__(self): ...
    def __iter__(self): ...
```

**`transforms.py`**

```python
class Compose:
    def __init__(self, transforms): ...
    def __call__(self, image, corners): ...

# 기하 변환 - (image, corners) 동시 변환, 코너 순서 보존
class Resize: ...
class RandomHorizontalFlip: ...    # 코너 재정렬 포함
class RandomVerticalFlip: ...      # 코너 재정렬 포함
class RandomRotation: ...          # 최대 +-5도, clipping 검증 (F3)
class RandomPerspective: ...
class RandomScale: ...

# 이미지 전용 변환
class ColorJitter: ...
class GaussianBlur: ...
class GaussianNoise: ...
class ToTensor: ...
class Normalize: ...
```

**`labelme.py` / `smartdoc.py` / `midv2020.py`**

```python
def parse_labelme(json_dir, image_dir): ...   # -> gt_corners.csv 형식 DataFrame
def prepare_smartdoc(raw_dir): ...             # -> gt_corners.csv 형식 DataFrame
def prepare_midv2020(raw_dir): ...             # -> gt_corners.csv 형식 DataFrame (mask -> corners)
```

> **CSV 스키마 (공통)**: `image_dir,image_name,x1,y1,x2,y2,x3,y3,x4,y4` (정규화 [0,1], 소수점 6자리,
> TL->TR->BR->BL 시계방향)

**`synthetic.py`**

```python
def generate_fringe_image(A, B, f, phi, direction="horizontal"): ...
# I(x, y) = A + B * cos(2 * pi * f * x + phi), Domain Adaptation 단계용 (F4 제약)
```

### 6.3 `src/metrics`

**`metrics.py`**

```python
def polygon_iou(pred_corners, gt_corners): ...   # (4,2), (4,2) -> float, utils.geometry.polygon_area 사용
def mcd(pred_corners, gt_corners): ...           # -> float, Mean Corner Distance
def max_cd(pred_corners, gt_corners): ...        # -> float, Max Corner Distance
def reprojection_error(pred_corners, gt_corners, ref_corners): ...
# ref_corners: 정규 사각형 기준점. utils.homography 사용
```

### 6.4 `src/models/base`

**`base_model.py`**

```python
class BaseModel(nn.Module):
    def forward(self, images): ...   # images: (N, C, H, W) -> raw 출력 (방법론별 형태 자유)
```

**`base_preprocessor.py`**

```python
class BasePreprocessor:
    def __call__(self, corners): ...   # corners: (N, 4, 2) -> 학습 타깃 (방법론별 형태)
```

**`base_postprocessor.py`**

```python
class BasePostprocessor:
    def __call__(self, raw_output): ...   # raw_output -> corners: (N, 4, 2) 표준 포맷
```

**`base_loss.py`**

```python
class BaseLoss:
    def __call__(self, raw_output, target): ...   # -> scalar loss
```

**`base_wrapper.py`**

```python
class BaseWrapper:
    def __init__(self, model, optimizer=None): ...
    def train_step(self, images, corners): ...    # -> dict(loss=...)
    def eval_step(self, images, corners): ...      # -> dict(loss=..., corners_pred=(N,4,2))
    def predict_step(self, images): ...            # -> corners_pred: (N, 4, 2)
```

### 6.5 `src/models/<name>`

각 방법론 폴더는 `base/`의 5종 클래스를 상속한다. `wrapper.py`는 생성자에서 같은 폴더의
`model`/`preprocessor`/`postprocessor`/`loss`(+옵티마이저)를 직접 구성해 감싼다.

**예: `direct/`** (Method A)

```python
# model.py
class DirectModel(BaseModel):
    def __init__(self, backbone="resnet18", pretrained=True): ...
    def forward(self, images): ...   # -> (N, 8) raw logits

# preprocessor.py
class DirectPreprocessor(BasePreprocessor):
    def __call__(self, corners): ...   # (N,4,2) -> (N,8) 정규화 좌표

# postprocessor.py
class DirectPostprocessor(BasePostprocessor):
    def __call__(self, raw_output): ...   # (N,8) -> sigmoid -> (N,4,2)

# loss.py
class DirectLoss(BaseLoss):
    def __call__(self, raw_output, target): ...   # sigmoid + SmoothL1 또는 Wing Loss

# wrapper.py
class DirectWrapper(BaseWrapper):
    def __init__(self, backbone="resnet18", lr=1e-3): ...
    # 내부에서 DirectModel/DirectPreprocessor/DirectPostprocessor/DirectLoss/Adam optimizer 구성
```

**나머지 방법론**은 동일한 5개 파일 패턴을 따르되 내용만 다르다 (6절 상단 표 참조).
- `heatmap/`: `HeatmapModel`(deconv head), `HeatmapPreprocessor`(corners->gaussian heatmap),
  `HeatmapPostprocessor`(soft-argmax), `HeatmapLoss`(heatmap MSE), `HeatmapWrapper`
- `hybrid/`: `HybridModel`(MobileNetV3-UNet), `HybridPreprocessor`(corners->mask),
  `HybridPostprocessor`(Canny+Hough+cornerSubPix), `HybridLoss`(BCE/Dice), `HybridWrapper`
- `line/`: `LineModel`(M-LSD 또는 Canny/Hough 래퍼), `LinePreprocessor`(corners->직선 타깃),
  `LinePostprocessor`(직선 그룹화+교점 계산, 직선 교점 연산은 이 폴더 내부에 둠), `LineLoss`, `LineWrapper`
- `doc/`: `DocModel`(DocTr/DocScanner 파인튜닝 + 어댑터), `DocPreprocessor`(direct와 동일 방식),
  `DocPostprocessor`(direct와 동일 방식), `DocLoss`, `DocWrapper`

### 6.6 `src/core`

**`factory.py`**

```python
def get_transform(split, input_size=512): ...
def get_dataset(split, csv_path, input_size=512): ...
def get_dataloader(split, csv_path, input_size=512, batch_size=16, seed=42): ...
def get_wrapper(method, device=None, **kwargs): ...   # method: direct|heatmap|hybrid|line|doc
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
    def evaluate(self, dataloader): ...   # -> dict(iou=..., mcd=..., max_cd=..., reproj_error=...)
    def save(self, result, output_dir=None): ...
```

**`predictor.py`**

```python
class Predictor:
    def __init__(self, wrapper, output_dir=None): ...
    def predict(self, dataloader): ...    # -> pred_corners.csv 형식 DataFrame
    def save(self, preds, output_dir=None): ...
```

## 7. `scripts`

```text
scripts/
├── benchmark.py
├── config.py
├── evaluate.py
├── fix_corner_order.py
├── predict.py
├── prepare_data.py
└── train.py
```

### 7.1 `config.py`

```python
DEFAULTS = {
    "data_dir": "/mnt/d/datasets/roi-corner",
    "output_dir": "outputs",
    "seed": 42,
    "method": "direct",
    "input_size": 512,
    "batch_size": 16,
    "max_epochs": 50,
}
def get_experiment(cfg): ...   # -> "{method}_bs{batch_size}_ep{max_epochs}"
def parse_args(): ...          # 모든 스크립트가 공유하는 ArgumentParser
```

### 7.2 인수 (`parse_args`)

| 인수 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `--method` | str | `direct` | `direct`, `heatmap`, `hybrid`, `line`, `doc` |
| `--device` | str | `None` | 생략 시 자동선택, `cpu`/`gpu` 강제 지정 가능 |
| `--batch_size` | int | `16` | 배치 크기 |
| `--max_epochs` | int | `50` | 학습 에폭 수 |
| `--save` | flag | `False` | 결과 저장 여부 |
| `--checkpoint` | str | `None` | 체크포인트 경로 (None 이면 자동 계산) |
| `--output_dir` | str | `None` | 결과 저장 경로 |

> `data_dir`, `seed`, `input_size`는 CLI 인수가 아닌 `DEFAULTS`에서 직접 읽는다.

### 7.3 스크립트별 사용법

```
python scripts/prepare_data.py --source smartdoc --raw_dir <path>
python scripts/train.py --method direct --max_epochs 50
python scripts/evaluate.py --method direct --checkpoint outputs/direct/model.pth
python scripts/predict.py --method direct --checkpoint outputs/direct/model.pth
python scripts/benchmark.py   # 전체 방법론 평가 -> outputs/comparison/results.csv
```

## 8. `experiments`

```text
experiments/
└── run.py
```

`CONFIGS` 리스트에 정의된 방법론/하이퍼파라미터 조합을 순서대로 `scripts/`를 subprocess로
호출하여 일괄 실행한다.

```
python experiments/run.py --mode train
python experiments/run.py --mode evaluate
python experiments/run.py --mode predict
python experiments/run.py --mode all
```

## 9. 참조 프로젝트

이 프로젝트는 **레거시 프로젝트**(`../../99_deprecated/260622_roi-direct-regression`)의
데이터 파이프라인 설계와 `docs/plans/roi-detection-calm-papert.md`의 통합 계획, 그리고
**MNIST 프로젝트**(`../260626_mnist-from-scratch`)의 구조/문서 컨벤션을 참조하여 작성한다.
두 프로젝트 명칭과 세부 참조 범위는 [CLAUDE.md](./CLAUDE.md) 7절을 참조한다.
단계별 작업 계획은 [PLAN.md](./PLAN.md)를 참조한다.
