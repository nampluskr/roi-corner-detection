# PLAN.md

이 문서는 `roi-corner-detection` 프로젝트를 구축하기 위한 작업 계획이다.

## 1. Context

이 프로젝트는 아래 두 프로젝트만 참조하여 작성한다.

- **레거시 프로젝트** (`../../99_deprecated/260622_roi-direct-regression`): Direct
  Regression(방법 A) 단일 베이스라인으로 시작했으나 Stage 3 이후 구현이 멈춘 채 deprecated
  되었다. 다만 그 프로젝트 안에는:
  - **Stage 1-2 데이터 파이프라인이 완성**되어 있음
    (`src/utils/geometry.py`, `src/data/{dataset,transforms,labelme,smartdoc,midv2020}.py`)
  - **`docs/plans/roi-detection-calm-papert.md`**: 방법론 카탈로그(A~J,
    `_project/sources/P3-methodology-catalog.md`) 중 상위 5개
    (A.Direct, B.Heatmap, I.Hybrid, D.Line, J.Doc-pretrained)를 통합하는 상세 설계가 이미 존재
- **MNIST 프로젝트** (`../260626_mnist-from-scratch`): numpy/pytorch 두 구현을 공통 구조·
  SSOT README로 통합한 선례. 폴더 구조, SSOT README/CLAUDE.md 작성 방식, 코딩 규칙의 출처

이번 프로젝트(`roi-corner-detection`)는 MNIST 프로젝트와 같은 운영 방식으로, 하나의 PyTorch
프로젝트 안에서 여러 코너 검출 "방법론"을 통합하여 평가/비교한다. 방법론 축 통합이라는 점에서
MNIST 프로젝트의 "프레임워크 축 통합"과는 다르지만, 레거시 프로젝트의
`roi-detection-calm-papert.md`가 정확히 이 요구를 겨냥한 설계이므로 이를 채택하고 MNIST
프로젝트의 운영 컨벤션(SSOT README, CLAUDE.md 동기화/코딩 규칙, "참조만 하고 재구현" 원칙,
자체 git 저장소)을 덧입힌다.

## 2. 확정된 결정 사항

1. **방법론 범위**: 상위 5개(A.Direct, B.Heatmap, I.Hybrid, D.Line, J.Doc-pretrained) 채택
2. **git 저장소**: `roi-corner-detection`는 워크스페이스와 별도로 자체 git 저장소로 init
   (GitHub remote 연결은 사용자가 추후 직접 진행)
3. **레거시 코드 재사용 방식**: 레거시 프로젝트의 Stage1-2 코드는 참조만 하고 완전 재구현
   (MNIST 프로젝트와 동일 원칙, 복사/심링크 금지)
4. **루트 폴더**: `configs/` 없음. `data/`(정답/예측 CSV, git-ignore), `docs/`(설계 문서) 추가.
   나머지(`experiments/`, `notebooks/`, `outputs/`, `scripts/`, `src/`)는 MNIST 프로젝트와 동일.
5. **방법론과 wrapper 관계**: task 축이 "코너 좌표 검출" 하나뿐이라 wrapper를 top-level
   폴더로 분리하지 않는다. 각 `src/models/<name>/`가 `model.py`, `preprocessor.py`,
   `postprocessor.py`, `loss.py`, `wrapper.py` 5개 파일(파일명 통일, 내용은 방법론별)을 갖고,
   공통 추상 인터페이스는 `src/models/base/`에 모은다 (`BaseModel`, `BasePreprocessor`,
   `BasePostprocessor`, `BaseLoss`, `BaseWrapper`).
6. **메트릭 배치**: `src/metrics/`를 `core/`와 별개인 top-level 폴더로 둔다.
   내부는 `metrics.py` 단일 파일 (대부분 몇 줄짜리 순수 함수라 폴더 분리는 과함).
7. **유틸 배치**: `src/utils/`는 `geometry.py`(코너/폴리곤 순수 연산), `homography.py`
   (호모그래피 추정/역투영) 2개 파일. 방법론 하나만 쓰는 보조 연산(예: `line`의 직선 교점
   계산)은 승격하지 않고 해당 방법론 폴더 안에 둔다.

## 3. 목표 디렉토리 구조

레거시 프로젝트의 `roi-detection-calm-papert.md` 구조를 채택하고 MNIST 프로젝트 컨벤션에 맞춘
최종 구조. 상세 시그니처는 [README.md](./README.md)를 SSOT로 삼는다.

```text
roi-corner-detection/
├── .gitignore
├── CLAUDE.md
├── PLAN.md
├── README.md
├── data/
├── docs/
├── experiments/
│   └── run.py
├── notebooks/
│   ├── data/
│   └── results/
├── outputs/
│   ├── direct/ heatmap/ hybrid/ line/ doc/
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
    │   ├── direct/    (loss.py, model.py, postprocessor.py, preprocessor.py, wrapper.py)
    │   ├── doc/       (동일 5개 파일)
    │   ├── heatmap/   (동일 5개 파일)
    │   ├── hybrid/    (동일 5개 파일)
    │   └── line/      (동일 5개 파일)
    └── utils/
        ├── geometry.py
        └── homography.py
```

## 4. 실행 단계

### Phase 0 - 프로젝트 스캐폴딩 (진행 중)

- 0-1단계 (완료): 루트 폴더 구조 확정
- 0-2단계 (완료): `src/` 하위 폴더 구조 확정 (core/data/metrics/models/utils, models 하위 base+5방법론)
- 0-3단계 (완료): `src/utils/`, `src/data/` 파일 리스트 확정, `src/models/base/`,
  `src/models/<name>/`의 5파일 패턴 및 `src/metrics/metrics.py` 확정
- 0-4단계 (이번 턴 완료): 위 결정 사항을 `README.md`(SSOT)/`CLAUDE.md`/`PLAN.md` 3개 파일에 반영
- **다음 턴 (사용자 진행 예정)**: 사용자가 `README.md`의 시그니처(함수/클래스 인터페이스)를
  다시 검토. 검토 완료 후 `git init`, `.gitignore` 작성, 실제 디렉토리/파일 스켈레톤 생성 진행

### Phase 1 - 공유 데이터/유틸 재구현

참조: `<레거시 프로젝트>/src/{utils,data}` (복사 금지, 재구현)

- `src/utils/geometry.py`: `order_corners`, `mask_to_corners`, `corners_to_mask`,
  `polygon_area`, `clip_corners_to_bounds`
- `src/utils/homography.py`: `estimate_homography`, `reproject_points`
- `src/data/dataset.py`: `CornerDataset` (csv_path, transform / `.split(ratio, seed)`)
- `src/data/dataloader.py`: `Dataloader` (`torch.utils.data.DataLoader` 래퍼)
- `src/data/transforms.py`: 기하 변환(Resize, Flip+코너 재정렬, Rotation +-5도 clip 검증) + 광학 변환
- `src/data/{labelme,smartdoc,midv2020}.py`: 데이터셋 파서
- `src/data/synthetic.py`: 합성 Fringe 패턴 생성 (Domain Adaptation 단계, 실제 사용은 후순위)
- `src/metrics/metrics.py`: `polygon_iou`, `mcd`, `max_cd`, `reprojection_error`
- `scripts/prepare_data.py`, `scripts/fix_corner_order.py`

### Phase 2 - 공통 인터페이스 + Method A(Direct) 전체 파이프라인

- `src/models/base/`: `BaseModel`, `BasePreprocessor`, `BasePostprocessor`, `BaseLoss`,
  `BaseWrapper` 5종 추상 클래스
- `src/models/direct/{model.py, preprocessor.py, postprocessor.py, loss.py, wrapper.py}`:
  ResNet18/34/50 + FC head, Sigmoid+SmoothL1/Wing Loss
  (참조: `<레거시 프로젝트>/_legacy/quad_regression/src/models/losses.py`)
- `src/core/{factory.py, trainer.py, evaluator.py, predictor.py}` 구현
  (참조: `<레거시 프로젝트>/_legacy/quad_regression/src/engine.py`의 fit/fit_early_stop 패턴,
  MNIST 프로젝트 `src/core/trainer.py`의 `Trainer(wrapper, output_dir=None)` 시그니처)
- `scripts/{config.py, train.py, evaluate.py, predict.py}` 구현 (`--method` 인수로 방법론 선택)
- **검증**: `python -c "from src.models.direct.model import DirectModel; ..."` shape 확인,
  `python scripts/train.py --method direct --max_epochs 1` 로 1 epoch 학습 스모크 테스트

### Phase 3 - 나머지 방법론 순차 구현 (Method A 검증 후 별도 세션 권장)

우선순위 순: B.Heatmap -> I.Hybrid -> D.Line -> J.Doc-pretrained
각 방법론은 `src/models/<name>/`에 5개 파일(model/preprocessor/postprocessor/loss/wrapper)만
추가하고 `src/models/base/`, `src/core/`, `src/metrics/`, `src/utils/`를 그대로 재사용한다.

### Phase 4 - 비교 프레임워크

- `scripts/benchmark.py`: 전체 방법론 학습된 체크포인트로 동일 테스트셋 평가,
  `outputs/comparison/results.csv` (Polygon IoU, MCD, Reprojection Error, GPU/CPU latency, 모델 크기) 생성
- `notebooks/results/compare_methods.ipynb`: 방법론별 성능/속도 비교 표, 실패 케이스 시각화

## 5. 이번 세션 실행 범위

Phase 0의 폴더/파일 구조 확정(0-1 ~ 0-4단계)까지 진행했다. 실제 디렉토리/파일 생성은
사용자가 `README.md`의 시그니처를 재검토한 뒤 진행한다. Phase 1-2(공유 코드 + Method A
전체 파이프라인)는 그 다음 세션에서 진행하고, Phase 3(나머지 4개 방법론)과 Phase 4
(비교 스크립트)는 이후 세션에서 이어간다.

## 6. 검증 방법

1. `python -c "from src.models.direct.model import DirectModel; import torch; m = DirectModel(); print(m(torch.randn(2,3,512,512)).shape)"` -> `torch.Size([2, 8])`
2. `python scripts/prepare_data.py ...` 로 `gt_corners.csv` 생성 확인 (실제 이미지 데이터가 없다면 스킵하고 코드 정합성만 확인)
3. `python scripts/train.py --method direct --max_epochs 1` 스모크 테스트 (데이터 없으면 더미 텐서 기반 유닛 테스트로 대체)
4. `git log --oneline` 로 Phase별 커밋 확인
