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

1. **방법론 범위**: 총 10개 모델 채택. 우선순위 1-3은 고정 - 1.direct(A), 2.seg(C,
   Segmentation Corner를 hybrid에서 독립 승격), 3.detect(신규 설계, 바운딩박스 기반
   keypoint detector. 카탈로그 G의 DETR 방식이 아님). 4-10은 레거시 카탈로그의 나머지
   7개(B, I, D, J, E, H, F)를 프로젝트 적합도 순으로 재정렬 - 4.heatmap(B), 5.hybrid(I),
   6.line(D), 7.doc(J), 8.homography(E), 9.foundation(H), 10.gcn(F).
   상세 비교는 `docs/roi-corner-detection-models.md` 참조
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
   각 메트릭은 `BaseMetric`(`base_metric.py`)을 상속하는 클래스이며, 메트릭당 파일
   하나로 분리한다(`polygon_iou.py`, `mcd.py`, `max_cd.py`, `reprojection_error.py`,
   `pck.py`). `__call__(pred_corners, gt_corners, ...)`로 함수처럼 호출한다.
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
│   ├── detect/ direct/ doc/ foundation/ gcn/ heatmap/ homography/ hybrid/ line/ seg/
│   └── comparison/
├── scripts/
│   ├── benchmark.py
│   ├── config.py
│   ├── create_data.py
│   ├── evaluate.py
│   ├── fix_data.py
│   ├── predict.py
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
    │   ├── base_metric.py
    │   ├── max_cd.py
    │   ├── mcd.py
    │   ├── pck.py
    │   ├── polygon_iou.py
    │   └── reprojection_error.py
    ├── models/
    │   ├── base/
    │   │   ├── base_loss.py
    │   │   ├── base_model.py
    │   │   ├── base_postprocessor.py
    │   │   ├── base_preprocessor.py
    │   │   └── base_wrapper.py
    │   ├── detect/      (loss.py, model.py, postprocessor.py, preprocessor.py, wrapper.py)
    │   ├── direct/      (동일 5개 파일)
    │   ├── doc/         (동일 5개 파일)
    │   ├── foundation/  (동일 5개 파일)
    │   ├── gcn/         (동일 5개 파일)
    │   ├── heatmap/     (동일 5개 파일)
    │   ├── homography/  (동일 5개 파일)
    │   ├── hybrid/      (동일 5개 파일)
    │   ├── line/        (동일 5개 파일)
    │   └── seg/         (동일 5개 파일)
    └── utils/
        ├── geometry.py
        └── homography.py
```

## 4. 실행 단계

### Phase 0 - 프로젝트 스캐폴딩 (완료)

- 0-1단계 (완료): 루트 폴더 구조 확정
- 0-2단계 (완료): `src/` 하위 폴더 구조 확정 (core/data/metrics/models/utils, models 하위 base+5방법론)
- 0-3단계 (완료): `src/utils/`, `src/data/` 파일 리스트 확정, `src/models/base/`,
  `src/models/<name>/`의 5파일 패턴 및 `src/metrics/metrics.py` 확정
- 0-4단계 (완료): 위 결정 사항을 `README.md`(SSOT)/`CLAUDE.md`/`PLAN.md` 3개 파일에 반영
- 0-5단계 (완료): `README.md` 시그니처 재검토 완료

### Phase 1 - 공유 데이터/유틸 재구현 (진행 중)

참조: `<레거시 프로젝트>/src/{utils,data}` (복사 금지, 재구현)

- [x] `src/data/dataset.py`: `Dataset`(`torch.utils.data.Dataset` 상속, `csv_path`
  str/리스트 지원, `transform` 생략 시 `ToTensor()` 기본 적용)을 상속하는
  `CornerDataset`(코너 포함 CSV)/`ImageDataset`(코너 없는 CSV)로 분리, `Subset`
  (`.split(ratio, seed)`/`.subset(num_samples, seed)` 반환, `set_transform` 지원)
- [x] `src/data/dataloader.py`: `Dataloader` (`torch.utils.data.DataLoader` 상속,
  MNIST 프로젝트 패턴 적용: train은 shuffle/drop_last/num_workers=4, 나머지는 미적용)
- [x] `src/data/transforms.py`: `Compose` + 기하 변환(`Resize`, `RandomHorizontalFlip`,
  `RandomVerticalFlip`, `RandomRotation`, `RandomPerspective`, `RandomScale`) + 이미지 전용
  변환(`ColorJitter`, `GaussianBlur`, `ToTensor`, `Normalize`, `GaussianNoise`) 11개 클래스
  전체 구현. `pytorch_env`(torch 2.5.1+cu121/torchvision 0.20.1+cu121)에서 전체 파이프라인
  실행 및 플립 시 코너 순서 보존 검증 완료
- [x] `src/utils/geometry.py`: `order_corners`, `is_invalid_corners`, `mask_to_corners`,
  `polygon_area` 구현 완료 (`corners_to_mask`는 아직 미구현, 필요 시점에 추가)
- [x] `src/utils/homography.py`: `estimate_homography`, `reproject_points` 구현 완료
- [x] `src/data/smartdoc.py`, `src/data/midv2020.py`: `create_data(data_dir, output_path)`로
  raw 데이터를 파싱해 `gt_corners.csv` 저장 (`labelme.py`는 아직 미구현)
- [x] `src/data/images.py`: `create_data(data_dir, output_path)`로 코너 없는 순수 이미지
  폴더를 스캔해 `image_dir,image_name` CSV 저장 (`ImageDataset`용)
- [ ] `src/data/synthetic.py`: 합성 Fringe 패턴 생성 (Domain Adaptation 단계, 실제 사용은 후순위)
- [x] `src/metrics/`: `base_metric.py`(`BaseMetric`)를 상속하는 `polygon_iou.py`(`PolygonIoU`),
  `mcd.py`(`MCD`), `max_cd.py`(`MaxCD`), `reprojection_error.py`(`ReprojectionError`),
  `pck.py`(`PCK`) 클래스 구현 완료. 각각 `__call__(pred_corners, gt_corners, ...)`로 호출
- [x] `scripts/create_data.py`: `--dataset {smartdoc,midv2020,images}` CLI로 `create_data()` 호출
- [x] `scripts/fix_data.py`: `fix_images`(존재하지 않는 이미지 파일 행 제거),
  `fix_corners`(코너 순서 재정렬 + 퇴화 샘플 제거)

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

### Phase 3 - 나머지 방법론 순차 구현 (direct 검증 후 별도 세션 권장)

우선순위 순: seg -> detect -> heatmap -> hybrid -> line -> doc -> homography ->
foundation -> gcn
각 방법론은 `src/models/<name>/`에 5개 파일(model/preprocessor/postprocessor/loss/wrapper)만
추가하고 `src/models/base/`, `src/core/`, `src/metrics/`, `src/utils/`를 그대로 재사용한다.
모든 방법론은 공통 `Dataloader`로부터 입력을 받고, 전용 `preprocessor.py`가 표준 코너
(N, 4, 2)를 방법론별 학습 타깃으로 변환한다.

### Phase 4 - 비교 프레임워크

- `scripts/benchmark.py`: 10개 방법론 전체의 학습된 체크포인트로 동일 테스트셋 평가,
  `outputs/comparison/results.csv` (Polygon IoU, MCD, Reprojection Error, GPU/CPU latency, 모델 크기) 생성
- `notebooks/results/compare_methods.ipynb`: 방법론별 성능/속도 비교 표, 실패 케이스 시각화

## 5. 이번 세션 실행 범위

Phase 0(폴더/파일 구조 확정, README 시그니처 재검토)을 완료했고, Phase 1에서
`src/data/dataset.py`, `dataloader.py`, `transforms.py` 3개 파일 구현과 `pytorch_env`에서의
실행 검증까지 마쳤다. 이어서 `src/utils/geometry.py`(`order_corners`, `is_invalid_corners`,
`mask_to_corners`), `src/data/smartdoc.py`, `src/data/midv2020.py`, `scripts/create_data.py`,
`scripts/fix_data.py`를 구현하고 실제 raw 데이터(`/mnt/d/datasets/smart_doc_extracted`,
`/mnt/d/datasets/midv2020_processed`)로 `data/smartdoc/gt_corners.csv`(24,860행),
`data/midv2020/gt_corners.csv`(68,409행) 생성 및 `fix_data.py` 실행까지 검증했다.
이후 세션에서 `src/utils/homography.py`(`estimate_homography`, `reproject_points`),
`src/data/images.py`(코너 없는 이미지 폴더 스캔), `src/metrics/`(`BaseMetric` 및
`PolygonIoU`/`MCD`/`MaxCD`/`ReprojectionError`/`PCK` 클래스)까지 구현하고 검증을
마쳤다. 이어서 EDA 노트북(`notebooks/eda-{midv2020,smartdoc}-{dataset,transforms}.ipynb`)
작업 중 `src/utils/plot.py`의 `show_samples`를 재작성했다: `get_samples()`/`Dataset`이
반환하는 `(N,3,H,W)` 텐서를 직접 받도록 시그니처를 바꾸고(`load_raw_samples`,
`apply_transform` 제거), `denormalize=False` 인자와 `Polygon(fill=False)` 코너 오버레이,
고정 라벨 오프셋(`OFFSETS`)을 추가했다. 함께 `src/data/transforms.py`에 `ToNumpy`
클래스와 `Normalize`/`Denormalize`의 `IMAGENET_MEAN`/`IMAGENET_STD` 기본값을 추가했고
(`src/core/factory.py`에서는 두 상수를 제거), `src/data/dataset.py`의 `Subset`에 누락되어
있던 `subset()` 메서드(split 이후 재샘플링 시 `AttributeError` 발생 버그)를 추가했다.
남은 Phase 1 항목(`src/data/labelme.py`, `src/data/synthetic.py`)과 Phase 2(공통
인터페이스 + Method A 파이프라인)는 다음 세션에서 이어간다. Phase 3(나머지 4개 방법론)과
Phase 4(비교 스크립트)는 이후 세션에서 진행한다.

## 6. 검증 방법

1. `python -c "from src.models.direct.model import DirectModel; import torch; m = DirectModel(); print(m(torch.randn(2,3,512,512)).shape)"` -> `torch.Size([2, 8])`
2. `python scripts/create_data.py --dataset smartdoc --data_dir <raw_dir> --output_path data/smartdoc/gt_corners.csv` 로 `gt_corners.csv` 생성 확인 (완료: 24,860행 생성, `midv2020`도 68,409행 생성 완료)
3. `python scripts/fix_data.py data/smartdoc/gt_corners.csv data/midv2020/gt_corners.csv` 로 코너 순서/퇴화 샘플 보정 확인 (완료: smartdoc `fixed=0 removed=0`, midv2020 `fixed=1 removed=0`)
4. `python scripts/train.py --method direct --max_epochs 1` 스모크 테스트 (데이터 없으면 더미 텐서 기반 유닛 테스트로 대체)
4. `git log --oneline` 로 Phase별 커밋 확인
