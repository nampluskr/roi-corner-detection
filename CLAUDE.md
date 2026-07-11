# CLAUDE.md

이 파일은 `roi-corner-detection` 프로젝트에서 작업할 때 따르는 공통 지침이다.

## 1. 프로젝트 개요

평면 사각형 객체의 4개 코너 좌표를 검출하는 12개 방법론(direct, homography, heatmap, seg,
hybrid, det, gcn, doc, foundation, line, torchseg, torchdet)을 공유 데이터/평가 파이프라인
위에서 구현하고 성능을 비교하는 단일 PyTorch 프로젝트이다.
방법론 목록과 구현 순서, 제약(F1-F8)은 `README.md`를 참조한다.

## 2. 통합 원칙

이 프로젝트는 task 축이 "코너 좌표 검출" 하나뿐이고, 방법론(12개) 차이는 raw 출력 형태
자체(좌표/마스크/박스/heatmap/직선 등)가 다르다. 따라서 다음 두 축으로 공통성을 확보한다.

- **인터페이스 공통**: 모든 방법론은 `src/models/base/`의 4종 추상 클래스(`BaseModel`,
  `BasePreprocessor`, `BasePostprocessor`, `BaseWrapper`)를 직접 상속한다.
- **방법론 독립**: `src/models/<name>/`은 다른 방법론 폴더를 import하거나 상속하지 않는다.
  같은 알고리즘을 쓰더라도 각 방법론의 4개 파일 안에 독립적으로 구현한다.
- **실행 공통**: `src/core/trainer.py`의 `Trainer`, `src/core/evaluator.py`의 `Evaluator`,
  `src/core/predictor.py`의 `Predictor`는 방법론에 상관없이 `Wrapper` 인스턴스 하나만 받아
  동일하게 구동된다.
- **step 공통**: `BaseWrapper`가 기본 `train_step`/`eval_step`과 단순 대입 세터
  `set_optimizer`/`set_scheduler`를 제공하며, 각 wrapper는 `__init__`에서 `set_optimizer` 다음에
  `set_scheduler(self.scheduler or ReduceLROnPlateau(self.optimizer, ...))`로 기본 scheduler를
  지정한다(optimizer의 `AdamW` 지정과 대칭, 커스텀은 `scheduler` 인자로 주입). 방법론별 loss
  계산 차이는 `compute_losses()`만 오버라이드하고, 모델 호출 규약이 다를 때만 전체 step을
  오버라이드한다. 누적 결과는 `get_loss_results()`/`get_metric_results()`로 조회한다.
- **구성 고정**: 각 Wrapper는 `super().__init__()` 앞에서 같은 폴더의 preprocessor와
  postprocessor를 `preprocessor or XxxPreprocessor()` 형태로 지정해 `BaseWrapper`에 필수
  인자로 전달한다. 인자로 주입되면 그 값을 쓰고 없으면 기본값을 생성하며(losses/metrics와
  동일 취지), 런타임에 교체하지 않는다.
- **loss 공통**: loss는 `src/losses/`에 `BaseLoss` 상속 클래스(loss당 파일 하나)로 두며,
  metrics와 대칭으로 모델 이름이 아닌 의미 기반 이름(`WingLoss` 등)으로 정의한다. 각 방법론
  wrapper가 `losses` dict로 이들을 조합해 쓴다. `BaseLoss`는 `weight`(기본 1.0)를 가지며
  `train_step`이 `loss_fn.weight`를 곱해 여러 loss를 가중합한다. `compute_losses`를
  오버라이드할 때 반환 dict의 key는 `losses` dict의 key와 일치시킨다.
- **출력 공통**: 모든 방법론의 `postprocessor.py`는 raw 출력을 표준 코너 좌표 `(N, 4, 2)`로
  변환하며, 그 이후 `pred_corners.csv` 스키마(`image_id,x1,y1,x2,y2,x3,y3,x4,y4`, 정규화 [0,1])와
  `src/metrics/`의 `BaseMetric` 상속 클래스(메트릭당 파일 하나)들의 공통 메트릭 적용은
  방법론 이름조차 몰라도 되는 완전 공통 로직이다.
- **CLI 공통**: `scripts/train.py --method <name>` 등 CLI 인수와 동작 형태 동일

방법론별로 다른 부분은 `src/models/<name>/`(model/preprocessor/postprocessor/wrapper)
4개 파일로 한정한다. 공유 계층(`src/core/`, `src/data/`, `src/losses/`, `src/metrics/`,
`src/utils/`)에는 방법론 이름을 하드코딩하지 않는다.

## 3. 공통 파일 구조

```text
data/
docs/
experiments/
notebooks/
outputs/
scripts/
├── config.py
├── create_data.py
├── evaluate.py
├── fix_data.py
├── predict.py
└── train.py
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
│   ├── labelme.py
│   ├── midv2020.py
│   ├── smartdoc.py
│   ├── synthetic.py
│   └── transforms.py
├── losses/
│   ├── base_loss.py
│   └── wing_loss.py
├── metrics/
│   ├── base_metric.py
│   ├── max_cd.py
│   ├── mcd.py
│   ├── pck.py
│   ├── polygon_iou.py
│   └── reprojection_error.py
├── models/
│   ├── base/
│   │   ├── base_model.py
│   │   ├── base_postprocessor.py
│   │   ├── base_preprocessor.py
│   │   └── base_wrapper.py
│   ├── det/      (model.py, postprocessor.py, preprocessor.py, wrapper.py)
│   ├── direct/      (동일 4개 파일)
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
    └── homography.py
```

`configs/` 폴더는 두지 않는다. 하이퍼파라미터는 `scripts/config.py`의 `DEFAULTS` 딕셔너리로
관리한다 (MNIST 프로젝트와 동일).

## 4. 방법론별 전용 파일

각 방법론 폴더(`det/direct/doc/foundation/gcn/heatmap/homography/hybrid/line/seg/torchdet/torchseg`)는
항상 **같은 이름의 파일 4개**(`model.py`, `preprocessor.py`, `postprocessor.py`, `wrapper.py`)를
가지며, 내용만 방법론별로 다르다. loss는 `src/losses/`의 재사용 클래스를 `wrapper.py`가
`losses` dict로 조합해 쓴다. raw 출력 형태와 후처리 방식 차이는 다음과 같다 (구현 순서 순).
각 파일의 대표 클래스는 대응하는 `BaseModel`/`BasePreprocessor`/`BasePostprocessor`/
`BaseWrapper`를 직접 상속한다.

| 방법론 | raw 출력 | 후처리 |
|---|---|---|
| `direct` | (N, 8) 좌표 logits | sigmoid + reshape |
| `homography` | (N, 8) offset | 정준 좌표 + offset |
| `heatmap` | (N, 4, H, W) heatmap | soft-argmax |
| `seg` | (N, 1, H, W) quad 마스크 | findContours + approxPolyDP |
| `hybrid` | (N, 1, H, W) 세그멘테이션 마스크 | Canny + Hough + cornerSubPix |
| `det` | (N, A, 5+4) 그리드 박스 예측 | 클래스별 top-1 박스 중심 |
| `gcn` | 초기 코너 + GCN 반복 정제 출력 | 최종 반복 출력 |
| `doc` | (N, 8) 좌표 logits (사전학습 기반) | sigmoid + reshape |
| `foundation` | (N, 8) 좌표 logits (frozen backbone) | sigmoid + reshape |
| `line` | 직선 세그먼트 | 직선 그룹화 + 교점 계산 |
| `torchseg` | (N, 1, H, W) quad 마스크 (torchvision seg 모델) | findContours + approxPolyDP (독립 구현) |
| `torchdet` | torchvision 검출 출력 (박스/라벨/점수) | 클래스별 top-1 박스 중심 |

방법론 전용이면서 다른 방법론이 쓰지 않는 보조 연산(예: `line`의 직선 교점 계산)은
공유 `src/utils/`로 승격하지 않고 해당 방법론 폴더 내부(`postprocessor.py` 등)에 둔다.
다른 방법론에서도 필요해지는 시점에 `src/utils/`로 승격한다.

## 5. 작업 규칙

- 코드 작성 시 `README.md`에 정의된 스켈레톤과 시그니처를 반드시 적용한다. `README.md`를 SSOT로 삼는다.
- 방법론 모델을 구현하거나 성능을 평가할 때는 해당 방법론의 `docs/models/` 상세 문서를 반드시 참조한다.
  방법론 파일명은 구현 순서 번호를 접두어로 갖는다 (`01_direct-coordinate-regression`,
  `02_homography-regression`, `03_heatmap-keypoint-detection`, `04_segmentation-corner`,
  `05_dl-classical-cv-hybrid`, `06_bbox-keypoint-detection`, `07_polygon-gcn`,
  `08_document-pretrained`, `09_foundation-adapter`, `10_line-intersection`,
  `11_torchvision-segmentation`, `12_torchvision-detection`).
  문서의 설계(head 구조, 학습 타깃, loss, 후처리, 평가 기준)와 구현/평가가 어긋나면 문서를 기준으로 조정한다.
- 새 방법론을 추가할 때는 `src/models/<name>/`에 4개 파일(model/preprocessor/postprocessor/wrapper)만
  추가하고 `src/core/`, `src/data/`, `src/losses/`, `src/metrics/`, `src/utils/`는 수정하지 않는다
  (필요 시 `factory.py`의 `get_wrapper` dispatch 분기만 추가, 새 loss가 필요하면 `src/losses/`에
  의미 기반 클래스로 추가).
- `src/models/base/`의 4종 추상 클래스(공개 시그니처)를 변경하면 이미 구현된 모든 방법론에 동시 반영한다.
- `scripts/` CLI 인수와 동작이 달라지면 전체 방법론에 동일하게 적용되도록 구현한다.
- 방법론 구현 순서는 `PLAN.md`의 난이도/속도 기준 순서(direct $\to$ homography $\to$ heatmap $\to$ seg
  $\to$ hybrid $\to$ det $\to$ gcn $\to$ doc $\to$ foundation $\to$ line $\to$ torchseg $\to$ torchdet)를 따른다.
- 각 방법론은 `main`에서 분기한 `method/<name>` 브랜치(예: `method/direct`)에서 구현/학습/평가를
  진행하고, 검증을 마친 뒤 `main`에 merge한다. 다음 방법론 브랜치는 이전 방법론 merge 완료 후에
  분기한다.

## 6. 문서 및 코드 작성 규칙

- 마크다운 문서와 코드 파일에 em dash, 유니코드 화살표, 이모지를 사용하지 않는다.
- 마크다운 문서에서 화살표 표기가 필요하면 `->` 대신 `$\to$`를 사용한다.
- 단, 폴더 구조 트리에서는 ├, ─, │, └ 문자를 사용한다.
- 폴더 구조와 파일 리스트는 탐색기와 동일한 형태/순서를 유지한다: 폴더 알파벳순 $\to$ 파일 알파벳순.
- `docs/` 하위 방법론 상세 보고서 작성 시 `docs/docs-plan.md`의 문서명/목차/템플릿을 따른다.

### 6.1 노트북 작성 규칙

- Jupyter 노트북 셀의 `source` 배열에서 마지막 줄은 `\n`으로 끝내지 않는다.
- 즉, `source` 배열의 마지막 원소는 개행 문자 없이 끝나야 한다.

## 7. 레거시 참조 코드

이 프로젝트는 아래 두 프로젝트만 참조하여 작성한다. 참조만 하고 복사/심링크하지 않으며,
전체 재구현을 원칙으로 한다. 이후 모든 문서와 대화에서 두 프로젝트는 다음 명칭으로 지칭한다.

| 명칭 | 경로 | 참조 목적 |
|---|---|---|
| **레거시 프로젝트** (legacy 프로젝트) | `../../99_deprecated/260622_roi-direct-regression` | 도메인 로직 참조 - 코너 기하 연산, 데이터 파이프라인, 방법론 설계(A-J), 통합 계획 |
| **MNIST 프로젝트** | `../260626_mnist-from-scratch` | 구조/컨벤션 참조 - 폴더 구조, SSOT README/CLAUDE.md 작성 방식, 코딩 규칙 |

레거시 프로젝트 세부 참조 경로:

| 경로 | 참조 대상 |
|---|---|
| `<레거시 프로젝트>/src/{utils,data}` | 코너 기하 연산, CSV 기반 Dataset/Transform 설계 |
| `<레거시 프로젝트>/docs/plans/roi-detection-calm-papert.md` | 방법론 통합 구조 설계 원안 |
| `<레거시 프로젝트>/_project/sources/P3-methodology-catalog.md` | 방법론 A-J 상세 카탈로그 |
| `<레거시 프로젝트>/_project/sources/P5-metrics-and-schema.md` | 메트릭 카탈로그 (Polygon IoU, MCD, MaxCD, Reprojection Error 등) |
| `<레거시 프로젝트>/_legacy/quad_regression/src` | Trainer(`engine.py`)/Loss 구현 패턴 |

MNIST 프로젝트 세부 참조 대상:

| 경로 | 참조 대상 |
|---|---|
| `<MNIST 프로젝트>/CLAUDE.md`, `<MNIST 프로젝트>/README.md` | SSOT 문서 작성 방식, 동기화/작업 규칙 서술 형식 |
| `<MNIST 프로젝트>/mnist-numpy`, `<MNIST 프로젝트>/mnist-pytorch` | `src/{core,data,models,utils}` 폴더 구성, `scripts/`, `experiments/run.py` 패턴, `Trainer(wrapper, output_dir=None)` 시그니처 |

## 8. Python 코딩 규칙

- 코드 내 한국어 삽입 금지. 식별자, 주석, 문자열 모두 영어로 작성한다.
- 세로 정렬(공백으로 맞추는 컬럼 정렬)을 사용하지 않는다.
- `pathlib.Path` 대신 `os.path`를 사용한다.
- 타입 힌트를 사용하지 않는다.
- 파일 첫 줄에 `# src/data/filename.py: one-line description` 형식으로 프로젝트 루트 기준 경로를 포함하여 작성한다.
  이 헤더 주석 다음에는 빈 줄을 하나 삽입한 뒤 import 구문을 작성한다.
- 클래스와 최상위 함수에 1줄 docstring을 작성한다. 클래스의 메서드에는 docstring을 달지 않는다.
- 주석은 필요한 경우에만 최소한으로 삽입한다.
- `src/` 하위 모든 폴더에 빈 `__init__.py`를 둔다.
- `src/` 내부 파일 간 임포트는 `src.xxx` 절대 임포트를 사용한다. 상대 임포트(`..`, `.`)는 사용하지 않는다.
- `src/` 외부(`scripts/`, `experiments/` 등)에서는 `sys.path`로 프로젝트 루트를 추가한 뒤 `src.xxx`로 임포트한다.

## 9. 실행 환경

- 코드 실행 및 검증(`torch`, `torchvision` 등 import 확인, 스크립트 실행)은 conda 환경
  `pytorch_env`를 사용한다 (`torch 2.5.1+cu121`, `torchvision 0.20.1+cu121`).
- `torch_py311_cuda118` 환경은 `torch`가 커스텀 빌드(`2.5.1.post303`)라 `torchvision` 임포트 시
  `torchvision::nms` 연산자 오류(ABI 불일치)가 발생한다. `torchvision`이 필요한 작업에는 사용하지 않는다.

## 10. 하위 문서

- [[README]] - SSOT: 방법론 카탈로그, 폴더 구조, 모듈 시그니처
- [[PLAN]] - Phase별 작업 계획 및 검증 방법
- [[roi-corner-detection-models]] - 12개 방법론 상세 비교 (`docs/common/roi-corner-detection-models.md`)
- [[docs-plan]] - 방법론 상세 보고서 문서명/목차/템플릿 (`docs/docs-plan.md`)
