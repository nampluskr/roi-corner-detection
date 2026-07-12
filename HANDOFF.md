# ROI Corner Detection ver1 to ver2 Handoff

## 1. 문서 목적과 ver2 전환 범위

이 문서는 `260701_roi-corner-detection-ver1`의 구조와 용어를 정리하여
`260712_roi-corner-detection-ver2`에서 재구현을 시작할 수 있게 하는 인수인계 문서다.
ver2는 ver1의 12개 코너 검출 방법론, 공통 코너 좌표 규약, 데이터 파이프라인 및 평가
메트릭을 계승한다. 목표는 방법론 자체를 교체하는 것이 아니라, 실험 단위와 모델 구성의
명명, CLI, 산출물, benchmark를 일관된 계약으로 다시 구성하는 것이다.

ver1은 보존한다. ver2는 ver1의 파일을 이동하거나 import하지 않고 독립 프로젝트로
구현한다. 필요한 도메인 로직과 설계는 ver1을 참조하되 코드를 복사하지 않는다.

## 2. 표준 용어와 네이밍 원칙

모든 식별자는 소문자 snake_case를 사용한다. 사람이 읽는 제목에는 설명형 이름을 쓰고,
CLI, 경로, 설정 파일, benchmark에는 아래의 짧은 식별자를 사용한다.

| 용어 | 표준 의미 | 예시 |
|---|---|---|
| dataset | 학습 또는 평가에 사용하는 데이터 단계 | `public`, `synthetic`, `measured` |
| data source | dataset을 구성하는 개별 원천 | `smartdoc`, `midv2020` |
| method | 코너 검출의 학습 타깃, raw 출력, 후처리 규약을 함께 정하는 방법론 | `direct`, `heatmap`, `torchdet` |
| model | 하나의 method 안에서 선택 가능한 완성된 신경망 구성 | `resnet50_gap`, `deeplabv3_resnet50` |
| backbone | model 내부의 특징 추출기 또는 사전학습 encoder | `resnet50`, `dinov2_vits14` |
| run | 하나의 완전히 해석된 학습 설정으로 실행한 단위 | 입력 크기, batch size, learning rate, seed가 고정된 실행 |
| checkpoint | 특정 run에서 저장한 모델 상태 | `model_best.pth`, `model_last.pth` |

`dataset`, `method`, `model`, `backbone`은 서로 대체하지 않는다. 특히 `backbone`은
model의 일부이므로, torchvision의 완성 모델이나 M-LSD 전체 모델을 `backbone`이라고
부르지 않는다.

## 3. `method`, `model`, `backbone`의 역할 정의

### 3.1 Method

`method`는 사용자가 선택하는 최상위 알고리즘 단위다. 다음 12개 코드를 유지한다.

```text
direct, homography, heatmap, seg, hybrid, det, gcn, doc,
foundation, line, torchseg, torchdet
```

method는 method 전용 preprocessor, model, postprocessor, wrapper와 loss 조합을 결정한다.
따라서 `--method`는 필수 CLI 인자이며, `src/models/<method>/`의 구현 폴더 이름과
`get_wrapper(method, ...)`의 dispatch key로 동일하게 사용한다. `src/models/`는 모델
구현 계층의 패키지 이름이고, 하위 폴더 이름이 method 코드다.

### 3.2 Model

`model`은 특정 method에서 학습 가능한 전체 네트워크 구성을 뜻한다. model 이름에는
backbone뿐 아니라 method별 head, decoder, detector 계열처럼 결과를 바꾸는 구조 차이가
반영되어야 한다.

예를 들어 direct method의 `resnet50_gap`과 `resnet50_spatial`은 서로 다른 model이며,
`resnet50`은 두 model에 공통으로 들어갈 수 있는 backbone이다. `deeplabv3_resnet50`,
`fasterrcnn_resnet50_fpn`, `dinov2_vits14_linear`처럼 외부 라이브러리의 완성 모델은
그 자체를 model로 부른다.

`--model`은 필수 CLI 인자로 한다. method별 지원 model과 기본값은 하나의 model registry에
정의하고, factory는 `method`와 `model`의 조합이 유효한지 검증한 뒤 wrapper를 생성한다.

### 3.3 Backbone

`backbone`은 model 구현 내부에서만 사용하는 구성 요소 이름이다. 공통 CLI 인자
`--backbone`은 ver2에서 제거한다. 모델 구조를 바꾸려는 실행은 `--model`을 사용하며,
model 내부의 세부 encoder 선택이 실험 대상일 때도 별도의 model registry 항목으로 등록한다.
이 규칙으로 ver1에서 `--backbone`에 혼재한 encoder, 전체 segmentation model, detector,
foundation model의 의미를 분리한다.

## 4. CLI 인자와 실험 설정 네이밍

모든 train, evaluate, predict, run, benchmark 진입점은 같은 해석 규칙을 사용한다.
일반 실행의 식별 인자는 다음과 같다.

```text
--dataset <public|synthetic|measured>
--method <method_code>
--model <model_name>
--image_size <int>
--batch_size <int>
--learning_rate <float>
--max_epochs <int>
--seed <int>
```

학습은 `--init_checkpoint`로 직전 데이터 단계의 가중치를 받을 수 있다. `--checkpoint`은
evaluate와 predict에서 읽을 가중치를 명시할 때만 사용한다. `--output_dir`은 기본 경로를
벗어나야 하는 예외적 경우에만 허용하며, 일반 batch 실험에서는 사용하지 않는다.

설정은 하나의 dict 또는 JSON으로 해석한 뒤에만 실행한다. 해석된 설정에는 dataset,
method, model, 모든 학습 인자, CSV 경로 목록, 초기 checkpoint 경로, 생성한 exp_name을
기록한다. 사용자 지정 경로 문자열을 dataset 식별자로 사용하지 않는다.

## 5. 데이터셋 단계 네이밍: `public`, `synthetic`, `measured`

ver2의 dataset은 원천 데이터 하나가 아니라 학습 단계 단위다.

| dataset | data source | 용도 | 기본 초기화 |
|---|---|---|---|
| `public` | SmartDoc, MIDV-2020 | 일반 코너 검출 사전학습 | ImageNet 또는 method별 공개 사전학습 가중치 |
| `synthetic` | 합성 fringe 이미지와 코너 레이블 | 검사 도메인 적응 | 같은 method와 model의 `public` checkpoint |
| `measured` | 실측 PMD 이미지와 코너 레이블 | 최종 미세 조정과 배포 평가 | 같은 method와 model의 `synthetic` checkpoint |

dataset registry는 각 dataset의 CSV 경로 목록, 기본 초기 checkpoint 정책, 설명을 보유한다.
`public`은 SmartDoc와 MIDV-2020 CSV를 함께 사용한다. `synthetic`과 `measured`는 단일 또는
복수 CSV를 사용할 수 있지만 dataset 코드와 경로 규칙은 변하지 않는다.

데이터의 train, valid, test 분할은 dataset 내부의 재현 가능한 split이다. `dataset`과
`split`은 서로 다른 축이므로 `public_train`처럼 결합한 식별자를 만들지 않는다.

## 6. 실험명, 가중치 파일명, 산출물 경로 규칙

모든 run의 식별자는 다음 규칙으로 생성한다.

```text
exp_name = <method>_<model>_i<image_size>_bs<batch_size>_lr<learning_rate>_ep<max_epochs>_s<seed>
```

예시는 다음과 같다.

```text
direct_resnet50_gap_i224_bs4_lr1e-3_ep50_s42
torchseg_deeplabv3_resnet50_i512_bs2_lr1e-4_ep30_s42
```

learning rate는 `1e-3`처럼 지수 표기를 사용하고, 소수점과 더하기 기호는 exp_name에 쓰지
않는다. method와 model은 exp_name에 포함하여 로그, CSV 행, 복사된 checkpoint에서도 실행을
식별할 수 있게 한다.

기본 산출물 경로는 아래와 같이 고정한다.

```text
outputs/<dataset>/<method>/<model>/<exp_name>/
├── model_best.pth
├── model_last.pth
├── config.json
├── history.json
├── eval_result.json
├── pred_corners.csv
└── run.log
```

checkpoint 파일명은 고정하고, 전체 경로가 실행 인자를 표현하도록 한다. `model_best.pth`는
validation monitor 기준 최선 상태, `model_last.pth`는 마지막 epoch 상태다. checkpoint를
다른 위치로 복사해야 할 때는 `<dataset>_<exp_name>_best.pth` 형식의 파일명을 사용한다.

`experiments/<dataset>/`에는 해당 dataset의 batch 설정과 benchmark 표를 둔다.
`notebooks/<dataset>/`에는 해당 dataset의 학습, 오류 분석, benchmark 해석 노트북을 둔다.
두 폴더 모두 method와 model 하위 폴더를 추가할 수 있으나, outputs와 다른 식별 규칙을
만들지 않는다.

## 7. Benchmark 결과 컬럼과 실험 식별자 규칙

benchmark의 한 행은 하나의 checkpoint 평가 결과다. benchmark는 전역 기본 CSV가 아니라
해당 행의 `dataset` registry로 test dataloader를 만들고, 행의 method와 model로 wrapper를
복원한다. 서로 다른 dataset의 점수는 같은 CSV에 기록할 수 있지만 반드시 dataset 컬럼으로
구분한다.

기본 컬럼 순서는 다음과 같다.

```text
dataset
method
model
exp_name
checkpoint
init_checkpoint
image_size
batch_size
learning_rate
max_epochs
seed
train_size
valid_size
test_size
iou
mcd
max_cd
reproj_error
sr
pck
cpu_latency_ms
gpu_latency_ms
params
size_mb
```

지원되는 method에 한해 추가 지표가 생기면 기존 공통 컬럼 뒤에 추가한다. benchmark의
결과 파일은 `experiments/<dataset>/benchmark.csv`에 저장하고, 여러 dataset을 합친 비교표는
`experiments/benchmark_all.csv`에 저장한다. 결과 CSV에는 경로 계산에 사용한 모든 인자가
독립 컬럼으로 있어야 하며, exp_name만으로 설정을 역추정하지 않는다.

## 8. ver2 프로젝트 폴더 및 모듈 구조

ver2는 다음 최상위 구조를 사용한다.

```text
roi-corner-detection-ver2/
├── data/
│   ├── measured/
│   ├── public/
│   └── synthetic/
├── docs/
├── experiments/
│   ├── measured/
│   ├── public/
│   └── synthetic/
├── notebooks/
│   ├── measured/
│   ├── public/
│   └── synthetic/
├── outputs/
│   ├── measured/
│   ├── public/
│   └── synthetic/
├── scripts/
├── src/
│   ├── core/
│   ├── data/
│   ├── losses/
│   ├── metrics/
│   ├── models/
│   └── utils/
├── README.md
└── PLAN.md
```

`src/models/<method>/`은 ver1과 동일하게 `model.py`, `preprocessor.py`,
`postprocessor.py`, `wrapper.py`를 가진다. ver2의 차이는 method별 코드 분리가 아니라,
model registry가 method와 model 조합을 명시적으로 관리하고 공통 CLI가 그 조합을 전달하는
점이다.

`outputs/`는 실행 산출물이며 Git으로 추적하지 않는다. `experiments/`는 재현 가능한
실험 설정과 benchmark CSV를, `notebooks/`는 분석을 관리한다. 각 폴더의 dataset 하위
구조는 동일한 식별자를 사용한다.

## 9. ver1 대비 용어, 인터페이스, 경로 구조 변경사항

| 항목 | ver1 | ver2 |
|---|---|---|
| 방법론 선택 | `--method` | `--method` 유지 |
| 네트워크 선택 | 전역 `--backbone` | 전역 `--model` |
| backbone 의미 | encoder와 완성 모델이 혼재 | model 내부 구성 요소로 한정 |
| 데이터 선택 | 기본 `csv_path` 목록 | `--dataset`과 dataset registry |
| wrapper 생성 | `get_wrapper(method, device, **kwargs)` | `get_wrapper(method, model, device, **kwargs)` |
| 실행명 | method, batch size, epoch, backbone 일부 | method, model, 이미지 크기, batch size, learning rate, epoch, seed |
| 기본 출력 경로 | `outputs/<method>/<exp_name>/` | `outputs/<dataset>/<method>/<model>/<exp_name>/` |
| checkpoint | `model.pth` 하나 | `model_best.pth`, `model_last.pth` |
| benchmark 입력 | 전역 기본 test CSV | 행별 dataset registry의 test CSV |
| benchmark 출력 | `outputs/comparison/results.csv` | `experiments/<dataset>/benchmark.csv`와 `experiments/benchmark_all.csv` |

ver1의 `DEFAULT_BACKBONES`, `get_experiment`, `get_output_dir`, `get_wrapper_kwargs`,
`experiments/configs.py`, `experiments/run.py`, `experiments/benchmark.py`는 위 계약에
맞춰 재설계 대상이다. method 구현의 코너 좌표 입출력 계약과 공통 metric 계산은 유지한다.

## 10. ver1에서 ver2로의 리팩터링 단계 계획

### Phase 1. 프로젝트 골격과 명명 계약 고정

- ver2에 최상위 폴더와 빈 `__init__.py`를 생성한다.
- README와 PLAN에 이 문서의 용어와 산출물 규칙을 SSOT로 기록한다.
- dataset registry와 model registry의 공개 인터페이스를 먼저 정의한다.

### Phase 2. 데이터와 설정 계층 구현

- `public`, `synthetic`, `measured` registry를 구현하고 각 CSV 목록을 연결한다.
- 공통 CLI에서 `--dataset`, `--method`, `--model` 및 실행 인자를 해석한다.
- 해석된 설정을 `config.json`으로 저장하고 exp_name과 output directory를 단일 함수로
  계산한다.

### Phase 3. 모델 선택 계층 이식

- ver1의 각 method 구현을 독립적으로 이식한다.
- method별 model registry에서 지원 model, 기본 model, 내부 backbone 구성을 선언한다.
- factory와 wrapper가 method와 model 조합을 검증하도록 변경한다.

### Phase 4. 학습과 checkpoint 계층 구현

- Trainer가 `model_best.pth`와 `model_last.pth`를 구분해 저장한다.
- synthetic과 measured 학습에서 `--init_checkpoint` 계보를 config와 benchmark에 기록한다.
- train, evaluate, predict가 동일한 기본 output directory 계산을 공유하게 한다.

### Phase 5. Batch run과 benchmark 구현

- dataset별 experiment config를 읽어 train, evaluate, predict를 실행한다.
- benchmark가 config의 dataset, method, model, checkpoint를 사용해 한 행을 생성한다.
- dataset별 benchmark와 전체 benchmark를 생성하고 notebook에서 직접 읽을 수 있게 한다.

### Phase 6. 방법론별 회귀 검증과 문서 완성

- 12개 method와 각 기본 model에 대해 생성, 1 epoch 학습, 평가, 예측을 확인한다.
- public, synthetic, measured의 checkpoint 초기화 계보를 확인한다.
- README, PLAN, CLI guide, 모델 문서를 실제 인터페이스와 동기화한다.

## 11. 검증 기준과 문서 동기화 계획

ver2 착수 후 다음 조건을 만족해야 한다.

1. `--dataset`, `--method`, `--model` 조합으로 train, evaluate, predict의 출력 경로가
   동일하게 계산된다.
2. 유효하지 않은 method와 model 조합, 존재하지 않는 dataset, 누락된 checkpoint는 실행 전에
   명확한 오류로 종료된다.
3. 모든 run directory에 해석된 `config.json`과 best, last checkpoint가 남는다.
4. benchmark의 한 행에 checkpoint를 복원하는 데 필요한 식별자와 모든 공통 평가 지표가
   들어간다.
5. benchmark는 평가 대상 run의 dataset test split을 사용한다.
6. 12개 method의 기본 model이 공통 코너 출력 규약, metric, CSV 출력 규약을 유지한다.
7. README는 공개 인터페이스의 SSOT, PLAN은 진행 상태와 검증 기준의 SSOT로 유지하고,
   이 문서는 ver1에서 ver2로 넘어가는 설계 판단의 SSOT로 유지한다.

ver2에서 이 문서의 계약을 변경해야 하면 먼저 ver2의 README와 PLAN을 변경하고, 변경 사유와
ver1 대비 차이를 ver2의 HANDOFF 또는 architecture 문서에 기록한다.
