# PLAN.md

이 문서는 `roi-corner-detection` 프로젝트의 현재 구현 상태와 남은 작업을 관리하는
로드맵이다. 모듈 구조와 공개 시그니처의 SSOT는 `README.md`이며, 이 문서는 상태, 검증
순서, 완료 기준만 다룬다.

## 1. 프로젝트 범위와 구현 순서

이 프로젝트는 평면 사각형 객체의 4개 코너 좌표를 검출하는 12개 방법론을 하나의 공통
데이터, 학습, 평가 파이프라인에서 구현하고 비교한다. 방법론의 raw 출력은 달라도 모든
후처리는 표준 코너 좌표 `(N, 4, 2)`를 반환한다.

구현 순서는 난이도와 구현 속도를 기준으로 다음과 같다.

`direct` $\to$ `homography` $\to$ `heatmap` $\to$ `seg` $\to$ `hybrid` $\to$ `det` $\to$ `gcn` $\to$ `doc`
$\to$ `foundation` $\to$ `line` $\to$ `torchseg` $\to$ `torchdet`

상세 설계는 각 방법론의 `docs/models/` 문서를, 방법론 전체 비교와 우선순위 근거는
`docs/common/roi-corner-detection-models.md`와
`docs/common/roi-corner-detection-implementation-order.md`를 참조한다.

## 2. 공통 아키텍처

- 모든 방법론은 `src/models/<name>/`에 `model.py`, `preprocessor.py`,
  `postprocessor.py`, `wrapper.py`의 동일한 4개 파일을 둔다.
- `model.py`는 방법론별 raw 출력을 만들고, `preprocessor.py`는 표준 코너 좌표를
  방법론별 학습 타깃으로 변환하며, `postprocessor.py`는 raw 출력을 표준 코너 좌표로
  변환한다.
- loss는 모델 폴더에 두지 않는다. `src/losses/`에서 의미 기반 파일별 클래스로 정의하고,
  모든 loss는 `BaseLoss`를 상속한다. 각 wrapper가 필요한 loss를 `losses` dict로 조합한다.
- `src/models/base/`의 4종 추상 클래스와 `src/core/`, `src/data/`, `src/metrics/`,
  `src/utils/`는 방법론에 독립적인 공통 계층이다. 공통 계층의 공개 시그니처와 세부 동작은
  `README.md`를 따른다.

## 3. 현재 구현 상태

### 완료된 공통 기반

- 데이터셋, dataloader, 이미지 변환, SmartDoc 및 MIDV2020 데이터 생성, 이미지 목록 생성
- 코너 기하, 호모그래피 유틸, 공통 메트릭, 의미 기반 `BaseLoss` 계층
- `Trainer`, `Evaluator`, `Predictor`, `factory`, 학습, 평가, 예측 CLI 파이프라인

### 구현 및 등록 완료 방법론

12개 방법론 모두 `src/models/<name>/`의 4개 전용 파일과 `factory.get_wrapper()` 등록을
갖고 있으며, `main`에서 생성할 수 있다.

| 코드 | 구현 상태 | 현재 구현 요약 |
|---|---|---|
| `direct` | 구현 및 등록 완료 | GAP 또는 공간 보존 head를 선택하는 직접 좌표 회귀 기준선 |
| `homography` | 구현 및 등록 완료 | 정준 사각형 기준 bounded offset 회귀 |
| `heatmap` | 구현 및 등록 완료 | deconvolution heatmap과 soft-argmax |
| `seg` | 구현 및 등록 완료 | custom encoder-decoder mask 예측과 contour 후처리 |
| `hybrid` | 구현 및 등록 완료 | mask 예측 후 Canny, Hough, cornerSubPix 정밀화 |
| `det` | 구현 및 등록 완료 | 코너별 작은 box를 예측하는 custom grid detector |
| `gcn` | 구현 및 등록 완료 | 초기 코너를 GCN으로 반복 정제 |
| `doc` | 구현 및 등록 완료 | ImageNet ResNet, 공간 보존 head, backbone warmup |
| `foundation` | 구현 및 등록 완료 | frozen DINOv2 patch grid와 경량 공간 보존 head |
| `line` | 구현 및 등록 완료 | M-LSD MobileNetV2/FPN tp-map과 직선 교점 후처리 |
| `torchseg` | 구현 및 등록 완료 | torchvision segmentation 모델 대조군 |
| `torchdet` | 구현 및 등록 완료 | torchvision detection 모델 대조군 |

여기서 구현 완료는 모듈 구조, 공통 인터페이스 연결, factory 등록을 뜻한다. 모든 방법론이
동일한 데이터와 조건에서 충분히 수렴했고 성능이 검증되었다는 뜻은 아니다. 특히 `line`은
초기 학습에서 낮은 IoU가 관찰되어 target, loss, 후처리 threshold를 포함한 안정화 검증이
필요하다.

## 4. 남은 실행 단계와 완료 기준

### Phase 1 - 방법론별 통합 검증과 안정화

- 12개 wrapper 각각에 대해 생성, 1 epoch 학습, checkpoint 평가, 예측 CSV 생성을 확인한다.
- 각 방법론의 preprocessor, raw output, postprocessor shape와 표준 코너 규약을 검증한다.
- `line`은 dense line target, center/displacement loss, 직선 그룹화 threshold를 우선 점검하고
  validation IoU와 Success Rate가 함께 개선되는지 확인한다.
- 새 구조를 추가할 때는 전용 4파일과 `factory.get_wrapper()` dispatch를 함께 갱신한다. 새 loss가
  필요하면 `src/losses/`에 의미 기반 `BaseLoss` 하위 클래스로 추가한다.
- 방법론 전용 보조 연산은 해당 방법론 폴더에 두고, 둘 이상의 방법론에서 재사용될 때만
  `src/utils/`로 승격한다.

### Phase 2 - 보조 데이터 확장

- LabelMe 파서와 합성 데이터 생성은 방법론 구현을 막지 않는 별도 보조 작업으로 진행한다.
- 추가 데이터는 기존 CSV 스키마와 표준 정규화 코너 좌표 규약을 유지해야 한다.

### Phase 3 - 전체 비교 평가

- 동일한 데이터 분할, 입력 해상도, 평가 조건 및 checkpoint 선택 기준으로 12개 방법론을
  benchmark한다.
- 비교 결과에는 Polygon IoU, MCD, MaxCD, Reprojection Error, PCK, Success Rate,
  추론 지연, 모델 크기를 포함한다.
- 정량 결과와 실패 사례 시각화를 비교 산출물로 남긴다.

각 방법론과 공통 계층 변경은 다음 완료 기준을 만족해야 한다.

1. `factory.get_wrapper()` 등록과 wrapper 생성이 확인된다.
2. 전처리, 모델, 후처리의 입출력 형태와 공통 loss, metric 연동이 확인된다.
3. 제한된 실제 데이터로 1 epoch 학습, 체크포인트 평가, 예측 CSV 생성이 수행된다.
4. 예측 좌표가 표준 스키마와 코너 유효성 규칙을 만족하는지 확인한다.
5. 공통 계층을 변경한 경우 12개 방법론의 import, wrapper 생성, 예측 경로를
   회귀 검증한다.
