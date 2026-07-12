# Ultralytics YOLO Detection (yolo)

## 설치

이 방법론은 `ultralytics==8.4.92`를 사용한다. 현재 프로젝트의 `pytorch_env`는 Python
3.10, PyTorch 2.5.1+cu121, torchvision 0.20.1+cu121로 이 패키지의 기본 요구 사항을
충족한다. 이미 설치된 PyTorch와 torchvision을 다시 설치하지 않도록 활성 환경에서 다음을
실행한다.

```text
conda activate pytorch_env
python -m pip install "ultralytics==8.4.92" --upgrade-strategy only-if-needed
python -m pip check
python -c "import ultralytics, torch, torchvision; print(ultralytics.__version__); print(torch.__version__); print(torchvision.__version__)"
```

YOLO11의 COCO 사전학습 weight는 다른 backbone과 동일하게 `/mnt/d/backbones/`에 둔다.
네트워크에 연결된 최초 1회에는 해당 디렉터리에서 다음 명령으로 `yolo11n.pt`와
`yolo11s.pt`를 준비한다.

```text
cd /mnt/d/backbones
conda run -n pytorch_env python -c "from ultralytics import YOLO; YOLO('yolo11n.pt'); YOLO('yolo11s.pt')"
```

이 명령은 Ultralytics의 weight 탐색 및 내려받기 기능을 사용한다. 이후 학습과 추론은 로컬
weight 파일을 사용하도록 구성한다. Ultralytics 소프트웨어의 AGPL-3.0 라이선스와 weight의
사용 조건을 확인해야 하며, 폐쇄형 또는 상용 배포에서는 Enterprise License 필요 여부를
별도로 검토한다.

## 초록

Ultralytics YOLO Detection은 코너를 네 개의 작은 객체로 바꾸고, 단일 단계 객체 검출기로
각 객체의 위치와 정체성을 동시에 예측하는 방법이다. 네 코너는 TL, TR, BR, BL의 네 class로
분리되며, 각 class의 최고 점수 box 중심이 최종 코너가 된다. YOLO는 backbone, neck,
detection head가 한 번의 forward pass에서 다중 해상도 예측을 만들므로, 두 단계 검출기보다
간결한 추론 경로를 가진다. 이 문서는 YOLO11n과 YOLO11s를 이용해 코너 검출에 적용하는
이론, target 표현, loss, 후처리 및 공통 학습 파이프라인 연결 규약을 설명한다.

## 1. 서론

### 1.1 단일 단계 검출기로 코너를 찾는 이유

객체 검출은 이미지 안에서 무엇이 있는지와 어디에 있는지를 함께 예측한다. 단일 단계
검출기는 후보 영역을 먼저 만드는 별도 단계 없이, 특징 맵의 여러 위치에서 class 점수와
box를 직접 예측한다. YOLO 계열은 이 방식을 효율적으로 구현한 대표적인 구조다.

코너는 면적이 없는 점이므로 일반 객체 검출기의 직접 target이 될 수 없다. 그러나 각 코너를
중심으로 하는 작은 정사각형 box를 정의하면, 코너 검출은 네 개의 class를 갖는 작은 객체
검출 문제로 바뀐다. class가 코너 순서를 보존하므로, 검출 후 별도의 순서 추정이나 Hungarian
matching이 필요하지 않다.

이 방법은 직접 설계한 `det`과 torchvision 검출기를 재사용하는 `torchdet` 사이의 비교 축을
제공한다. `det`은 이 문제에 맞춘 최소 grid detector이고, `torchdet`은 torchvision 모델을
사용하며, `yolo`는 Ultralytics가 제공하는 단일 단계 검출기와 그 사전학습 생태계를 사용한다.

### 1.2 문제 정의

입력 이미지를 $I \in \mathbb{R}^{3 \times H \times W}$, 표준 코너를

$$
Y = \{(x_k, y_k)\}_{k=0}^{3} \in [0, 1]^{4 \times 2}
$$

로 둔다. 코너 순서는 $0=$ TL, $1=$ TR, $2=$ BR, $3=$ BL이다. 각 코너에 대해 중심은
$(x_k, y_k)$이고 폭과 높이가 $r$인 가상 box $b_k$를 정의한다. 학습 목표는 네 개 class의
box 집합을 예측하는 함수 $f_\theta$다.

$$
f_\theta(I) = \{(\hat{b}_m, \hat{p}_m, \hat{c}_m)\}_{m=1}^{M}
$$

여기서 $\hat{b}_m$은 예측 box, $\hat{p}_m$은 confidence, $\hat{c}_m$은 class다. 최종
코너는 class별 최고 점수 예측의 box 중심을 고르는 결정적 후처리 $g$로 복원한다.

$$
\hat{Y} = g\big(f_\theta(I)\big) \in [0, 1]^{4 \times 2}
$$

## 2. 이론 및 아키텍처

### 2.1 Backbone, Neck, Detection Head

YOLO 검출기는 세 부분으로 구성된다. backbone은 입력 이미지에서 의미 있는 특징을 추출하고,
neck은 서로 다른 해상도의 특징을 결합하며, detection head는 각 해상도에서 box와 class를
예측한다. 작은 물체와 큰 물체는 필요한 공간 해상도가 다르므로, 하나의 특징 맵만 쓰지 않고
여러 scale을 함께 사용한다.

입력 크기가 $640 \times 640$일 때 일반적인 세 검출 scale은 다음과 같다.

| 특징 scale | Stride | 공간 shape | 역할 |
|---|---:|---:|---|
| P3 | 8 | $80 \times 80$ | 작은 코너 box의 위치 구분 |
| P4 | 16 | $40 \times 40$ | 중간 크기 문맥 |
| P5 | 32 | $20 \times 20$ | 넓은 문맥과 큰 구조 |

backbone의 깊은 층은 넓은 수용 영역을 가져 사각형 전체의 방향과 문맥을 이해하는 데 유리하다.
얕은 층은 해상도가 높아 코너 주변의 국소 위치를 구분하는 데 유리하다. neck의 top-down 및
bottom-up 결합은 이 두 종류의 정보를 P3, P4, P5에 함께 전달한다.

```text
이미지 (3x640x640)
  -> backbone -> 다중 해상도 특징
  -> neck     -> 융합 특징 P3, P4, P5
  -> head     -> box 분포, class logit, confidence
  -> NMS      -> class별 후보 box
  -> top-1    -> 코너 (4x2)
```

### 2.2 Task-Aligned Assignment

학습에서는 모든 grid 위치가 같은 중요도를 갖지 않는다. 정답 box와 공간적으로 가깝고 class
예측도 좋은 후보를 양성으로 선택해야 한다. task-aligned assignment는 class 점수 $s$와 box
IoU $u$를 함께 사용하는 alignment metric을 정의한다.

$$
t = s^{\alpha} u^{\beta}
$$

$\alpha$와 $\beta$는 분류와 위치 품질의 비중을 조절하는 상수다. $t$가 큰 후보는 정답 class를
잘 맞히면서 정답 box에도 잘 겹치므로 양성 학습 신호가 된다. 이 방식은 코너처럼 target이
적고 배경 후보가 매우 많은 문제에서, 단순 거리 기준보다 분류와 위치 목표를 더 일관되게
맞춘다.

### 2.3 Box Regression과 CIoU Loss

예측 box $B$와 정답 box $B^*$의 위치 품질은 intersection over union으로 측정한다.

$$
\mathrm{IoU}(B, B^*) = \frac{|B \cap B^*|}{|B \cup B^*|}
$$

완전히 겹치지 않는 초기 학습 단계에서는 IoU만으로 center 위치와 종횡비 차이를 충분히
구분하기 어렵다. Complete IoU는 center 거리와 종횡비 항을 추가한다.

$$
\mathcal{L}_{\mathrm{CIoU}} = 1 - \mathrm{IoU}
+ \frac{\rho^2(\mathbf{c}, \mathbf{c}^*)}{d^2}
+ \eta v
$$

$\mathbf{c}$와 $\mathbf{c}^*$는 두 box 중심, $d$는 두 box를 함께 감싸는 최소 box의 대각선
길이, $v$는 종횡비 차이, $\eta$는 그 가중치다. 코너 box는 고정된 정사각형을 target으로
삼으므로 중심 오차가 주된 항이지만, CIoU는 clipping으로 box 크기가 달라지는 경계 코너에도
안정적인 회귀 신호를 준다.

### 2.4 Distribution Focal Loss와 Class Loss

YOLO의 box 경계는 연속값 하나를 바로 회귀하는 대신, 인접한 이산 bin에 대한 확률 분포로
표현할 수 있다. 정답 거리 $y$가 두 bin $i$, $i+1$ 사이에 있을 때 Distribution Focal Loss는
두 bin의 log probability를 선형 보간해 사용한다.

$$
\mathcal{L}_{\mathrm{DFL}} =
-(i+1-y) \log p_i - (y-i) \log p_{i+1}
$$

이 표현은 box 경계의 작은 연속 변화에도 안정적인 gradient를 주며, CIoU와 함께 위치 loss를
구성한다. class 예측은 각 class에 대한 binary cross entropy로 학습한다.

$$
\mathcal{L}_{\mathrm{cls}} = -\sum_{c=0}^{3}
\left[y_c \log \sigma(z_c) + (1-y_c) \log(1-\sigma(z_c))\right]
$$

전체 loss는 box, class, DFL 항의 가중합이다.

$$
\mathcal{L} = \lambda_{\mathrm{box}} \mathcal{L}_{\mathrm{CIoU}}
+ \lambda_{\mathrm{cls}} \mathcal{L}_{\mathrm{cls}}
+ \lambda_{\mathrm{dfl}} \mathcal{L}_{\mathrm{DFL}}
$$

가중치는 Ultralytics detection model의 기본 loss 설정을 사용한다. 학습 로그에는 이 내부
항들의 합을 하나의 detection loss로 기록하고, 최종 모델 선택은 공통 Polygon IoU로 한다.

### 2.5 NMS와 코너 복원

검출 head는 같은 코너에 대해 여러 후보를 만들 수 있다. 비최대 억제(NMS)는 confidence가 높은
box부터 선택하고 이미 선택한 같은 class box와 IoU가 큰 중복 후보를 제거한다. NMS 이후 class
$k$에 대한 후보 집합을 $D_k$라고 하면, 최종 box는 다음과 같다.

$$
\hat{b}_k = \arg\max_{(b,p,k) \in D_k} p
$$

선택한 box가 $(x_1, y_1, x_2, y_2)$이고 입력 폭과 높이가 $W$, $H$이면, 정규화 코너는

$$
\hat{x}_k = \frac{x_1+x_2}{2W}, \qquad
\hat{y}_k = \frac{y_1+y_2}{2H}
$$

이다. 네 class 모두가 존재하고 score 기준을 통과하며, 네 점이 유효 사각형을 이루는 경우에만
결과를 성공으로 취급한다. 그렇지 않으면 해당 표본의 코너를 NaN으로 반환해 Success Rate가
후처리 실패를 분리해서 집계한다.

## 3. 결론

Ultralytics YOLO Detection은 코너를 class가 있는 소형 box로 바꿔, 성숙한 단일 단계 검출기를
코너 추정에 사용하는 방법이다. 다중 해상도 특징과 사전학습 weight를 활용하면서도 class별
top-1 중심 선택으로 표준 코너 순서를 바로 복원할 수 있다. YOLO11n은 비용이 작은 기준
모델이고, YOLO11s는 더 큰 표현력과 비용의 절충안이다. 이 방법론은 custom grid detector와
torchvision detector 사이에서 라이브러리 구현의 성능과 비용을 비교하는 기준을 제공한다.

## 참고 자료

- Redmon, J., Divvala, S., Girshick, R., Farhadi, A. "You Only Look Once:
  Unified, Real-Time Object Detection." CVPR 2016.
- Jocher, G., Chaurasia, A., Qiu, J. "Ultralytics YOLO." 공개 구현,
  https://github.com/ultralytics/ultralytics
- Zheng, Z., et al. "Distance-IoU Loss: Faster and Better Learning for Bounding
  Box Regression." AAAI 2020.
- Li, X., et al. "Generalized Focal Loss: Learning Qualified and Distributed
  Bounding Boxes for Dense Object Detection." NeurIPS 2020.

## 부록 A. 데이터 인터페이스

YOLO target은 이미지별 box 목록을 하나의 batch tensor로 결합한다. 표준 코너
$Y \in [0,1]^{4 \times 2}$에서 각 코너 $k$에 대해 정규화 box 크기 $r=1/16$을 사용한다.

$$
b_k = (x_k, y_k, r, r)
$$

경계에서 box가 이미지를 벗어날 수 있으므로, 실제 폭과 높이는 중심에서 각 경계까지의
거리로 clipping한다. 이때 box 중심은 원래 코너를 유지한다. batch 크기 $N$에서 target은
다음 필드를 갖는다.

| 필드 | shape | 의미 |
|---|---|---|
| `img` | $(N, 3, 640, 640)$ | YOLO 입력 범위인 $[0,1]$ image tensor |
| `batch_idx` | $(4N, 1)$ | 각 box가 속한 이미지 index |
| `cls` | $(4N, 1)$ | $0,1,2,3$ 코너 class |
| `bboxes` | $(4N, 4)$ | normalized center `xywh` box |

공통 dataloader는 ImageNet normalization을 적용하므로, YOLO wrapper는 loss와 inference 전에
그 정규화를 되돌려 $[0,1]$ 범위의 image tensor를 만든다. 이 변환은 image 값만 바꾸며
정규화 좌표 target에는 영향을 주지 않는다.

postprocessor는 NMS가 반환한 `xyxy`, confidence, class 결과를 class별 top-1로 줄인 후 box
중심을 정규화한다. class 누락, score 부족, 퇴화 코너는 NaN으로 처리한다.

## 부록 B. 학습 전략

YOLO 적용은 공통 `BaseWrapper` 인터페이스를 유지한다. Ultralytics의 고수준 `model.train()`
루프를 호출하지 않고, 내부 detection model의 forward, loss, NMS를 `YoloWrapper`의
`train_step`, `eval_step`, `predict_step`에 연결한다. 따라서 기존 `Trainer`의 early stopping,
공통 Evaluator의 코너 메트릭, Predictor의 `pred_corners.csv` 출력을 그대로 사용한다.

기본 모델과 입력 크기는 다음과 같다.

| 항목 | 기본값 |
|---|---|
| variant | `yolo11n`, `yolo11s` |
| 입력 해상도 | $640 \times 640$ |
| class 수 | 4 |
| class 순서 | TL, TR, BR, BL |
| box 크기 | 이미지 한 변의 $1/16$ |
| optimizer | Nesterov SGD, learning rate $0.01$, momentum $0.937$ |
| weight decay | $5 \times 10^{-4}$ |

학습 데이터는 다른 방법론과 같은 세 단계를 따른다.

| 단계 | 데이터 | 목적 |
|---|---|---|
| Pre-training | SmartDoc + MIDV-2020 | 일반 사각형 코너의 검출 위치 학습 |
| Domain Adaptation | 합성 fringe 이미지 | fringe 반사와 광학 변동 적응 |
| Fine-tuning | 실측 fringe 이미지 | 실제 장비 조건의 정밀도 보정 |

최종 비교에서는 YOLO만 640으로 측정하지 않는다. 입력 해상도 영향을 분리하기 위해 13개
방법론을 모두 640 조건에서 학습 또는 평가하고, 기존 224 결과는 개발 기준선으로 구분한다.

## 부록 C. 제약 적합성 분석

| 제약 | 대응 |
|---|---|
| F1 | 코너별 소형 box 중심 네 개가 임의 볼록 사각형을 표현하므로 전체 quad를 하나의 box로 근사하지 않는다 |
| F2 | 단일 대형 객체 환경에서는 네 코너가 충분히 떨어져 있어 class별 top-1 선택이 안정적이다 |
| F3 | target box clipping으로 경계 근처 코너도 유효한 학습 box를 만든다 |
| F4 | COCO 사전학습 weight와 공개, 합성, 실측의 세 단계 학습을 함께 사용한다 |
| F5 | box 중심은 연속 좌표지만 detection stride의 영향을 받으므로 reprojection error와 PCK를 함께 확인한다 |
| F6 | YOLO11n과 YOLO11s를 함께 측정해 정확도와 CPU, GPU 지연의 절충을 비교한다 |
| F7 | ColorJitter, blur, noise와 합성 fringe의 반사, glare 변형으로 광학 변화에 대응한다 |
| F8 | class마다 정확히 하나의 코너가 있다는 가정이 class별 top-1 후처리를 단순화한다 |
