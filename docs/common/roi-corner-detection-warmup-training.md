# ROI Corner Detection Warmup 학습과 단계적 Fine-tuning

## 초록

사전학습 backbone과 새로 초기화한 head를 한 번에 같은 방식으로 학습하면, 불안정한
head가 만든 gradient가 backbone의 유용한 표현을 빠르게 훼손할 수 있다. Warmup 학습은
초기 몇 epoch 동안 backbone을 동결하고 head만 학습한 뒤, 정해진 시점에 backbone을
해제하여 전체 모델을 작은 학습률로 fine-tuning하는 전략이다. 이 문서는 warmup의
필요성을 gradient 관점에서 설명하고, optimizer parameter group을 처음부터 등록한 채
`requires_grad`만 전환하는 구현을 정리한다. 또한 scheduler, early stopping,
BatchNorm과의 상호작용을 분석하고, 이 프로젝트의 `Trainer`와 `BaseWrapper` hook을
이용해 같은 실행 안에서 동결과 해제를 자동화하는 방법을 제시한다.
Backbone 특징의 위치 표현은 [공간 정보 손실과 보존](roi-corner-detection-spatial-information.md),
gradient를 만드는 목적 함수는 [손실 함수의 원리와 적용](roi-corner-detection-loss-functions.md)을
함께 참조한다.

## 1. 서론

### 1.1 전이 학습과 사전학습 모델

전이 학습은 큰 데이터셋에서 학습한 모델의 파라미터를 새로운 문제의 초기값으로
사용하는 방법이다. 입력 이미지에서 선, 모서리, 질감, 형태를 추출하는 능력은 서로
다른 시각 문제 사이에서도 상당 부분 재사용할 수 있다. ImageNet에서 학습한 ResNet이나
일반 장면의 직선을 학습한 M-LSD를 처음부터 다시 학습하지 않는 이유가 여기에 있다.

사전학습 파라미터를 $\theta_b$, 새 작업을 위한 head 파라미터를 $\theta_h$라고 하면
모델은 다음과 같이 쓸 수 있다.

$$
F(I; \theta_b, \theta_h)
= h\bigl(b(I; \theta_b); \theta_h\bigr)
$$

$b$는 backbone, $h$는 head이다. 전이 학습 시작 시 $\theta_b$는 이미 의미 있는 값을
가지지만 $\theta_h$는 대개 무작위 초기값이다. 두 파라미터 집합의 출발 상태가 다르므로
학습 초기에도 서로 다른 취급이 필요하다.

### 1.2 Backbone, Encoder, Head, Decoder 용어

이 문서에서 backbone과 encoder는 입력을 특징 맵으로 변환하는 앞부분을 가리킨다.
head는 특징을 최종 raw 출력으로 바꾸는 뒷부분의 일반 명칭이다. 공간 해상도를 다시
키우는 segmentation이나 line 모델에서는 head를 decoder라고도 부른다.

| 용어 | 역할 | 예시 |
|---|---|---|
| backbone | 일반 시각 특징 추출 | ResNet, MobileNetV2, DINOv2 |
| encoder | 입력을 낮은 해상도 특징으로 압축 | ResNet stage, MobileNet block |
| head | 특징을 과제별 출력으로 변환 | FC(8), heatmap head |
| decoder | 낮은 해상도 특징을 공간 출력으로 복원 | U-Net decoder, M-LSD FPN |

문맥에 따라 backbone과 encoder, head와 decoder를 같은 의미로 사용할 수 있지만,
optimizer parameter group을 구성할 때는 실제 모듈 경계를 명확히 해야 한다.

### 1.3 Warmup 학습의 목적

Warmup의 목표는 학습률을 단순히 서서히 높이는 일반적인 learning-rate warmup과 다르다.
이 문서에서 warmup은 다음 두 단계를 뜻한다.

1. 초기 $W$ epoch 동안 backbone의 parameter gradient를 차단하고 head만 학습한다.
2. $W$ epoch가 끝나면 backbone을 해제하고 전체 모델을 차등 학습률로 fine-tuning한다.

단계 흐름은 다음과 같다.

```text
pretrained backbone + random head
    $\to$ backbone freeze
    $\to$ head warmup
    $\to$ backbone unfreeze
    $\to$ full fine-tuning
```

## 2. Warmup이 필요한 이유

### 2.1 무작위 Head와 초기 Gradient

손실을 $\mathcal{L}$이라고 하면 backbone gradient는 연쇄 법칙으로 다음과 같이
계산된다.

$$
\frac{\partial \mathcal{L}}{\partial \theta_b}
=
\frac{\partial \mathcal{L}}{\partial h}
\frac{\partial h}{\partial b}
\frac{\partial b}{\partial \theta_b}
$$

무작위 head는 아직 backbone 특징을 올바른 코너 좌표나 line map으로 해석하지 못한다.
따라서 초기 $\partial \mathcal{L}/\partial h$와 $\partial h/\partial b$는 새 작업에 맞게
정렬되지 않았다. 이 gradient를 큰 pretrained backbone에 바로 전달하면, 유용한 특징이
head의 초기 오류에 맞추어 불필요하게 이동할 수 있다.

### 2.2 사전학습 표현의 파국적 망각

사전학습으로 얻은 일반 표현이 새 데이터의 제한된 분포에 과도하게 맞춰지면서 사라지는
현상을 파국적 망각이라고 한다. 특히 실측 데이터가 적거나 batch가 작으면 한두 batch의
우연한 패턴이 backbone 전체에 큰 영향을 줄 수 있다. Warmup은 backbone을 잠시 고정해
head가 기존 표현을 사용하는 법부터 배우게 하므로 초기 망각 위험을 줄인다.

Warmup이 항상 이득인 것은 아니다. 사전학습 도메인과 목표 도메인이 크게 다르면
backbone 적응을 너무 오래 막는 것이 오히려 손해일 수 있다. 따라서 warmup epoch는
고정된 이론값이 아니라 검증 실험으로 선택한다.

### 2.3 Backbone 동결의 수학적 의미

동결은 forward 계산을 제거하는 것이 아니다. 이미지가 backbone을 통과해 특징을 만드는
과정은 그대로 수행한다. 차이는 backward에서 parameter gradient를 만들지 않는다는 점이다.

$$
\theta_b^{(t+1)} = \theta_b^{(t)},
\qquad 0 \le t < W
$$

PyTorch에서는 각 parameter에 다음 값을 설정한다.

```python
for param in model.backbone.parameters():
    param.requires_grad = False
```

이 상태에서는 $\theta_b$의 `.grad`가 `None`이므로 optimizer가 해당 parameter를
업데이트하지 않는다. head에는 gradient가 생성되므로 정상적으로 학습된다.

### 2.4 Head 선행 학습과 전체 Fine-tuning

Warmup이 끝나면 backbone을 다시 학습 가능 상태로 바꾼다.

```python
for param in model.backbone.parameters():
    param.requires_grad = True
```

이후에는 head가 이미 과제의 기본 출력 규칙을 학습했으므로 backbone에 전달되는 gradient가
초기보다 안정적이다. backbone은 작은 학습률로 목표 도메인의 fringe, 문서 경계 또는
line 구조에 적응하고, head는 더 큰 학습률로 계속 조정된다.

## 3. 단계적 학습 절차

### 3.1 Optimizer Parameter Group 사전 등록

동결 해제 시 optimizer를 다시 만들지 않으려면 학습 시작 전에 backbone과 head를 모두
optimizer에 등록한다.

```python
backbone_ids = {id(param) for param in model.backbone.parameters()}
head_params = [
    param for param in model.parameters()
    if id(param) not in backbone_ids
]

optimizer = AdamW([
    {"params": model.backbone.parameters(), "lr": 1e-5},
    {"params": head_params, "lr": 1e-4},
])
```

두 group은 같은 optimizer step을 사용하지만 서로 다른 학습률을 유지한다. 동결 중인
backbone은 optimizer에 존재해도 gradient가 없으므로 건너뛴다.

### 3.2 학습 시작 시 Backbone 동결

Wrapper 생성 시점보다 실제 fit 시작 시점에 동결하는 편이 상태 전이를 명확하게 만든다.
이 프로젝트에서는 `Trainer.fit`과 `Trainer.fit_early_stop`이 첫 epoch 전에
`on_fit_start(max_epochs)`를 호출한다.

```python
def on_fit_start(self, max_epochs):
    self._epochs_done = 0
    self.set_backbone_requires_grad(
        self._epochs_done >= self.warmup_epochs
    )
```

`warmup_epochs=0`이면 비교식이 참이므로 처음부터 backbone을 학습한다.

### 3.3 Warmup Epoch 동안 Head 학습

Warmup 중에도 `model.train()`, forward, loss, backward, optimizer step의 전체 흐름은
동일하다. `requires_grad=False`인 parameter만 backward graph의 학습 대상에서 제외된다.
따라서 공통 Trainer에 방법론별 조건문을 넣을 필요가 없다.

이 설계는 Trainer가 방법론 이름이나 warmup 정책을 알지 못하게 한다. Trainer는 hook만
호출하고, 동결 정책은 Wrapper가 소유한다.

### 3.4 N Epoch 이후 Backbone 자동 해제

각 epoch의 검증이 끝난 뒤 `on_epoch_end(valid_score)`가 호출된다. 완료한 epoch 수가
`warmup_epochs`와 같아지는 순간 backbone을 해제한다.

```python
def on_epoch_end(self, valid_score=None):
    super().on_epoch_end(valid_score)
    self._epochs_done += 1
    if self._epochs_done == self.warmup_epochs:
        self.set_backbone_requires_grad(True)
```

`warmup_epochs=3`이면 epoch 1, 2, 3은 head만 학습하고 epoch 3 종료 시 해제한다.
따라서 backbone의 첫 optimizer update는 epoch 4에서 일어난다.

### 3.5 차등 학습률 Fine-tuning

동결 해제 후 parameter update는 다음과 같다.

$$
\theta_b \leftarrow \theta_b - \eta_b \nabla_{\theta_b}\mathcal{L},
\qquad
\theta_h \leftarrow \theta_h - \eta_h \nabla_{\theta_h}\mathcal{L}
$$

일반적으로 $\eta_b < \eta_h$로 둔다. 이 프로젝트의 대표 설정은
$\eta_b=10^{-5}$, $\eta_h=10^{-4}$이다. 이 비율은 절대 규칙이 아니라 사전학습 표현을
보존하면서 새 head를 빠르게 학습하기 위한 초기값이다.

### 3.6 전체 Epoch 상태 전이

`warmup_epochs=3`의 상태 전이는 다음과 같다.

| 시점 | Backbone | Head | Backbone update |
|---|---|---|---|
| fit 시작 | 동결 | 학습 가능 | 없음 |
| epoch 1 | 동결 | 학습 | 없음 |
| epoch 2 | 동결 | 학습 | 없음 |
| epoch 3 | 동결 | 학습 | 없음 |
| epoch 3 종료 | 해제 | 학습 가능 | 아직 없음 |
| epoch 4 이후 | 학습 | 학습 | 있음 |

## 4. 학습 제어기와의 상호작용

### 4.1 requires_grad와 AdamW

PyTorch optimizer는 gradient가 `None`인 parameter를 업데이트 대상 목록에서 제외한다.
AdamW의 decoupled weight decay도 이 parameter에는 실행되지 않는다. 따라서 backbone을
optimizer에 미리 넣는 것만으로 동결이 깨지지 않는다.

동결 해제 후 첫 backward에서 gradient가 생기면 AdamW가 해당 parameter의 moment 상태를
필요한 시점에 생성한다. head의 moment 상태는 warmup 동안 축적된 값을 계속 사용한다.

### 4.2 Optimizer를 재생성하지 않는 이유

동결 해제 시 optimizer를 새로 만들면 head가 축적한 Adam의 1차, 2차 moment와 scheduler가
관리하던 학습률 상태가 사라진다. optimizer를 유지하면 학습 과정이 하나의 연속된
실험으로 남고 checkpoint와 로그 해석도 단순해진다.

단, 사용자가 custom optimizer를 Wrapper에 주입한다면 처음부터 backbone과 head가 모두
등록되어 있어야 한다. decoder만 가진 custom optimizer는 `requires_grad=True`로 바꾸어도
backbone을 업데이트할 수 없다.

### 4.3 Scheduler와 Warmup

현재 `BaseWrapper.on_epoch_end`는 `ReduceLROnPlateau`에 검증 점수를 전달한 뒤 방법론별
Wrapper 로직으로 돌아온다. `DocWrapper`는 그 다음에 epoch 수를 증가시키고 backbone을
해제한다. 따라서 scheduler는 warmup 중에도 plateau를 누적한다.

이 동작은 backbone이 막 해제된 직후 이미 두 group의 학습률이 줄어드는 결과를 만들 수
있다. 다음 정책 중 무엇을 사용할지는 실험 전에 고정해야 한다.

1. 현재 공통 동작을 유지해 warmup과 fine-tuning을 하나의 plateau 구간으로 본다.
2. 동결 해제 시 scheduler 상태를 재시작해 두 단계를 분리한다.
3. warmup 동안 scheduler step을 생략한다.

현재 프로젝트의 `DocWrapper`는 첫 번째 정책이다. 다른 정책을 도입하면 공통 학습 규약과
checkpoint 재현성까지 함께 검토해야 한다.

### 4.4 Early Stopping과 Patience

Trainer의 early stopping은 epoch 1부터 검증 점수의 개선 여부를 계산한다. warmup 완료
시점에 wait counter를 자동으로 초기화하지 않는다. 따라서 patience가 너무 작으면
backbone이 충분히 학습되기 전에 종료될 수 있다.

안전한 초기 조건은 다음과 같다.

$$
P > W + E_{adapt}
$$

$P$는 early-stopping patience, $W$는 warmup epoch, $E_{adapt}$는 동결 해제 후 성능
변화를 관찰할 최소 epoch 수이다. 예를 들어 $W=3$이고 해제 후 최소 3 epoch를 보려면
patience를 6보다 크게 두는 방안을 검토한다. 실제 값은 점수 변동성과 데이터 크기에 따라
정한다.

### 4.5 BatchNorm 통계의 예외

`requires_grad=False`는 parameter만 동결한다. BatchNorm의 `running_mean`,
`running_var`, `num_batches_tracked`는 parameter가 아니라 buffer이다. 공통 train step이
`model.train()`을 호출하면 backbone의 BatchNorm 통계는 warmup 중에도 갱신된다.

```text
BatchNorm weight, bias: requires_grad로 동결
running_mean, running_var: train mode에서 계속 갱신
```

작은 batch에서는 이 통계가 불안정할 수 있다. 완전 동결이 필요하면 매 train step에서
backbone을 다시 `eval()` 상태로 두는 별도 정책이 필요하다. 단순히 `on_fit_start`에서
한 번 `backbone.eval()`을 호출하면 다음 `model.train()`에서 다시 train mode가 되므로
충분하지 않다. 현재 `DocWrapper`는 parameter만 동결하고 BatchNorm 통계는 갱신하는
정책이다.

## 5. Warmup Epoch 결정 방법

### 5.1 Decoder-only 수렴 시점

먼저 backbone을 계속 동결한 실험으로 decoder가 기존 특징을 얼마나 빠르게 사용하는지
볼 수 있다. 이 실험의 validation 곡선이 평탄해지는 시점은 warmup 후보를 정하는 참고값이다.
그러나 decoder-only 최고 epoch가 곧 최적 warmup epoch라는 보장은 없다. backbone을
해제한 뒤의 적응 성능까지 포함해 비교해야 한다.

### 5.2 Warmup Epoch Ablation

다른 조건을 고정하고 다음과 같은 후보를 비교한다.

| 설정 | 의미 |
|---|---|
| `warmup_epochs=0` | 처음부터 전체 fine-tuning |
| `warmup_epochs=1` | 최소 head 안정화 |
| `warmup_epochs=3` | 기본 후보 |
| `warmup_epochs=5` | 긴 decoder 적응 |

각 실험은 같은 seed, 데이터 split, 초기 checkpoint, 최대 epoch와 patience를 사용해야 한다.
Warmup이 길수록 전체 parameter update 횟수가 줄어드므로 최대 epoch가 너무 짧지 않은지도
확인한다.

### 5.3 IoU와 Success Rate 기반 선택

Loss 감소만으로 warmup 값을 선택하면 안 된다. 최종 목적은 코너 품질이므로 Polygon IoU,
MCD, MaxCD와 같은 task metric을 사용한다. 후처리가 실패할 수 있는 line, segmentation,
detection 계열은 Success Rate도 함께 본다.

선택 우선순위는 다음과 같이 둘 수 있다.

1. 검증 Success Rate가 허용 기준을 만족한다.
2. Polygon IoU가 높다.
3. MCD와 MaxCD가 낮다.
4. 여러 seed에서 순위가 안정적이다.

### 5.4 과소 및 과도한 Warmup

Warmup이 너무 짧으면 head의 불안정한 gradient가 backbone을 손상할 수 있다. 반대로 너무
길면 backbone이 목표 도메인에 적응할 시간이 부족하고, early stopping이 해제 직후
학습을 끝낼 수 있다.

| 상태 | 관찰 가능한 현상 |
|---|---|
| 과소 warmup | 초기 loss 진동, backbone 성능 급락, seed 민감도 증가 |
| 적절한 warmup | head loss 안정화 후 metric 개선 지속 |
| 과도한 warmup | warmup 후반 metric 정체, 해제 후 적응 전에 종료 |

## 6. 결론

Warmup은 pretrained backbone과 새 head의 초기 상태 차이를 반영하는 단계적 최적화
전략이다. 핵심 구현은 backbone과 head를 optimizer에 처음부터 등록하고, Wrapper hook에서
`requires_grad`만 전환하는 것이다. 이 방식은 optimizer와 scheduler를 재생성하지 않고
한 번의 `train.py` 실행으로 head warmup과 전체 fine-tuning을 연결한다.

Warmup은 독립적인 성능 향상 장치가 아니라 초기 gradient를 제어하는 수단이다. 최적
epoch는 validation metric으로 결정해야 하며 scheduler, early stopping, BatchNorm buffer의
동작을 함께 고려해야 재현 가능한 학습 전략이 된다.

## 참고 자료

- Yosinski, J., Clune, J., Bengio, Y., Lipson, H. "How Transferable Are Features in Deep Neural Networks?" NeurIPS, 2014.
- Howard, J., Ruder, S. "Universal Language Model Fine-tuning for Text Classification." ACL, 2018.
- Kornblith, S., Shlens, J., Le, Q. V. "Do Better ImageNet Models Transfer Better?" CVPR, 2019.
- PyTorch. "Autograd Mechanics."
- PyTorch. "Per-parameter Options in Optimizer."

## 부록 A. 프로젝트 학습 Hook

### A.1 Trainer와 BaseWrapper 호출 순서

공통 호출 순서는 다음과 같다.

```text
Wrapper construction
    $\to$ Trainer.fit or fit_early_stop
    $\to$ wrapper.on_fit_start(max_epochs)
    $\to$ train epoch
    $\to$ validation epoch
    $\to$ wrapper.on_epoch_end(valid_score)
    $\to$ early-stopping decision
```

`BaseWrapper.on_fit_start`는 기본적으로 아무 일도 하지 않는다. warmup이 필요한 Wrapper만
이를 override한다. `BaseWrapper.on_epoch_end`는 scheduler step을 담당하므로 override한
메서드는 `super().on_epoch_end(valid_score)`를 호출해야 한다.

### A.2 on_fit_start와 on_epoch_end

재사용 가능한 최소 패턴은 다음과 같다.

```python
def set_backbone_requires_grad(self, requires_grad):
    for param in self.model.backbone.parameters():
        param.requires_grad = requires_grad

def on_fit_start(self, max_epochs):
    self._epochs_done = 0
    self.set_backbone_requires_grad(
        self._epochs_done >= self.warmup_epochs
    )

def on_epoch_end(self, valid_score=None):
    super().on_epoch_end(valid_score)
    self._epochs_done += 1
    if self._epochs_done == self.warmup_epochs:
        self.set_backbone_requires_grad(True)
```

`max_epochs`는 공통 hook signature를 맞추기 위해 받으며 현재 패턴에서는 사용하지 않는다.

### A.3 Custom Optimizer 계약

Wrapper가 기본 optimizer를 만들 때는 backbone과 head group을 모두 등록한다. 사용자가
optimizer를 주입하면 Wrapper는 그 구성을 변경하지 않는다. 따라서 custom optimizer는
다음 조건을 만족해야 한다.

- 해제 후 학습할 모든 backbone parameter를 포함한다.
- head 또는 decoder parameter를 포함한다.
- parameter가 중복 group에 들어가지 않는다.
- scheduler가 같은 optimizer 인스턴스를 참조한다.

## 부록 B. Wrapper 구현

### B.1 공통 구현 패턴

Wrapper 생성자의 권장 순서는 다음과 같다.

1. model, preprocessor, postprocessor를 생성한다.
2. `BaseWrapper` 생성자를 호출한다.
3. backbone과 나머지 parameter를 분리한다.
4. 두 optimizer group을 등록한다.
5. scheduler, losses, metrics를 등록한다.
6. `warmup_epochs`와 epoch counter를 초기화한다.

이 순서는 프로젝트의 공통 Wrapper 구성 원칙과 optimizer, scheduler 대칭성을 유지한다.

### B.2 DocWrapper 분석

현재 `DocWrapper`는 `warmup_epochs=3`을 기본값으로 사용한다. ResNet encoder와 공간 보존
head를 optimizer에 각각 $10^{-5}$, $10^{-4}$로 등록하고 `on_fit_start`에서 encoder를
동결한다. 세 번째 epoch 종료 후 자동으로 해제하므로 네 번째 epoch부터 전체 모델을
학습한다. 이는 이 문서의 참조 구현이다.

### B.3 LineWrapper 적용

현재 `LineWrapper`는 M-LSD backbone을 생성자에서 영구 동결하고 optimizer에는 decoder만
등록한다. warmup으로 전환하려면 영구 동결을 제거하고 다음 두 group을 처음부터 등록해야
한다.

```python
self.set_optimizer(self.optimizer or AdamW([
    {"params": self.model.backbone.parameters(), "lr": 1e-5},
    {"params": decoder_params, "lr": 1e-4},
]))
```

그 다음 `DocWrapper`와 같은 hook을 추가한다. optimizer에 backbone이 없는 상태에서
`requires_grad=True`만 설정하는 것은 효과가 없다.

### B.4 경계 조건

- `warmup_epochs=0`: fit 시작부터 backbone을 학습한다.
- `warmup_epochs>=max_epochs`: 실행 내내 backbone이 동결된다.
- 같은 Wrapper로 fit을 다시 호출하면 counter가 0으로 초기화되고 warmup도 다시 시작한다.
- early stopping으로 best model을 복원해도 `requires_grad` 상태는 state dict에 포함되지 않는다.
- 추론에서는 `requires_grad` 상태가 출력값에 영향을 주지 않는다.

## 부록 C. 실행과 검증

### C.1 CLI와 설정

기본값을 Wrapper에 고정하면 기존 `--method` 명령만으로 동작한다. 여러 warmup 값을 CLI에서
비교하려면 공통 argument parser와 factory kwargs 전달 경로에 `warmup_epochs`를 추가해야
한다. 이 변경은 적용 대상 Wrapper가 인수를 받을 때만 전달되도록 설계해야 한다.

### C.2 Parameter Group 확인

학습 전 다음 항목을 확인한다.

- optimizer group이 정확히 두 개인가
- backbone group 학습률이 $10^{-5}$인가
- head group 학습률이 $10^{-4}$인가
- 두 group의 parameter ID가 겹치지 않는가
- 전체 model parameter가 두 group의 합과 일치하는가

### C.3 동결 및 해제 테스트

`warmup_epochs=2`인 작은 테스트에서 다음을 검증한다.

1. `on_fit_start` 후 backbone의 모든 `requires_grad`가 `False`이다.
2. head parameter는 `True`이다.
3. epoch 1 종료 후 backbone은 계속 동결 상태이다.
4. epoch 2 종료 후 backbone이 모두 `True`가 된다.
5. 해제 후 backward에서 backbone gradient가 생성된다.
6. optimizer 인스턴스 ID와 head의 Adam 상태가 전환 전후 동일하다.
