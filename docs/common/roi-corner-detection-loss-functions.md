# ROI Corner Detection 손실 함수의 원리와 적용

## 초록

손실 함수는 모델의 예측과 정답 차이를 하나의 미분 가능한 값으로 바꾸고, 어떤 오류를
얼마나 강하게 수정할지 결정한다. ROI 코너 검출 프로젝트에는 좌표, offset, heatmap,
mask, grid box, class와 line displacement처럼 서로 다른 raw 출력이 공존하므로 하나의
손실을 모든 방법론에 적용할 수 없다. 이 문서는 MSE, Smooth L1, Wing, BCE, Cross
Entropy, Focal, Dice의 수식과 gradient 특성을 설명한다. 또한 여러 손실을 가중합하는
방법, 희소 위치만 회귀하는 masked loss, 반복 출력을 감독하는 deep supervision을
정리한다. 부록에서는 프로젝트의 `BaseLoss` 누적 인터페이스와 방법론별 실제 조합을
분석하고, loss가 감소해도 최종 코너 metric이 악화될 수 있는 원인을 진단한다.
출력 head의 위치 표현은 [공간 정보 손실과 보존](roi-corner-detection-spatial-information.md),
동결 구간의 gradient 제어는
[Warmup 학습과 단계적 Fine-tuning](roi-corner-detection-warmup-training.md)을 참조한다.

## 1. 서론

### 1.1 손실 함수의 역할

모델 $f_\theta$가 입력 $x$에서 예측 $\hat{y}=f_\theta(x)$를 만들고 정답이 $y$일 때,
손실 함수 $\mathcal{L}(\hat{y},y)$는 두 값의 불일치를 scalar로 표현한다. 학습은 평균
손실을 작게 만드는 parameter를 찾는 문제이다.

$$
\theta^*=arg\min_\theta
\frac{1}{N}\sum_{n=1}^{N}
\mathcal{L}\bigl(f_\theta(x_n),y_n\bigr)
$$

Gradient descent는 손실의 기울기를 이용한다.

$$
\theta \leftarrow \theta-\eta\nabla_\theta\mathcal{L}
$$

따라서 손실 함수는 단순한 점수표가 아니다. 큰 오류와 작은 오류, 양성과 음성, 겹침과
좌표 차이 중 무엇에 큰 gradient를 줄지 정하는 학습 규칙이다.

### 1.2 손실 함수와 평가 메트릭의 차이

손실 함수는 backward가 가능하도록 거의 모든 지점에서 미분 가능해야 한다. 평가 metric은
사람이 원하는 최종 품질을 직접 표현하며 미분 가능할 필요가 없다. Polygon IoU는 사각형
겹침을 잘 나타내지만 polygon clipping과 분기 때문에 현재 구현을 그대로 학습 손실로 쓰기
어렵다. 대신 좌표, mask 또는 heatmap 손실로 학습하고 IoU로 평가한다.

| 구분 | 손실 함수 | 평가 메트릭 |
|---|---|---|
| 목적 | parameter update 방향 생성 | 최종 성능 해석 |
| 미분 가능성 | 필요 | 불필요 |
| 예시 | Wing, BCE, Dice | Polygon IoU, MCD, SR |
| 계산 시점 | train과 validation | train, validation, test |

Loss 감소와 metric 개선은 자주 함께 일어나지만 반드시 일치하지는 않는다. preprocessor와
postprocessor가 다르거나, 손실이 최종 기하 품질을 충분히 표현하지 못하면 둘이 반대로
움직일 수 있다.

### 1.3 예측값, Logit, 확률, Target

Logit $z\in\mathbb{R}$는 sigmoid나 softmax 이전의 제한 없는 모델 출력이다. 이진 확률은
sigmoid로 얻는다.

$$
p=\sigma(z)=\frac{1}{1+e^{-z}}
$$

다중 class 확률은 softmax로 얻는다.

$$
p_k=\frac{e^{z_k}}{\sum_j e^{z_j}}
$$

손실 함수가 logit을 기대하는지 확률을 기대하는지 반드시 구분해야 한다.
`binary_cross_entropy_with_logits`는 내부에서 sigmoid와 로그 계산을 안정적으로 결합한다.
여기에 sigmoid를 먼저 적용하면 sigmoid가 중복되고 gradient가 달라진다.

Target의 의미도 출력마다 다르다.

- 좌표 target: 정규화된 실수 $[0,1]$
- heatmap target: Gaussian peak를 가진 연속값 $[0,1]$
- mask target: foreground와 background의 이진값
- class target: class index를 나타내는 정수
- displacement target: 중심에서 endpoint까지의 signed offset

### 1.4 Reduction과 Batch 평균

원소별 손실 $\ell_i$를 scalar로 만들기 위해 reduction을 적용한다.

$$
\mathcal{L}_{sum}=\sum_i\ell_i,
\qquad
\mathcal{L}_{mean}=\frac{1}{M}\sum_i\ell_i
$$

`mean`은 tensor 원소 수가 달라도 값의 scale을 비교하기 쉽지만, 양성 수가 매우 적은
dense prediction에서는 많은 background가 평균을 지배할 수 있다. `sum`을 사용하면
이미지 크기와 batch 크기에 따라 gradient scale이 바뀐다. 어떤 축과 원소를 평균하는지
손실마다 명확히 해야 한다.

이 프로젝트의 재사용 loss는 기본적으로 PyTorch 함수의 `mean` reduction을 사용한다.
`BaseLoss`는 batch별 scalar를 다시 sample count로 가중하여 epoch 평균을 누적한다.

## 2. 좌표 회귀 손실

### 2.1 Mean Squared Error

MSE는 오차 $e=\hat{y}-y$의 제곱 평균이다.

$$
\mathcal{L}_{MSE}=\frac{1}{M}\sum_{i=1}^{M}e_i^2
$$

원소별 gradient는 다음과 같다.

$$
\frac{\partial \ell}{\partial \hat{y}}=2e
$$

오차가 클수록 gradient가 선형으로 커지므로 큰 오류를 빠르게 줄이는 데 유리하다. 반면
잘못된 label이나 outlier 하나가 전체 update를 지배할 수 있다. Gaussian heatmap처럼
target 전체가 연속적인 회귀값일 때 단순하고 안정적인 기준이 된다.

### 2.2 Smooth L1과 Huber Loss

PyTorch Smooth L1은 임계값 $\beta$ 안에서 이차, 밖에서 선형인 함수이다.

$$
\ell_{SL1}(e)=
\begin{cases}
\dfrac{e^2}{2\beta}, & |e|<\beta \\
|e|-\dfrac{\beta}{2}, & |e|\ge\beta
\end{cases}
$$

Gradient는 다음과 같다.

$$
\frac{\partial \ell_{SL1}}{\partial e}=
\begin{cases}
\dfrac{e}{\beta}, & |e|<\beta \\
\operatorname{sign}(e), & |e|\ge\beta
\end{cases}
$$

작은 오차에서는 부드럽고 큰 오차에서는 gradient 크기가 1로 제한되어 outlier에 강하다.
좌표, offset, box regression에 널리 사용된다. PyTorch의 `SmoothL1Loss(beta)`와 전통적인
Huber loss는 scale 표기 방식이 다를 수 있으므로 구현 수식을 확인해야 한다.

현재 기본값은 $\beta=1$이다. 정규화 좌표나 $[-0.25,0.25]$ offset에서는 대부분 오차가
이차 구간에 들어간다. 선형 구간의 강건성을 실제로 사용하려면 target scale에 맞춘
$\beta$를 검증해야 한다.

### 2.3 Wing Loss

Wing Loss는 작은 좌표 오차에서 로그 곡선을 사용하고 큰 오차에서는 선형으로 이어진다.

$$
\ell_{wing}(e)=
\begin{cases}
w\ln\left(1+\dfrac{|e|}{\epsilon}\right), & |e|<w \\
|e|-C, & |e|\ge w
\end{cases}
$$

연속성을 위한 상수는 다음과 같다.

$$
C=w-w\ln\left(1+\frac{w}{\epsilon}\right)
$$

작은 오차 구간의 gradient 크기는 다음과 같다.

$$
\left|\frac{\partial\ell_{wing}}{\partial e}\right|
=\frac{w}{\epsilon+|e|}
$$

오차가 0에 가까울수록 상대적으로 큰 gradient를 주므로 landmark의 미세 조정을 강조한다.
큰 오차에서는 선형으로 증가해 MSE보다 outlier에 강하다.

프로젝트 기본값은 $w=10$, $\epsilon=2$이다. 정규화 좌표 오차는 최대 1이므로 실제
`direct`, `doc`, `foundation`에서는 모든 정상 오차가 $|e|<w$인 로그 구간에 들어간다.
즉 현재 설정에서는 선형 branch가 사실상 사용되지 않는다. 원 논문의 pixel-scale
parameter를 정규화 좌표에 그대로 적용한 결과이므로, $w$와 $\epsilon$은 coordinate
scale과 함께 ablation할 필요가 있다.

### 2.4 오차 크기에 따른 Gradient 비교

| 손실 | 작은 오차 | 큰 오차 | 특징 |
|---|---|---|---|
| MSE | gradient가 0으로 선형 감소 | 계속 증가 | outlier 민감 |
| Smooth L1 | 이차 구간 | gradient 제한 | 안정적 회귀 |
| Wing | 작은 오차를 강하게 강조 | 선형 구간 | 미세 landmark 보정 |

손실 선택은 이름보다 실제 오차 범위와 parameter를 기준으로 해야 한다. 모든 오차가 한
구간에만 들어가면 piecewise loss를 사용하더라도 다른 구간의 장점은 얻지 못한다.

### 2.5 정규화 좌표와 Sigmoid 적용

코너 target은 $[0,1]$ 범위이다. 모델 raw output $z$에 sigmoid를 적용해
$\hat{y}=\sigma(z)$로 제한할 수 있다. `WingLoss(apply_sigmoid=True)`는 loss 내부에서
이 변환을 수행한다.

장점은 학습 출력과 inference postprocessor의 좌표 범위가 일치한다는 점이다. 단점은
$|z|$가 클 때 sigmoid derivative $\sigma(z)(1-\sigma(z))$가 작아져 gradient가 포화되는
것이다. 모델, loss, postprocessor 중 어느 위치에서 sigmoid를 적용하는지 하나의 계약으로
관리해야 한다.

### 2.6 Outlier와 서브픽셀 오차

서브픽셀 정밀도는 작은 coordinate error를 계속 줄여야 한다. Wing은 이를 직접 강조하고,
Smooth L1은 안정성을 제공한다. 그러나 label 자체가 mask에서 양자화되었거나 잘못
정렬되었다면 작은 오차를 강하게 학습하는 것이 label noise에 과적합하는 결과를 낳는다.
Loss parameter를 조정하기 전에 label 정밀도와 corner order를 먼저 검증해야 한다.

## 3. 분류 및 조밀 예측 손실

### 3.1 Binary Cross Entropy

이진 target $y\in\{0,1\}$과 확률 $p$의 BCE는 다음과 같다.

$$
\ell_{BCE}(p,y)=-y\log p-(1-y)\log(1-p)
$$

Mask의 각 pixel이나 grid objectness처럼 서로 독립적인 이진 판단에 사용한다. 기본 mean
reduction에서는 foreground가 매우 적으면 background 항이 전체 손실을 지배할 수 있다.

### 3.2 BCEWithLogits의 수치 안정성

확률을 직접 계산한 뒤 로그를 취하면 $p$가 0이나 1에 가까울 때 overflow 또는
`log(0)` 문제가 생길 수 있다. BCEWithLogits는 log-sum-exp 형태로 sigmoid와 BCE를
결합한다.

$$
\ell(z,y)=\max(z,0)-zy+\log(1+e^{-|z|})
$$

프로젝트의 `BCELoss`는 `binary_cross_entropy_with_logits`를 사용하므로 model은 sigmoid
이전 logit을 전달해야 한다.

### 3.3 Softmax Cross Entropy

서로 배타적인 $K$개 class에서 정답 class가 $t$라면 cross entropy는 다음과 같다.

$$
\ell_{CE}(z,t)=-\log\frac{e^{z_t}}{\sum_{k=1}^{K}e^{z_k}}
$$

Target은 one-hot float가 아니라 일반적으로 정수 class index이다. 프로젝트의 grid
detection은 양성 cell에서 네 corner class 중 하나를 분류할 때 사용한다.

### 3.4 Focal Loss

Focal Loss는 쉬운 sample의 BCE를 줄이고 어려운 sample에 집중한다.

$$
p_t=
\begin{cases}
p, & y=1 \\
1-p, & y=0
\end{cases}
$$

$$
\alpha_t=
\begin{cases}
\alpha, & y=1 \\
1-\alpha, & y=0
\end{cases}
$$

$$
\ell_{focal}=-\alpha_t(1-p_t)^\gamma\log p_t
$$

$\gamma=0$이면 alpha-balanced BCE와 같고, $\gamma$가 커질수록 이미 맞힌 sample의
기여가 빠르게 줄어든다. 프로젝트 기본값은 $\alpha=0.25$, $\gamma=2$이다.

### 3.5 Class Imbalance와 Hard Example

Dense grid에는 배경 cell이 양성보다 훨씬 많다. BCE 평균은 수많은 쉬운 배경으로도 작은
손실을 계속 누적한다. Focal modulation $(1-p_t)^\gamma$는 잘 맞힌 배경의 값을 거의
0으로 만들어 희소한 corner나 line center에 상대적으로 집중한다.

Alpha는 class 빈도만 보고 기계적으로 정하면 안 된다. 프로젝트 구현에서 양성 weight는
$\alpha$, 음성 weight는 $1-\alpha$이므로 기본 $\alpha=0.25$는 개별 양성 항의 weight를
음성보다 작게 둔다. Focal 원 논문의 detector 설정과 현재 target 밀도가 같은지 확인하고
검증해야 한다.

### 3.6 Gaussian Heatmap과 Soft Target

Heatmap target은 중심만 1이고 주변은 0과 1 사이인 Gaussian 값일 수 있다. MSE는 이를
연속 회귀 target으로 자연스럽게 처리한다. 일반 BCE도 soft target을 받을 수 있지만
그 값은 확률 label의 의미를 가진다.

현재 `FocalLoss`는 이진 target 전용 분기 대신 다음 연속 보간을 사용한다.

$$
p_t=py+(1-p)(1-y)
$$

$y\in(0,1)$인 Gaussian 주변에서도 계산은 가능하지만 CenterNet 계열의 modified focal
loss와 같은 수식은 아니다. Soft heatmap에 사용할 때는 target 정의와 loss 논문을 구분해야
한다.

## 4. 영역 중첩 손실

### 4.1 Dice Coefficient

두 집합 $A$, $B$의 Dice coefficient는 다음과 같다.

$$
\mathrm{Dice}(A,B)=\frac{2|A\cap B|}{|A|+|B|}
$$

완전 일치하면 1, 겹치지 않으면 0이다. Foreground 크기에 비례해 정규화되므로 background가
많은 segmentation에서도 ROI overlap을 직접 강조한다.

### 4.2 Soft Dice Loss

미분 가능한 확률 $p_i=\sigma(z_i)$와 target $y_i$를 사용하면 다음과 같다.

$$
\mathcal{L}_{Dice}
=1-\frac{2\sum_i p_iy_i+\epsilon}
{\sum_i p_i+\sum_i y_i+\epsilon}
$$

프로젝트 `DiceLoss`의 기본 $\epsilon$은 1이다. 분자와 분모 합은 batch와 모든 공간
원소 전체에서 계산된다.

### 4.3 BCE와 Dice의 상호 보완

BCE는 pixel별 local classification을 학습하고 Dice는 mask 전체 overlap을 학습한다.

$$
\mathcal{L}
=\lambda_{bce}\mathcal{L}_{BCE}
+\lambda_{dice}\mathcal{L}_{Dice}
$$

BCE는 경계 pixel 각각에 직접 gradient를 주지만 class imbalance에 민감하다. Dice는
foreground 비율에 덜 민감하지만 모든 pixel이 하나의 비율로 결합되어 local 오류 해석이
어렵다. 둘을 함께 사용하면 local 안정성과 global overlap을 결합할 수 있다.

### 4.4 빈 Mask와 Smoothing 상수

Prediction과 target이 모두 비어 있으면 Dice 분모가 0이 된다. $\epsilon$은 0으로 나누는
문제를 막고 빈 mask 일치에 합리적인 값을 준다. 그러나 $\epsilon$이 실제 foreground 합에
비해 너무 크면 작은 객체의 loss를 왜곡한다.

이 프로젝트는 항상 하나의 ROI가 있다는 전제를 사용하므로 정상 target은 빈 mask가 아니다.
빈 target이 나오면 smoothing으로 숨기기보다 데이터 변환 오류를 먼저 조사해야 한다.

### 4.5 Batch 단위와 Sample 단위 집계

현재 `DiceLoss`는 batch 전체의 intersection과 union을 먼저 합친다. 큰 ROI나 큰 batch
sample이 비율에 더 많은 영향을 줄 수 있다. Sample별 Dice를 계산한 뒤 평균하는 구현과
값이 다르다. Batch size가 실험마다 바뀌면 이 차이가 성능과 loss scale에 영향을 줄 수
있으므로 집계 단위를 고정한다.

## 5. 다중 손실 결합

### 5.1 가중합 목적 함수

여러 raw 출력을 가진 모델은 각 손실의 가중합으로 학습한다.

$$
\mathcal{L}_{total}=\sum_{k=1}^{K}\lambda_k\mathcal{L}_k
$$

Detection에서는 objectness, box, class를, segmentation에서는 BCE와 Dice를 결합한다.
각 $\lambda_k$는 단순한 중요도뿐 아니라 손실 scale과 gradient 크기를 맞추는 역할을 한다.

### 5.2 손실 크기와 Gradient Scale

두 손실 값이 같아도 parameter에 미치는 gradient norm은 다를 수 있다. 반대로 값이 크게
보이는 손실이 실제 update를 지배하지 않을 수도 있다. 가중치는 로그에 표시된 scalar만
보고 정하지 않고 다음을 함께 확인한다.

- 손실별 평균값과 변동
- shared backbone에 대한 gradient norm
- 각 output head의 학습 속도
- 최종 metric 변화

### 5.3 BaseLoss의 Weight

프로젝트의 각 `BaseLoss`는 `weight` 속성을 가진다. `compute_losses`가 반환한 raw loss에
`BaseWrapper.train_step`이 이 weight를 곱한다.

```python
loss = sum(
    self.losses[name].weight * value
    for name, value in losses.items()
)
```

예를 들어 `SmoothL1Loss(weight=5.0)`은 내부 forward 값은 그대로 기록하지만 backward에
사용되는 total에는 5배로 들어간다. 로그 값과 실제 가중 목적 함수를 구분해야 한다.

### 5.4 Masked Regression Loss

Grid나 heatmap의 모든 위치에 displacement 정답이 존재하는 것은 아니다. 양성 위치
$\mathcal{P}$에서만 회귀한다.

$$
\mathcal{L}_{disp}
=\frac{1}{|\mathcal{P}|D}
\sum_{i\in\mathcal{P}}\sum_{d=1}^{D}
\ell(\hat{v}_{i,d},v_{i,d})
$$

Mask를 곱한 뒤 전체 grid 원소 수로 평균하면 background가 많을수록 gradient가 작아진다.
분모는 양성 수와 displacement 차원으로 정규화해야 한다. 양성이 0인 batch는 분모를 최소
1로 제한하거나 해당 항을 생략한다.

### 5.5 Deep Supervision

GCN처럼 반복 출력 $Y^{(0)},\ldots,Y^{(T)}$을 내는 모델은 마지막 결과만 아니라 중간 결과도
감독할 수 있다.

$$
\mathcal{L}_{deep}=\sum_{t=0}^{T}\lambda_t
\mathcal{L}(Y^{(t)},Y^*)
$$

초기 head와 각 refinement 단계에 직접 gradient가 도달해 학습이 안정된다. 모든 단계를
같은 weight로 평균할지 뒤 단계에 큰 weight를 줄지는 별도 설계이다. 현재 GCN 구현은
모든 단계를 하나로 reshape해 같은 Smooth L1 평균을 적용한다.

### 5.6 학습 목적 함수와 로그 이름

로그 key는 의미를 정확히 나타내야 한다. 한 key 안에 여러 항을 합치면 어떤 항이
개선되었는지 알 수 없다. Line의 현재 `compute_losses`는 `center` key로 focal center와
masked displacement의 합을 반환하므로 backward 목적 함수에는 둘 다 들어간다. 그러나
`BaseLoss` 누적기는 focal loss를 호출할 때만 갱신되므로 epoch 로그의 `center` 값에는
displacement가 포함되지 않는다. 같은 이름이 학습 목적 함수와 로그에서 서로 다른 값을
뜻하는 상태이다.

진단이 필요한 모델은 다음 값을 별도 key로 누적하는 편이 좋다.

```text
center
disp
total
```

단, `compute_losses` 반환 key와 `self.losses` key가 일치해야 하는 현재 공통 계약을 유지하며
설계해야 한다.

## 6. 출력 표현별 손실 선택

### 6.1 직접 좌표 회귀

`direct`, `doc`, `foundation`은 8개 coordinate logit을 출력한다. 현재 Wing Loss가 내부
sigmoid를 적용해 정규화 좌표와 비교한다. 미세 오차 강조가 목적이지만 $w$와 $\epsilon$이
정규화 scale에 맞는지 검증해야 한다.

### 6.2 Homography Offset 회귀

`homography`는 canonical corner에서의 제한된 offset을 예측한다. Wrapper가 raw output에
$\alpha\tanh$를 적용한 뒤 Smooth L1로 target offset과 비교한다. 출력 제한과 0 초기화는
학습 시작점이 canonical quad가 되게 한다.

### 6.3 Heatmap 예측

`heatmap`은 네 Gaussian heatmap을 MSE로 회귀한다. 출력에 sigmoid를 적용하지 않고 raw
값과 Gaussian target을 직접 비교한다. Postprocessor에서는 별도로 spatial softmax를
사용하므로 학습 표현과 좌표 추출 표현이 다르다는 점을 이해해야 한다.

### 6.4 Segmentation Mask

`seg`, `hybrid`, `torchseg`는 BCE와 Dice를 같은 weight로 결합한다. BCE는 pixel별 mask
logit을, Dice는 batch 전체 overlap을 학습한다. 두 손실 모두 내부에서 필요할 때 sigmoid를
처리하므로 model은 raw mask logit을 반환한다.

### 6.5 Grid Detection

`det`은 다음 세 항을 사용한다.

| 항 | 손실 | 적용 위치 | 기본 weight |
|---|---|---|---:|
| objectness | Focal | 모든 grid cell | 1 |
| box | Smooth L1 | 양성 cell | 5 |
| class | Cross Entropy | 양성 cell | 1 |

양성 cell이 없는 batch에서는 box와 class 항을 반환하지 않는다. Objectness는 항상 학습된다.

### 6.6 GCN 반복 정제

`gcn`은 초기 좌표와 세 번의 refinement 출력을 모두 target과 비교한다. 모든 단계에 같은
Smooth L1을 적용하는 deep supervision이다. 초기 오차와 마지막 미세 오차의 scale이 다를
수 있으므로 단계별 weight를 분리하는 ablation이 가능하다.

### 6.7 Line Segment 표현

`line`은 중심 heatmap에 Focal Loss, 양성 중심의 endpoint displacement에 masked Smooth
L1을 적용한다. 최종 corner는 비미분 가능한 peak selection, segment grouping, line fitting과
intersection을 거쳐 만들어지므로 학습 손실과 Polygon IoU 사이의 간극이 크다. Target
segment 길이, postprocessor 최소 길이와 channel 규약이 일치하지 않으면 loss가 감소해도
최종 성능이 악화될 수 있다.

### 6.8 Torchvision 모델의 내부 손실

`torchdet`은 Faster R-CNN, RetinaNet 또는 SSD가 반환하는 loss dictionary를 합산한다.
이 손실은 프로젝트의 7개 재사용 `BaseLoss` 클래스가 아니라 torchvision detector 내부에서
계산된다. `TorchdetWrapper`의 `BaseLoss` 인스턴스는 forward를 호출하는 실제 손실 함수가
아니라 외부에서 계산된 합계를 epoch 단위로 누적하는 용도로만 사용된다.

## 7. 학습 실패와 진단

### 7.1 중복 Sigmoid와 잘못된 Logit 처리

다음 조합은 잘못된 예이다.

```python
prob = torch.sigmoid(raw_output)
loss = binary_cross_entropy_with_logits(prob, target)
```

두 번째 함수가 `prob`를 다시 logit으로 해석해 sigmoid를 또 적용한다. Model, loss,
postprocessor별로 raw output 계약을 표로 관리하고 sigmoid 위치를 하나로 고정한다.

### 7.2 Target Scale 불일치

Prediction은 pixel 단위인데 target은 $[0,1]$이거나, model output은 512 grid 기준인데
target은 224 grid 기준이면 회귀 loss가 의미를 잃는다. Shape가 같아도 단위가 다를 수
있으므로 다음을 함께 검사한다.

- tensor shape
- 값의 최소, 평균, 최대
- 좌표 단위와 stride
- corner 순서
- sigmoid 또는 tanh 적용 여부

### 7.3 Reduction과 Weight 오류

이미 mean reduction된 손실을 다시 원소 수로 나누거나, weight를 loss 내부와 Wrapper에서
두 번 곱하면 gradient가 의도보다 작아진다. 반대로 `sum`을 batch 크기로 나누지 않으면
batch가 클수록 update가 커진다. 각 loss의 reduction 위치를 한 번만 정의한다.

### 7.4 Mask 분모와 희소 Target

Masked loss에서 `mask.sum()`이 0이면 NaN이 발생할 수 있다. 분모를 clamp하는 것만으로
계산은 유지되지만 양성이 없는 이유를 숨길 수 있다. 정상 데이터에서 양성이 반드시 있어야
한다면 count를 로그하고 0인 batch를 데이터 오류로 처리한다.

### 7.5 NaN과 Gradient 폭발

NaN의 주요 원인은 0으로 나누기, `log(0)`, 잘못된 intersection, 지나치게 큰 logit과
유효하지 않은 target이다. 다음 순서로 찾는다.

1. raw output과 target의 유한성을 검사한다.
2. 각 loss 항을 분리해 유한성을 검사한다.
3. backward 후 parameter별 gradient norm을 검사한다.
4. 첫 NaN batch의 입력과 label을 저장해 재현한다.

### 7.6 Loss는 감소하지만 Metric은 악화되는 경우

가능한 원인은 다음과 같다.

- 손실 target과 postprocessor의 규칙이 서로 다르다.
- 여러 손실 중 쉬운 항만 빠르게 감소한다.
- 평균 loss가 작은 오류 다수를 강조하고 metric은 최악 코너에 민감하다.
- postprocessor threshold 근처에서 작은 logit 변화가 검출 성공을 뒤집는다.
- label 순서나 좌표 scale이 잘못되었다.
- validation metric이 실패 sample을 제외해 조건부 평균만 기록한다.

정답 raw target을 postprocessor에 직접 넣는 oracle test는 target과 decoder 호환성을
검사하는 강력한 방법이다. Oracle조차 좋은 metric을 내지 못하면 optimizer보다 데이터
표현과 후처리를 먼저 수정해야 한다.

### 7.7 손실 항별 로그 분리

다중 task 모델은 최소한 raw 손실별 epoch 평균과 실제 weight를 기록한다. 가능하면 total
weighted loss도 별도로 기록한다. Loss별 gradient norm까지 주기적으로 기록하면 scale
불균형을 빠르게 찾을 수 있다.

## 8. 결론

손실 함수는 출력 표현과 target 의미에 맞추어 선택해야 한다. MSE는 연속 heatmap의 단순한
기준이고, Smooth L1은 강건한 좌표와 box 회귀에 적합하며, Wing은 작은 landmark 오차를
강조한다. BCE와 Cross Entropy는 이진 또는 배타적 class를, Focal은 희소한 hard example을,
Dice는 foreground overlap을 다룬다.

이름만으로 손실을 선택해서는 안 된다. 좌표 scale, piecewise 임계값, sigmoid 위치,
reduction 축과 다중 손실 weight가 실제 gradient를 결정한다. 또한 학습 target과 최종
postprocessor 사이의 정합성이 loss 자체보다 중요할 수 있다. Loss와 metric을 함께 기록하고
oracle, gradient, shape 검사를 수행해야 재현 가능한 코너 검출 학습이 된다.

## 참고 자료

- Huber, P. J. "Robust Estimation of a Location Parameter." Annals of Mathematical Statistics, 1964.
- Feng, Z. H., Kittler, J., Awais, M., Huber, P., Wu, X. J. "Wing Loss for Robust Facial Landmark Localisation with Convolutional Neural Networks." CVPR, 2018.
- Lin, T. Y., Goyal, P., Girshick, R., He, K., Dollar, P. "Focal Loss for Dense Object Detection." ICCV, 2017.
- Milletari, F., Navab, N., Ahmadi, S. A. "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation." 3DV, 2016.
- Goodfellow, I., Bengio, Y., Courville, A. "Deep Learning." MIT Press, 2016.
- PyTorch. "Loss Functions" and `torch.nn.functional` documentation.

## 부록 A. 프로젝트 Loss 인터페이스

### A.1 BaseLoss 생명주기

`BaseLoss`는 미분 가능한 batch loss 계산과 epoch 누적을 함께 담당한다.

```text
reset
    $\to$ batch forward
    $\to$ update(loss.item(), count)
    $\to$ repeated batches
    $\to$ compute epoch mean
```

`__call__`은 subclass의 `forward`를 실행하고 `len(target)`을 sample count로 사용한다.
일반 batch tensor에서는 첫 축 크기지만, masked positive tensor에서는 양성 원소 수가 된다.
따라서 서로 다른 loss 로그는 서로 다른 count 기준으로 누적될 수 있다.

### A.2 reset, update, compute

`reset()`은 epoch 시작 시 `total`과 `count`를 0으로 만든다. `update(value,count)`는 batch
평균에 count를 곱해 누적하고 `compute()`는 전체 count로 나눈다.

$$
\overline{L}_{epoch}
=\frac{\sum_b L_b n_b}{\sum_b n_b}
$$

마지막 batch가 작아도 sample-weighted 평균을 얻는 구조이다.

### A.3 losses Dictionary Key 계약

Wrapper의 `compute_losses` 반환 dictionary key는 `self.losses` key와 일치해야 한다.
`BaseWrapper.train_step`이 반환 key로 loss 객체와 weight를 찾기 때문이다.

```python
self.set_losses({
    "bce": BCELoss(),
    "dice": DiceLoss(),
})

def compute_losses(self, raw_output, targets):
    target = self.preprocessor(targets)
    return {
        "bce": self.losses["bce"](raw_output, target),
        "dice": self.losses["dice"](raw_output, target),
    }
```

### A.4 BaseWrapper의 가중합 계산

`BaseWrapper`는 반환된 항만 합산한다. 각 값에 등록된 loss 객체의 `weight`를 곱한 뒤
backward한다. Epoch 로그의 개별 값에는 이 외부 weight가 반영되지 않는다.

### A.5 Custom Loss 주입 규칙

Custom loss는 다음 조건을 만족해야 한다.

- `BaseLoss`를 직접 상속한다.
- `forward(raw_output,target)`이 scalar tensor를 반환한다.
- 기본 reduction이 batch 간 비교 가능한 값이어야 한다.
- `weight`는 `BaseLoss.__init__`으로 전달한다.
- `compute_losses` key와 등록 key를 일치시킨다.
- loss 내부에서 `.detach()`나 `.item()`으로 gradient를 끊지 않는다.

## 부록 B. 구현된 손실 함수

### B.1 BCELoss

`binary_cross_entropy_with_logits`의 기본 mean reduction을 사용한다. 입력은 mask 또는
objectness logit, target은 같은 shape의 float tensor이다.

### B.2 CrossEntropyLoss

`cross_entropy`를 사용한다. 입력 마지막 class 축은 logit이고 target은 integer class
index이다. Grid detector에서는 양성 cell만 선택한 뒤 호출한다.

### B.3 DiceLoss

내부 sigmoid 후 batch 전체 intersection과 union을 계산한다. 기본 $\epsilon=1$이고
scalar Dice loss 하나를 반환한다.

### B.4 FocalLoss

BCEWithLogits를 원소별로 계산하고 alpha와 $(1-p_t)^\gamma$를 곱한 뒤 전체 평균한다.
기본값은 $\alpha=0.25$, $\gamma=2$이다.

### B.5 MSELoss

`mse_loss` 기본 mean reduction을 사용한다. 현재 heatmap regression에 적용된다.

### B.6 SmoothL1Loss

`smooth_l1_loss`의 `beta`를 노출한다. 기본 $\beta=1$이며 homography, GCN, detection box에
사용된다.

### B.7 WingLoss

선택적으로 raw output에 sigmoid를 적용하고 Wing 수식을 평균한다. 기본값은
`apply_sigmoid=False`, $w=10$, $\epsilon=2$이다. 좌표 wrapper는
`apply_sigmoid=True`로 생성한다.

## 부록 C. 방법론별 손실 구성

### C.1 Direct, Doc, Foundation

세 방법론 모두 8개 coordinate logit과 정규화 좌표에
`WingLoss(apply_sigmoid=True)`를 사용한다.

### C.2 Homography와 GCN

Homography는 제한된 canonical offset에 Smooth L1을 사용한다. GCN은 초기 및 반복 좌표
전체를 Smooth L1으로 deep supervision한다.

### C.3 Heatmap

네 Gaussian target heatmap과 raw heatmap 출력에 MSE를 적용한다.

### C.4 Seg, Hybrid, Torchseg

세 방법론 모두 같은 mask target에 BCE와 Dice를 적용하고 같은 기본 weight로 합산한다.

### C.5 Det와 Torchdet

`det`은 Focal objectness, weight 5 Smooth L1 box, Cross Entropy class를 사용한다.
`torchdet`은 torchvision detector가 생성한 모델별 내부 loss dictionary를 합산한다.

### C.6 Line

Line은 M-LSD center channel의 Focal Loss와 endpoint displacement의 masked Smooth L1을
합산한다. Backward에는 두 항이 들어가지만 현재 `center` epoch 로그는 Focal Loss
누적값만 나타낸다. 상세 진단과 재현 가능한 total loss 기록을 위해 항 분리가 필요하다.

## 부록 D. 검증 방법

### D.1 알려진 입력에 대한 수식 검증

Prediction과 target이 같은 경우, 일정한 오차인 경우, 극단적인 logit인 경우의 기대값을
손으로 계산해 구현과 비교한다. Piecewise loss는 임계값 바로 아래, 정확히 임계값, 바로
위에서 연속성을 검사한다.

### D.2 Gradient 유한성 검사

각 loss에 `requires_grad=True` 입력을 주고 backward한 뒤 모든 gradient가 finite인지
확인한다. 정답과 같은 지점, 큰 양수와 음수 logit, 빈 mask와 희소 mask를 포함한다.

### D.3 Weight 가중합 검사

고정된 두 scalar loss와 서로 다른 weight를 사용해 `BaseWrapper` total이 수식과 일치하는지
검사한다. 개별 누적 로그에는 weight가 곱해지지 않는 현재 계약도 함께 확인한다.

### D.4 Masked Loss 검사

Mask 밖 prediction을 크게 바꾸어도 displacement loss가 변하지 않아야 한다. 양성 count가
0, 1, 여러 개인 경우 분모와 gradient scale을 검사한다.

### D.5 Logit과 확률 입력 오류 검사

BCEWithLogits와 직접 sigmoid+BCE가 수치적으로 같은지 확인하고, sigmoid를 적용한 값을
BCEWithLogits에 잘못 넣었을 때 결과가 달라지는 대조 테스트를 둔다. Wing Loss의
`apply_sigmoid`와 postprocessor sigmoid도 중복 적용 여부를 확인한다.
