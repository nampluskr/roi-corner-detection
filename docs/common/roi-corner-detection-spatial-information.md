# ROI Corner Detection의 공간 정보 손실과 보존

## 초록

코너 검출은 이미지에 무엇이 있는지만 판단하는 문제가 아니라 그 대상의 경계가 어느
위치에서 만나는지 추정하는 문제이다. CNN과 Vision Transformer의 중간 특징은 채널 축과
공간 축을 함께 가지며, 공간 축은 각 특징이 입력의 어느 영역에서 관측되었는지를 나타낸다.
Global Average Pooling은 이 격자를 채널별 평균 하나로 줄이기 때문에 최종 head가 위치별
활성값을 직접 구분할 수 없게 한다. 반면 strided convolution, heatmap, encoder-decoder,
grid detection과 patch-token 재배열은 해상도를 줄이더라도 위치별 순서를 유지할 수 있다.
이 문서는 공간 정보의 의미, 차원 감소와 정보 파괴의 차이, 연산별 특성과 계산 비용을
설명한다. 또한 현재 프로젝트의 12개 방법론이 공간 정보를 어떤 경로로 최종 좌표까지
전달하는지 분석하고, 공정한 head ablation 절차를 제시한다.

## 1. 서론

### 1.1 코너 검출이 위치 추정 문제인 이유

이미지 분류의 목표는 이미지 전체에 속한 범주 $y$를 예측하는 것이다. 고양이가 왼쪽에
있든 오른쪽에 있든 범주는 고양이로 같으므로 위치 변화에 둔감한 표현이 유리할 수 있다.
반면 코너 검출은 다음 좌표를 직접 예측한다.

$$
Y = \{(x_1,y_1),(x_2,y_2),(x_3,y_3),(x_4,y_4)\}
$$

같은 OLED가 이미지 왼쪽으로 이동하면 모든 $x_k$도 이동해야 한다. 따라서 입력 위치
변화가 출력 위치 변화로 이어지는 표현이 필요하다. 코너의 모양을 인식하는 것과 그 모양이
발생한 위치를 보존하는 일이 동시에 중요하다.

### 1.2 분류와 좌표 회귀의 차이

분류 모델은 공간 위치를 요약한 전역 벡터로도 높은 성능을 낼 수 있다.

$$
I \mapsto F \in \mathbb{R}^{C\times H\times W}
\mapsto z \in \mathbb{R}^{C}
\mapsto \text{class logits}
$$

좌표 회귀는 같은 전역 벡터를 사용할 수 있지만, 공간 격자를 벡터로 줄이는 순간 위치가
간접적으로만 전달된다. 이 때문에 분류 backbone의 마지막 구조를 그대로 재사용하는 것이
항상 최선은 아니다.

### 1.3 공간 정보의 정의

이 문서에서 공간 정보란 특징 값과 입력 위치 사이의 대응 관계를 뜻한다. 특징 맵
$F\in\mathbb{R}^{C\times H\times W}$에서 $F_{c,i,j}$는 채널 $c$가 표현하는 패턴이
격자 위치 $(i,j)$에서 얼마나 강하게 나타났는지를 나타낸다.

- 채널 축 $C$: 무엇과 비슷한 특징인가
- 공간 축 $H\times W$: 그 특징이 어디에서 관측되었는가

공간 정보를 보존한다는 것은 입력 픽셀 하나하나를 그대로 유지한다는 뜻이 아니다.
해상도를 줄이더라도 출력의 각 위치가 서로 다른 입력 영역에 대응하고, 그 위치 순서를
후속 연산이 구분할 수 있으면 공간 구조가 보존되었다고 본다.

## 2. CNN 특징 맵과 위치 표현

### 2.1 채널 축과 공간 축

224x224 입력을 ResNet50에 넣으면 마지막 합성곱 특징은 일반적으로
$(2048,7,7)$이다. 2048개 채널은 다양한 패턴 표현이고 7x7은 입력을 나눈 공간 격자이다.
격자 한 칸은 단순히 32x32 픽셀만 보는 것이 아니라 여러 합성곱을 거쳐 넓어진 receptive
field를 가진다. 그럼에도 7x7의 서로 다른 index는 서로 다른 중심 위치에 대응한다.

예를 들어 같은 채널에서 다음 두 특징 맵은 평균이 같지만 위치 의미는 다르다.

```text
top-left activation          bottom-right activation

9 0 0                        0 0 0
0 0 0                        0 0 0
0 0 0                        0 0 9
```

좌상단 코너를 찾는 문제에서는 이 차이가 결정적이다.

### 2.2 입력 좌표와 특징 Grid의 대응

전체 stride가 $s$인 CNN에서 특징 맵 위치 $(i,j)$의 중심은 대략 입력 좌표
$(sj,si)$ 근방에 대응한다. padding과 kernel 크기를 포함한 정확한 receptive field 중심은
레이어별로 계산해야 하지만, 규칙적인 격자 대응은 유지된다.

입력 크기가 $(H_0,W_0)$이고 특징 크기가 $(H_f,W_f)$이면 정규화 좌표는 대략 다음처럼
연결된다.

$$
x \approx \frac{j+\delta_x}{W_f},
\qquad
y \approx \frac{i+\delta_y}{H_f}
$$

$\delta_x,\delta_y$는 cell 내부 offset이다. heatmap과 grid detection은 이 관계를 출력
표현에 직접 사용한다.

### 2.3 합성곱의 공간 등변성

합성곱은 같은 kernel을 모든 위치에 적용한다. 경계 효과를 무시하면 입력을 이동했을 때
특징 맵도 같은 방향으로 이동하는 translation equivariance를 가진다.

$$
f(T_{\Delta}I) \approx T_{\Delta}f(I)
$$

$T_{\Delta}$는 위치 이동 연산이다. 이 성질은 코너가 이동하면 heatmap peak도 이동해야
하는 dense prediction에 유리하다. stride, pooling, padding은 완전한 등변성을 약화하지만
격자 대응 자체를 없애지는 않는다.

### 2.4 Receptive Field

깊은 특징의 한 cell은 입력의 넓은 영역을 본다. 넓은 receptive field는 한 점의 국소
모양뿐 아니라 OLED 전체 사각형의 배치와 문맥을 판단하게 해준다. 그러나 receptive field가
넓다는 사실과 공간 위치가 사라졌다는 것은 다르다. 각 cell이 넓은 영역을 보더라도 cell
index가 남아 있으면 서로 다른 중심 위치를 구분할 수 있다.

코너 검출에는 두 성질이 함께 필요하다.

1. 넓은 문맥으로 네 코너가 같은 ROI에 속하는지 판단한다.
2. 충분한 공간 해상도로 경계 교점의 위치를 정밀하게 찾는다.

### 2.5 Downsampling과 해상도

stride 2 연산은 보통 $H\times W$를 절반으로 줄인다. 이 과정은 고주파 세부 정보를
일부 잃지만 각 출력 cell이 서로 다른 입력 구역에 대응하므로 위치 순서는 남는다.

$$
(C,H,W) \xrightarrow{\text{stride 2}} (C',\lceil H/2\rceil,\lceil W/2\rceil)
$$

따라서 downsampling은 공간 정밀도를 낮출 수 있지만 곧바로 공간 정보 전체를 파괴하지는
않는다. 정보 손실의 정도는 kernel, stride, anti-aliasing, 채널 수와 후속 head에 따라
달라진다.

## 3. 공간 정보가 손실되는 과정

### 3.1 Global Average Pooling

Global Average Pooling, GAP는 각 채널의 모든 공간 값을 평균한다.

$$
z_c = \frac{1}{HW}\sum_{i=1}^{H}\sum_{j=1}^{W}F_{c,i,j}
$$

입력 shape은 $(C,H,W)$이고 출력은 $(C)$이다. 7x7 특징이라면 채널당 49개 위치가 숫자
하나로 합쳐진다. GAP 이후의 FC는 $z_c$만 받으므로 어느 cell이 크게 활성화되었는지
직접 알 수 없다.

### 3.2 GAP의 공간 순열 불변성

특징 맵의 공간 위치를 임의 순열 $\pi$로 재배치해도 평균은 같다.

$$
\frac{1}{HW}\sum_{i,j}F_{c,i,j}
=
\frac{1}{HW}\sum_{i,j}F_{c,\pi(i,j)}
$$

따라서 GAP head는 자신이 입력받는 최종 특징 맵에서 좌상단 활성과 우하단 활성을
구분할 수 없다. 이 성질은 객체 존재 여부를 묻는 분류에는 유리할 수 있지만 좌표를
묻는 회귀에는 구조적 병목이다.

### 3.3 Flatten과 GAP의 차이

Flatten은 평균을 내지 않고 위치 순서대로 값을 펼친다.

$$
F\in\mathbb{R}^{C\times H\times W}
\xrightarrow{\mathrm{flatten}}
v\in\mathbb{R}^{CHW}
$$

$F_{c,0,0}$과 $F_{c,H-1,W-1}$은 서로 다른 vector index가 된다. 뒤의 Linear layer는
두 위치에 다른 weight를 적용할 수 있으므로 공간 배치를 구분한다.

Flatten 자체는 정보를 섞거나 평균하지 않지만 vector 길이가 $CHW$로 커진다. 고해상도
특징을 그대로 flatten하면 FC parameter와 과적합 위험이 커지므로 보통 convolution으로
공간 및 채널 크기를 먼저 줄인다.

### 3.4 차원 감소와 공간 파괴의 구분

차원이 감소한다고 항상 공간 정보가 사라지는 것은 아니다.

```text
(2048, 7, 7)
    $\to$ stride convolution
(128, 4, 4)
    $\to$ stride convolution
(64, 2, 2)
    $\to$ flatten
(256)
```

이 경로는 입력 값의 수를 크게 줄이지만 마지막 2x2의 네 위치가 서로 다른 vector index로
남는다. 반면 GAP는 어느 해상도에서 실행하든 공간 축 전체를 하나로 만든다.

차원 감소를 평가할 때는 다음을 묻는다.

1. 출력에 둘 이상의 공간 위치가 남아 있는가
2. 위치별 값이 후속 연산에서 서로 다른 index로 전달되는가
3. pooling 범위가 국소 영역인가 전체 영역인가
4. 해상도 감소 전에 필요한 위치 단서를 채널로 표현할 용량이 있는가

### 3.5 Vector Bottleneck

공간 축을 제거한 vector를 만든 뒤에는 후속 FC나 MLP가 원래 격자를 직접 복원할 수 없다.
동일한 vector로 압축된 여러 특징 맵은 어떤 후속 결정적 함수에서도 구분되지 않는다.

$$
g(F_1)=g(F_2) \Rightarrow h(g(F_1))=h(g(F_2))
$$

$g$가 GAP이고 $h$가 FC head라면 GAP에서 같아진 두 입력은 head가 아무리 커도 구분할 수
없다. 이를 vector bottleneck이라고 한다.

### 3.6 위치 정보의 채널 간접 부호화

GAP가 있다고 해서 실제 모델의 위치 정보가 항상 완전히 0이 되는 것은 아니다. 학습 가능한
backbone은 padding, 이미지 경계, 넓은 receptive field를 이용해 위치별 패턴을 서로 다른
채널에 미리 부호화할 수 있다. 예를 들어 한 채널은 좌상단 코너에만 반응하고 다른 채널은
우하단 코너에만 반응하도록 학습할 수 있다. 그러면 GAP 뒤에도 채널 값의 조합으로 위치를
간접 추론할 수 있다.

그러나 이는 다음 비용을 가진다.

- backbone이 좌표 task에 맞게 충분히 학습되어야 한다.
- 위치마다 별도 채널 표현을 사용해 표본 효율이 낮을 수 있다.
- 최종 격자를 직접 쓰는 head보다 구조적 보장이 약하다.
- 입력 크기나 배치가 바뀌면 간접 부호화가 불안정할 수 있다.

따라서 GAP를 공간 정보의 절대적 삭제라고 표현하기보다는, head가 공간 격자에 직접
접근하는 경로를 삭제한다고 표현하는 것이 정확하다.

### 3.7 동결 Backbone에서 GAP가 위험한 이유

backbone이 동결되면 목표 코너 위치를 채널에 새로 부호화하도록 적응할 수 없다. 분류나
자기지도 학습에서 얻은 채널이 우연히 위치 단서를 담고 있어야만 GAP head가 이를 사용할
수 있다. 따라서 frozen backbone에서는 patch grid나 CNN feature grid를 보존하는 head가
더 중요하다.

동결된 DINOv2의 patch token을 평균 하나로 만들기보다 2차원 grid로 재배열해 convolution과
flatten을 적용하는 이유가 여기에 있다. 자세한 동결 및 해제 전략은
[Warmup 학습과 단계적 Fine-tuning](roi-corner-detection-warmup-training.md)을 참조한다.
공간 출력별 학습 신호의 차이는
[손실 함수의 원리와 적용](roi-corner-detection-loss-functions.md)을 참조한다.

## 4. 공간 정보를 보존하는 구조

### 4.1 Grid 유지 후 Flatten

가장 단순한 공간 보존 회귀 head는 특징 맵을 작은 grid로 축소한 뒤 flatten한다.

$$
\hat{Y}=mathrm{FC}\bigl(\mathrm{Flatten}(\mathrm{ConvDown}(F))\bigr)
$$

이 구조는 최종 좌표를 8개 숫자로 출력하면서도 위치별 특징을 FC까지 전달한다. 단점은
입력 grid 크기에 따라 FC 입력 차원이 달라진다는 점이다. Adaptive pooling으로 고정된
$K\times K$ grid를 만들 수 있지만 $K=1$이면 다시 GAP와 같아진다.

### 4.2 Strided Convolution

Strided convolution은 국소 이웃을 학습 가능한 weight로 결합하면서 해상도를 줄인다.

$$
Y_{c',i,j}
=\sum_c\sum_{u,v}W_{c',c,u,v}X_{c,si+u,sj+v}
$$

평균 pooling과 달리 어떤 패턴을 남길지 학습할 수 있고 출력 위치 $(i,j)$도 유지된다.
다만 stride보다 높은 공간 주파수는 aliasing되거나 사라질 수 있으므로 필요한 최종 정밀도에
맞는 grid 크기를 유지해야 한다.

### 4.3 Heatmap 기반 예측

Heatmap 모델은 코너별 공간 지도 $H_k$를 출력한다.

$$
H\in\mathbb{R}^{4\times H_h\times W_h}
$$

soft-argmax를 사용하면 각 heatmap을 확률 분포로 바꾸고 좌표 기댓값을 계산한다.

$$
p_{k,i,j}=\frac{\exp(\beta H_{k,i,j})}
{\sum_{u,v}\exp(\beta H_{k,u,v})}
$$

$$
\hat{x}_k=\sum_{i,j}p_{k,i,j}\frac{j}{W_h-1},
\qquad
\hat{y}_k=\sum_{i,j}p_{k,i,j}\frac{i}{H_h-1}
$$

출력 자체가 위치 지도이므로 공간 보존이 가장 명시적이다.

### 4.4 Encoder-Decoder와 Skip Connection

Segmentation 모델은 낮은 해상도의 의미 특징을 decoder로 확대해 입력과 대응하는 mask를
만든다. 깊은 특징만 확대하면 세부 경계가 사라질 수 있으므로 encoder의 중간 특징을 같은
해상도 decoder와 연결한다.

$$
D_l = \mathrm{Conv}\bigl([\mathrm{Up}(D_{l+1}), E_l]\bigr)
$$

$E_l$은 encoder skip, $D_l$은 decoder 특징이다. skip은 얕은 층의 정밀한 위치와 깊은
층의 넓은 문맥을 결합한다.

### 4.5 Grid Detection

Grid detector는 특징 맵의 각 cell에서 objectness, class와 cell 내부 offset을 예측한다.

$$
x=\frac{j+\sigma(t_x)}{W_g},
\qquad
y=\frac{i+\sigma(t_y)}{H_g}
$$

cell index $(i,j)$가 거친 위치를, offset이 cell 내부의 세부 위치를 표현한다. Flatten으로
anchor 축을 만들더라도 decoder가 원래 row와 column을 복원하면 공간 정보가 유지된다.

### 4.6 grid_sample과 ROIAlign

초기 좌표가 있을 때 `grid_sample`은 해당 위치에서 특징 맵을 bilinear interpolation한다.
정규화 좌표 $(x,y)$를 $[-1,1]$ grid로 바꾸고 주변 네 cell의 가중합을 사용한다. GCN은
각 코너 위치의 특징을 이렇게 읽어 offset을 반복 보정한다.

ROIAlign도 proposal 영역을 일정 grid로 샘플링한다. 전체 이미지를 GAP하는 대신 관심
영역의 공간 배치를 유지하므로 detection head가 위치와 형태를 함께 사용할 수 있다.

### 4.7 Patch Token과 Positional Embedding

Vision Transformer는 이미지를 patch로 나누고 각 patch를 token으로 표현한다. 순수 attention은
token 순열을 자체적으로 구분하지 못하므로 positional embedding을 더한다.

$$
z_i=E\,\mathrm{flatten}(P_i)+e_i^{pos}
$$

최종 patch token의 순서를 원래 patch grid로 재배열하면 CNN 특징 맵처럼 사용할 수 있다.
prefix token은 제외하고 $N_p=H_pW_p$가 되는 patch token만 reshape해야 한다.

## 5. 연산별 공간 정보 특성

### 5.1 Strided Convolution

Strided convolution은 해상도와 세부 정보를 줄이지만 위치별 출력 grid를 유지한다. 공간
정보 보존 여부는 최종 grid 크기와 flatten 방식에 달려 있다. 연속된 stride 2 연산으로
1x1까지 줄이면 사실상 전역 vector가 된다.

### 5.2 Max Pooling과 Average Pooling

국소 pooling은 작은 window마다 값을 요약한다. Max pooling은 가장 강한 반응을, average
pooling은 평균 반응을 남긴다. 둘 다 window 내부의 정확한 위치는 잃지만 서로 다른 출력
window의 위치는 유지한다. Global pooling은 window가 전체 특징 맵이므로 공간 축 전체를
제거한다.

### 5.3 Adaptive Pooling

`AdaptiveAvgPool2d(K)`는 입력 크기와 관계없이 $K\times K$ 출력을 만든다.

- $K=1$: Global Average Pooling과 동일하다.
- $K>1$: 각 공간 구역의 평균을 별도 cell에 저장한다.
- 입력과 출력이 모두 $K\times K$: 실질적으로 grid 크기를 유지한다.
- 출력이 입력보다 크면 새로운 세부 정보를 만드는 것이 아니라 구간이 겹치거나 반복된다.

따라서 `AdaptiveAvgPool2d(4)`라는 이름에 average가 있어도 4x4 위치를 하나로 합치는 GAP는
아니다. 그러나 입력이 2x2인데 4x4로 출력해도 잃어버린 4x4 세부 정보가 복원되지는 않는다.

### 5.4 Interpolation과 Transposed Convolution

Interpolation은 기존 grid를 규칙적으로 확대한다. Bilinear interpolation은 연속적인 값을
만들지만 새로운 관측 정보를 추가하지 않는다. Transposed convolution은 학습 가능한
upsampling이지만 낮은 해상도에서 이미 사라진 경계를 저절로 되찾는 것은 아니다. skip
connection이나 얕은 특징이 정밀도를 보완한다.

### 5.5 Flatten과 Fully Connected Layer

Flatten은 위치를 vector index로 보존하지만 FC는 모든 위치를 자유롭게 섞는다. 이 때문에
입력 이동에 대한 등변성은 사라지고 입력 크기도 고정된다. 그러나 좌상단과 우하단에 서로
다른 weight를 줄 수 있어 전역 좌표 회귀에는 유용하다.

### 5.6 1x1 Convolution

1x1 convolution은 각 위치에서 채널만 선형 결합한다.

$$
Y_{c',i,j}=\sum_c W_{c',c}X_{c,i,j}+b_{c'}
$$

공간 이웃을 섞지 않고 $H\times W$를 그대로 유지하므로 채널 축소나 per-pixel class
logit 생성에 적합하다.

## 6. 공간 보존과 계산 비용

### 6.1 특징 맵 해상도와 메모리

특징 맵 메모리는 대략 $NCHW$에 비례한다. 해상도를 두 배로 하면 $H$와 $W$가 모두
두 배가 되어 activation 메모리는 약 네 배가 된다. backward에서는 gradient와 중간
activation도 보관하므로 실제 비용은 더 크다.

공간 보존은 가능한 최고 해상도를 무조건 유지하는 것이 아니라, 목표 오차와 계산 예산을
만족하는 최소 grid를 찾는 문제이다.

### 6.2 위치 정밀도와 문맥

얕은 고해상도 특징은 경계 위치가 정밀하지만 의미와 문맥이 약하다. 깊은 저해상도 특징은
ROI 전체를 잘 이해하지만 세부 위치가 거칠다. heatmap decoder와 U-Net skip은 두 정보를
결합한다. 직접 회귀의 공간 head는 깊은 grid의 순서를 보존하지만 얕은 경계를 직접 받지는
않으므로 정밀도에 상한이 있을 수 있다.

### 6.3 Upsampling의 한계

7x7 특징을 56x56으로 확대했다고 해서 56x56의 독립적인 관측을 얻는 것은 아니다.
학습 가능한 decoder는 7x7 문맥에서 가능성 높은 경계를 생성할 수 있지만, 원래 입력의
세부 edge는 skip이나 고해상도 branch가 없으면 간접적으로만 복원한다.

### 6.4 서브픽셀 정밀도

Argmax는 heatmap cell 중심에 좌표를 제한하지만 soft-argmax, offset regression, line
intersection은 연속 좌표를 낼 수 있다. 다만 출력이 연속이라는 사실만으로 입력 정보가
충분한 것은 아니다. 낮은 해상도 특징에서 나온 연속 좌표는 학습된 보간 결과이며 실제
서브픽셀 단서가 보존되었는지 별도 평가해야 한다.

### 6.5 구조 선택 기준

| 조건 | 적합한 구조 |
|---|---|
| 좌표 8개를 간단히 회귀 | 작은 grid + flatten + FC |
| 코너별 위치 분포가 중요 | heatmap + soft-argmax |
| ROI 전체 경계가 중요 | encoder-decoder + skip |
| 여러 후보 중 코너 선택 | grid 또는 proposal detection |
| 초기점 국소 보정 | grid_sample 또는 ROIAlign |
| frozen ViT 특징 재사용 | patch grid reshape + spatial head |

## 7. 결론

공간 정보는 특징 맵의 값과 입력 위치 사이의 대응 관계이다. GAP는 공간 전체를 채널별
평균 하나로 바꾸므로 head의 직접적인 위치 접근을 제거한다. 반면 strided convolution이나
국소 pooling은 해상도를 줄여도 여러 grid cell과 그 순서를 남길 수 있다. 그러므로
차원 감소와 공간 파괴를 같은 의미로 사용해서는 안 된다.

코너 검출에서는 위치 정밀도와 전역 문맥을 함께 확보해야 한다. 작은 grid를 flatten하는
회귀 head, heatmap, U-Net skip, grid detection, patch-token grid는 서로 다른 비용으로 이
조건을 만족한다. 최종 선택은 구조적 주장만으로 정하지 않고 같은 데이터와 parameter
예산에서 Polygon IoU와 corner distance를 비교해야 한다.

## 참고 자료

- He, K., Zhang, X., Ren, S., Sun, J. "Deep Residual Learning for Image Recognition." CVPR, 2016.
- Long, J., Shelhamer, E., Darrell, T. "Fully Convolutional Networks for Semantic Segmentation." CVPR, 2015.
- Ronneberger, O., Fischer, P., Brox, T. "U-Net: Convolutional Networks for Biomedical Image Segmentation." MICCAI, 2015.
- Newell, A., Yang, K., Deng, J. "Stacked Hourglass Networks for Human Pose Estimation." ECCV, 2016.
- Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." ICLR, 2021.
- Oquab, M., Darcet, T., Moutakanni, T., et al. "DINOv2: Learning Robust Visual Features without Supervision." 2023.

## 부록 A. 방법론별 공간 정보 분석

### A.1 Direct, Homography, Doc

`direct`는 `head_type="gap"`을 기본값으로 사용한다. 이 경로는 torchvision ResNet의
`avgpool`을 그대로 거쳐 vector를 만든 뒤 FC(8)을 적용한다. `head_type="spatial"`은
ResNet의 GAP 이전 특징 맵을 꺼내 strided convolution, adaptive 4x4 grid, flatten을
사용한다. 따라서 두 head를 직접 ablation할 수 있다.

`homography`와 `doc`의 현재 구현은 GAP 이전 ResNet 특징에 두 strided convolution과
`AdaptiveAvgPool2d(4)`, flatten, Linear(8)을 적용한다. 둘 다 grid 순서를 FC까지 전달한다.
다만 기본 224 입력의 ResNet50 특징은 7x7이고 두 stride 연산 뒤 2x2가 되므로 adaptive
4x4 출력은 잃은 세부 위치를 복원하지 않는다. 입력 및 head shape를 실제 실행으로 확인할
필요가 있다.

### A.2 Foundation과 GCN

`foundation`은 DINOv2의 prefix token을 제외하고 patch token을 정사각 grid로 reshape한다.
그 다음 strided convolution과 4x4 grid, flatten을 사용한다. backbone이 영구 동결되어도
patch 위치가 head에 직접 전달되므로 GAP보다 안전하다.

`gcn`은 공간 head로 초기 코너를 만들고, 원래 ResNet 특징 맵에서 각 코너 위치를
`grid_sample`로 읽어 반복 보정한다. 초기점과 refinement가 모두 공간 특징에 접근하지만
초기점이 크게 틀리면 국소 sampling도 잘못된 위치에서 시작할 수 있다.

### A.3 Heatmap, Segmentation, Hybrid

`heatmap`은 마지막 ResNet grid를 transposed convolution으로 확대해 코너별 heatmap을
출력한다. skip connection은 없으므로 최종 heatmap이 크더라도 세부 위치 정보는 깊은
특징에 의존한다.

`seg`와 `hybrid`는 여러 encoder stage를 decoder에 skip으로 연결한다. 낮은 해상도 문맥과
얕은 경계 정보를 함께 사용해 mask를 만든다. 두 모델은 같은 mask 구조를 사용하지만
postprocessor가 각각 contour 근사와 line 기반 고전 CV라는 차이가 있다.

### A.4 Det, Torchdet, Torchseg, Line

`det`은 ResNet grid마다 objectness, offset, box와 corner class를 예측한다. postprocessor가
선택된 cell의 row, column과 offset을 결합하므로 공간 구조가 유지된다.

`torchdet`의 Faster R-CNN, RetinaNet, SSD는 feature map, anchor, proposal 또는 ROIAlign을
통해 위치를 유지한다. `torchseg`는 입력과 대응하는 pixel logit을 출력한다. DeepLabV3의
ASPP처럼 일부 전역 pooling branch가 있어도 지역 convolution branch와 결합되므로 전체
표현이 전역 vector 하나로 붕괴하지 않는다.

`line`은 MobileNetV2의 여러 stage를 FPN 방식으로 결합하고 16채널 line map을 출력한다.
중심 heatmap과 endpoint displacement가 모두 공간 grid에 정의되므로 위치를 직접 보존한다.

## 부록 B. 공간 보존 Head 구현

### B.1 GAP 이전 특징 맵 추출

torchvision ResNet에서 `list(net.children())[:-2]`는 adaptive average pooling과 FC를
제외한다.

```python
self.backbone = nn.Sequential(*list(net.children())[:-2])
```

반대로 `net.fc = nn.Identity()`만 적용하면 ResNet forward 내부의 `avgpool`과 flatten은
계속 실행된다. 이 경우 backbone 출력은 이미 $(N,C)$ vector이다.

### B.2 Strided-conv Head

프로젝트의 공통 공간 head 형태는 다음과 같다.

```python
self.head = nn.Sequential(
    nn.Conv2d(in_channels, 128, kernel_size=3, stride=2, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1),
    nn.ReLU(inplace=True),
    nn.AdaptiveAvgPool2d(4),
    nn.Flatten(),
    nn.Linear(64 * 4 * 4, 8),
)
```

이 구조는 마지막 4x4 cell을 서로 다른 FC 입력으로 유지한다. 단, adaptive pooling 이전
특징이 4x4보다 작은 경우 정보가 늘어나는 것은 아니다.

### B.3 Tensor Shape 추적

forward hook 또는 작은 dummy input으로 다음 shape를 기록한다.

```text
input
    $\to$ backbone output
    $\to$ first stride convolution
    $\to$ second stride convolution
    $\to$ adaptive pooling
    $\to$ flatten
    $\to$ output
```

검증할 항목은 batch 축, 채널 축, 공간 축, FC 입력 차원이다. 지원하는 모든 image size와
backbone variant에서 reshape와 Linear 차원이 맞는지 확인한다.

### B.4 AdaptiveAvgPool2d 주의점

`AdaptiveAvgPool2d(4)`는 출력 크기를 4x4로 고정하지만 입력이 최소 4x4라는 조건을
자동으로 보장하지 않는다. 입력이 2x2여도 연산은 실행되지만 결과 4x4는 더 정밀한 관측이
아니다. 공간 head 설계 시 adaptive pooling 직전 크기를 먼저 계산하고, 필요하면 stride
수나 목표 grid 크기를 조정한다.

## 부록 C. 검증 및 Ablation

### C.1 GAP Head와 공간 보존 Head 비교

같은 pretrained checkpoint, 데이터 split, seed, optimizer와 epoch를 사용한다. direct의
`head_type`만 바꾸고 validation Polygon IoU, MCD, MaxCD를 비교한다. 단일 seed 최고값보다
여러 seed의 평균과 분산을 보고 판단한다.

### C.2 Parameter 수를 맞춘 비교

공간 head는 GAP+FC보다 parameter가 많을 수 있다. 향상이 공간 보존 때문인지 모델 용량
때문인지 분리하려면 채널 수를 조정해 총 parameter 수가 비슷한 대조군을 추가한다.
최소 비교는 다음 세 가지이다.

1. GAP + 작은 FC
2. GAP + parameter 수를 늘린 MLP
3. spatial grid + flatten + FC

### C.3 Backbone 동결 여부 비교

각 head를 frozen backbone과 trainable backbone에서 모두 비교한다. 예상되는 핵심 상호작용은
GAP head가 trainable backbone에서 위치를 채널에 간접 부호화할 가능성이 있지만 frozen
backbone에서는 그 적응이 불가능하다는 점이다. Warmup을 사용하는 실험은 동결 epoch와
전체 update 횟수까지 동일하게 기록한다.
