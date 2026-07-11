# Torchvision Segmentation Corner (torchseg)

## 초록

Torchvision Segmentation Corner는 코너 검출을 픽셀 단위 영역 분할로 우회한 뒤 마스크
윤곽에서 코너를 복원하는 세그멘테이션 계열 방법론이며, 분할 모델을 직접 설계하는 대신
검증된 세그멘테이션 아키텍처(FCN, DeepLabV3, LR-ASPP) 전체를 backbone과 밀집 예측
head까지 통째로 재사용한다는 점이 특징이다. 손으로 인코더-디코더를 조립하는 대신,
대규모 분할 데이터로 사전학습된 표준 구조를 그대로 가져와 마지막 분류 층만 단일 채널
(사각형 내부 확률)로 바꿔 학습한다. 학습 신호는 픽셀 수만큼 조밀하고, 마스크에서 코너를
얻는 후처리(이진화 + 컨투어 추출 + 폴리곤 근사)는 세그멘테이션 계열이 공유한다. 성숙한
아키텍처와 사전학습 가중치를 이식해 적은 노력으로 강한 초기 성능을 얻는 대신, 모델
구조의 자유도가 낮고 검출 head가 코너에 특화되지 않는다는 절충을 가진다.

## 1. 서론

### 1.1 배경: 검증된 분할 아키텍처를 통째로 재사용한다

세그멘테이션은 십수 년간 표준 아키텍처가 축적된 분야이다. 완전 합성곱 신경망(FCN),
공간 피라미드로 다중 스케일 문맥을 모으는 DeepLabV3, 모바일용 경량 head를 쓰는
LR-ASPP 등은 구조와 학습 방법이 광범위하게 검증되어 있고, 공개 대규모 분할 데이터로
사전학습된 가중치가 함께 제공된다. 코너 검출을 "사각형 영역 분할"로 바꾸면 이 자산을
그대로 활용할 수 있다. 즉 분할 모델을 새로 설계하는 대신, 완성된 분할망을 가져와 출력
채널만 1개로 교체하고 코너 도메인에 맞춰 미세조정한다.

이 접근의 이점은 두 가지이다. 첫째, 아키텍처 설계와 튜닝 비용이 거의 들지 않는다.
디코더 구조, 스킵 연결, 업샘플 방식 같은 선택을 라이브러리가 이미 최적화해 두었다.
둘째, 분할 과제로 사전학습된 가중치는 "물체 경계를 픽셀 단위로 가르는" 능력을 이미
가지고 있어, 코너 도메인 데이터가 적어도 빠르게 수렴한다. 대가는 구조를 문제에 맞춰
바꿀 자유도가 낮다는 점이다.

### 1.2 세그멘테이션은 픽셀 단위 분류다

분류는 입력 하나에 범주 하나를 할당하는 문제이고, 세그멘테이션은 그 분류를 모든
픽셀에 대해 동시에 수행하는 문제이다. "이 픽셀은 사각형 내부인가"라는 이진 분류를
$H \times W$번 푸는 것과 같으며, 출력은 각 픽셀이 내부일 확률을 담은 확률 지도이다.
밀집 예측(dense prediction)이 성립하려면 출력이 입력과 대응하는 공간 격자여야 하므로,
모델은 해상도를 끝까지 유지하거나 복원하는 완전 합성곱 구조여야 한다.

### 1.3 문제 정의

입력은 컬러 이미지 한 장이다. 중간 출력은 사각형 영역의 확률 지도이고, 최종 출력은
TL(좌상단) $\to$ TR(우상단) $\to$ BR(우하단) $\to$ BL(좌하단) 시계 방향 순서의 코너 4개 좌표로,
각 좌표는 이미지 크기로 나눈 [0, 1] 정규화 값이다. 파이프라인은 "학습되는 분할망"과
"학습되지 않는 기하 후처리"의 2단계 직렬 구성이다.

## 2. 이론 및 아키텍처

### 2.1 문제 정식화

입력 이미지를 $I \in \mathbb{R}^{3 \times H \times W}$, 정답 코너를
$Y \in [0, 1]^{4 \times 2}$로 둔다. 코너가 이루는 볼록 사각형 영역 $Q(Y)$로부터 정답
마스크 $M \in \{0, 1\}^{H \times W}$를 픽셀 중심의 내부 포함 여부로 정의한다(래스터화).

$$
M_{ij} =
\begin{cases}
1 & \text{픽셀 } (i, j) \text{의 중심이 } Q(Y) \text{ 내부} \\
0 & \text{그 외}
\end{cases}
$$

학습 대상은 입력과 같은 해상도의 확률 지도를 출력하는 함수이다.

$$
f_\theta: \mathbb{R}^{3 \times H \times W} \to [0, 1]^{H \times W}
$$

최종 코너는 학습 파라미터가 없는 후처리 $g$로 얻는다.

$$
\hat{Y} = g\big(\mathbb{1}[P \ge \tau]\big) \in [0, 1]^{4 \times 2},
\qquad P = f_\theta(I)
$$

$\mathbb{1}[\cdot]$은 지시 함수, $\tau$는 이진화 임계값(보통 0.5)이다. 이 정식화는
세그멘테이션 계열이 공유하며, torchseg의 고유성은 $f_\theta$를 어떤 구조로 실현하는가
(2.2절)에 있다.

### 2.2 모델 구조

torchvision 분할 모델은 공통적으로 backbone(특징 추출기)과 분할 head(밀집 예측기)의
직렬 연결이며, 최종적으로 입력 해상도의 확률 지도를 낸다. 세 계열을 차례로 본다.

**(1) 공통 토대: 완전 합성곱과 밀집 예측.** 분류망은 마지막에 전역 평균 풀링과
완전연결층으로 공간 격자를 하나의 벡터로 접는다. 완전 합성곱망은 그 접는 단계를 없애고
$1 \times 1$ 합성곱을 써서 위치마다 같은 분류기를 적용한다. $1 \times 1$ 합성곱은 각
위치의 채널 벡터에 동일한 선형 변환을 적용하는 연산이므로, 공간 격자를 유지한 채 픽셀별
점수 지도를 만든다. 마지막에 쌍선형 보간(bilinear interpolation, 이웃 값의 가중 평균)
으로 backbone에서 줄어든 해상도를 입력 크기까지 되살린다.

**(2) FCN.** 가장 단순한 형태로, ResNet backbone의 최종 특징 맵에 몇 개의 합성곱과
$1 \times 1$ 분류 합성곱을 얹고 입력 해상도로 업샘플한다. 별도의 다중 스케일 모듈이나
정교한 디코더 없이, "backbone 특징 $\to$ 픽셀 분류 $\to$ 업샘플"의 최소 구성으로 밀집 예측을
구현한다. 구조가 단순해 계산이 가볍지만, 넓은 문맥을 한 스케일에서만 보므로 경계
정밀도는 다음 두 계열보다 낮은 편이다.

**(3) DeepLabV3와 atrous 합성곱.** 밀집 예측의 핵심 난제는 "해상도를 유지하면서 넓은
문맥을 보는 것"이다. 풀링으로 해상도를 줄이면 문맥은 넓어지지만 위치가 거칠어진다.
atrous(dilated) 합성곱은 필터 원소 사이에 간격 $r$(dilation rate)을 두어, 파라미터
수와 해상도를 그대로 둔 채 수용 영역만 넓힌다. 1차원으로 쓰면 다음과 같다.

$$
y(i) = \sum_{k} x(i + r \cdot k)\, w(k)
$$

$r = 1$이면 보통 합성곱, $r$이 커지면 같은 필터가 더 넓은 간격의 입력을 참조한다.
DeepLabV3는 backbone 후반부의 stride를 atrous로 대체해 출력 해상도를 1/8 또는 1/16로
높게 유지한다. 그 위에 ASPP(Atrous Spatial Pyramid Pooling)를 얹는데, ASPP는 서로 다른
dilation rate(예: 6, 12, 18)의 atrous 합성곱 가지들과 $1 \times 1$ 합성곱 가지, 그리고
전역 평균 풀링으로 이미지 전체 문맥을 담는 가지를 병렬로 두고 채널 방향으로 이어 붙인다.

$$
\mathrm{ASPP}(F) = \mathrm{Conv}_{1 \times 1}\Big(\big[\,
A_{r_1}(F);\ A_{r_2}(F);\ A_{r_3}(F);\ \mathrm{Conv}_{1 \times 1}(F);\
\mathrm{Up}(\mathrm{GAP}(F))\,\big]\Big)
$$

여기서 $A_r$은 dilation $r$의 atrous 합성곱, $\mathrm{GAP}$는 전역 평균 풀링,
$[\cdot]$은 채널 연결이다. 여러 스케일의 문맥을 한 번에 결합하므로, 크기와 원근이
다양한 사각형에도 안정적으로 경계를 잡는다.

**(4) LR-ASPP와 경량 backbone.** LR-ASPP(Lite Reduced ASPP)는 MobileNetV3 backbone에
얹는 경량 분할 head로, 모바일/CPU 배포를 겨냥한다. 무거운 ASPP 대신 두 가지를 쓴다.
하나는 $1 \times 1$ 합성곱으로 채널을 정리하는 본선이고, 다른 하나는 전역 평균 풀링 후
$1 \times 1$ 합성곱과 sigmoid로 채널별 가중치를 만들어 본선에 곱하는 주의(attention)
가지이다. 여기에 backbone 얕은 층의 고해상도 특징을 더해 경계를 보강한다. 정확도는
DeepLabV3보다 낮지만 훨씬 가벼워 F6(CPU 배포) 관점의 후보가 된다.

**출력 head와 sigmoid.** 세 계열 모두 마지막 $1 \times 1$ 합성곱의 출력 채널을 1로 두어
픽셀별 logit 지도 $Z \in \mathbb{R}^{H \times W}$를 내고, sigmoid
$\sigma(z) = 1/(1 + e^{-z})$를 원소별로 적용해 확률 지도 $P = \sigma(Z)$를 얻는다.
분류망과 달리 이 계열의 밀집 예측 head는 내부에서 입력 해상도까지 업샘플하므로, 출력이
입력과 같은 $H \times W$ 격자에서 나온다.

**전체 shape 흐름** (입력 $512 \times 512$, 배치 크기 $N$, DeepLabV3-ResNet50 기준):

| 단계 | 연산 | 출력 shape | 의미 |
|---|---|---|---|
| 입력 | - | (N, 3, 512, 512) | 컬러 이미지 배치 |
| backbone | ResNet(atrous 후반) | (N, 2048, 32, 32) | 고해상도 유지 문맥 특징, 1/16 |
| ASPP | 다중 dilation + GAP 결합 | (N, 256, 32, 32) | 다중 스케일 문맥 융합 |
| 분류 head | Conv + 1x1 Conv(1채널) | (N, 1, 32, 32) | 저해상 픽셀 logit |
| 업샘플 | bilinear $\to$ 입력 크기 | (N, 1, 512, 512) | 입력 해상도 logit 지도 |
| 출력 | sigmoid | (N, 1, 512, 512) | 확률 지도 P |

```text
이미지 (3x512x512)
   $\to$ [backbone: (atrous) Conv]   $\to$ 문맥 특징 (Cx H/16 x W/16)
   $\to$ [분할 head: ASPP/FCN/LR-ASPP] $\to$ 저해상 logit (1x H/16 x W/16)
   $\to$ [bilinear 업샘플]            $\to$ 입력 해상도 logit (1x512x512)
   $\to$ [sigmoid]                    $\to$ 확률 지도 P (1x512x512)
```

**손으로 만든 디코더와의 차이.** 세그멘테이션 계열의 다른 구현이 인코더에 커스텀 UNet
디코더(전치 합성곱 + 스킵)를 조립해 절반 해상도 마스크를 내는 것과 달리, torchseg는
라이브러리가 제공하는 밀집 예측 head를 그대로 쓰고 입력 전체 해상도의 마스크를 낸다.
구조 선택의 자유도를 포기하는 대신, 검증된 다중 스케일 문맥 모듈과 사전학습 가중치를
얻는다.

### 2.3 손실 함수

출력이 픽셀별 이진 확률이므로 손실도 세그멘테이션 계열이 공유한다. 픽셀별 이진 교차
엔트로피(BCE)와 영역 겹침을 재는 Dice 손실을 결합한다.

$$
\mathcal{L}_{\mathrm{BCE}} = -\frac{1}{HW} \sum_{i,j}
\Big[ M_{ij} \log P_{ij} + (1 - M_{ij}) \log (1 - P_{ij}) \Big]
$$

$$
\mathrm{Dice}(P, M) = \frac{2 \sum_{i,j} P_{ij} M_{ij} + \epsilon}
{\sum_{i,j} P_{ij} + \sum_{i,j} M_{ij} + \epsilon},
\qquad
\mathcal{L} = \mathcal{L}_{\mathrm{BCE}} + \lambda\, (1 - \mathrm{Dice}(P, M))
$$

BCE는 픽셀을 독립적으로 다뤄 매끄러운 그래디언트($P_{ij} - M_{ij}$)를 주고, Dice는
전경과 배경의 픽셀 수 불균형에 강한 영역 수준 척도를 더한다. $\lambda$는 보통 1로 둔다.
수치 안정성을 위해 sigmoid와 BCE는 합쳐 계산한다(BCE with logits). 코너 정밀도를
좌우하는 것은 경계 근방 소수 픽셀의 오차이므로, 입력 해상도로 예측을 내는 torchseg는
절반 해상도 디코더보다 경계 분해능에서 유리하다.

### 2.4 후처리 이론

후처리 $g$는 확률 지도에서 코너 4개를 복원하는 결정적(비학습) 절차로, 세그멘테이션
계열과 동일하다. (i) 임계값 $\tau = 0.5$로 이진화하고, (ii) 컨투어 추출(Suzuki-Abe
경계 추적)로 전경의 외곽 폐곡선을 얻어 면적 최대 컨투어를 고르며, (iii) 폴리곤 근사
(Douglas-Peucker)로 곡선을 꼭짓점 몇 개로 단순화한다. 근사 허용 오차 $\varepsilon$은
컨투어 둘레에 비례($\varepsilon = \alpha \cdot \mathrm{Perimeter}$)하게 두고 결과가
4점이 아니면 $\alpha$를 조정해 재시도한다. (iv) 얻은 4점을 무게중심 기준 각도로 시계
방향 정렬 후 $x + y$ 최소점을 시작점으로 삼아 표준 순서로 맞추고, (v) 이미지 크기로
나눠 정규화하며 퇴화 여부를 검사한다.

폴리곤 근사가 항상 4점을 보장하지는 않으므로, 컨투어 부재/4점 미수렴/퇴화의 세 경우에
검출 실패가 존재한다. 따라서 성공 플래그를 함께 반환해야 하며, 코너 분해능은 마스크
해상도(여기서는 입력 해상도)로 제한된다.

## 3. 결론

Torchvision Segmentation Corner는 세그멘테이션 기반 코너 검출을 "라이브러리 분할망
재사용"으로 실현한 방법론이다. 장점은 다음과 같다.

- **낮은 설계 비용**: 검증된 밀집 예측 아키텍처(FCN/DeepLabV3/LR-ASPP)를 그대로 써서
  디코더 설계와 튜닝 부담이 거의 없다.
- **사전학습 전이**: 대규모 분할 과제로 학습된 가중치가 경계 분할 능력을 이미 가져,
  적은 도메인 데이터에서도 빠르게 수렴한다.
- **입력 해상도 예측**: 밀집 예측 head가 입력 크기로 업샘플하므로 경계 분해능이 좋다.

한계는 구조 자유도가 낮고 분할 head가 코너에 특화되지 않는다는 점, 그리고 세그멘테이션
계열 공통의 후처리 취약성(4점 미보장)과 해상도 제한이다. 이 방법론은 커스텀 디코더를
직접 설계하는 접근과 "라이브러리 재사용 대 직접 설계"라는 대조 축을 제공한다.

## 참고 자료

- Long, J., Shelhamer, E., Darrell, T. "Fully Convolutional Networks for Semantic
  Segmentation." CVPR 2015. (FCN)
- Chen, L.-C., Papandreou, G., Schroff, F., Adam, H. "Rethinking Atrous Convolution
  for Semantic Image Segmentation." arXiv:1706.05587, 2017. (DeepLabV3, ASPP)
- Howard, A., et al. "Searching for MobileNetV3." ICCV 2019. (MobileNetV3, LR-ASPP)
- Milletari, F., Navab, N., Ahmadi, S.-A. "V-Net: Fully Convolutional Neural
  Networks for Volumetric Medical Image Segmentation." 3DV 2016. (Dice 손실)
- Suzuki, S., Abe, K. "Topological Structural Analysis of Digitized Binary Images
  by Border Following." CVGIP, 1985. (findContours)
- Douglas, D. H., Peucker, T. K. "Algorithms for the Reduction of the Number of
  Points Required to Represent a Digitized Line or its Caricature." The Canadian
  Cartographer, 1973. (approxPolyDP)

## 부록 A. 데이터 인터페이스

- **model**: `TorchsegModel`은 torchvision 분할 빌더(`deeplabv3_resnet50`,
  `fcn_resnet50`, `deeplabv3_mobilenet_v3_large`, `lraspp_mobilenet_v3_large`)를
  `num_classes=1`로 생성하고, forward는 모델 출력 dict의 `"out"` 텐서
  $(N, 1, H, W)$(입력 해상도)를 반환한다.
- **preprocessor**: 세그멘테이션 계열과 동일한 마스크 타깃 변환을 이 방법론 안에 독립 구현한다(코너
  $(N, 4, 2)$ $\to$ 채운 볼록 사각형 마스크 $(N, 1, H, W)$ 래스터화). 마스크 크기는 모델
  출력 해상도에서 받아 매 스텝 생성하므로 image_size 변화에 정합한다.
- **postprocessor**: 세그멘테이션 계열과 동일한 마스크$\to$코너 복원을 이 방법론 안에 독립 구현한다(sigmoid +
  0.5 이진화 + findContours(최대 면적) + approxPolyDP($\varepsilon$ 반복) + 표준 정렬 +
  정규화). 4점 근사 실패 시 대체 폴백을 거친다.
- **실패 모드**: 컨투어 부재, 4점 미수렴, 퇴화 판정(`is_invalid_corners`) 시 코너를
  NaN으로 반환한다. Evaluator의 `SuccessRate`가 유효 표본 비율로 SR을 집계하고, 정확도
  메트릭은 성공 표본에만 적용하며 항상 SR과 함께 보고한다 (README.md 4절).

## 부록 B. 학습 전략

3단계 학습 전략을 적용한다.

| 단계 | 데이터 | 목적 | 학습률 |
|---|---|---|---|
| 1. Pre-training | SmartDoc + MIDV-2020 (공개) | 일반적인 사각형 영역 분할 능력 확보 | 1e-3 |
| 2. Domain Adaptation | 합성 fringe 패턴 | 검사 도메인의 화면 특성에 적응 | 1e-4 |
| 3. Fine-tuning | 실측 데이터 (소량) | 최종 경계 정밀도 확보 | 1e-5 |

**가중치 초기화**: backbone과 분할 head는 대규모 공개 분할 과제로 사전학습된 가중치에서
시작하되, 클래스 수가 1로 바뀌는 최종 분류 층은 무작위 초기화한다(사전학습의 다중 클래스
분류 층과 형태가 달라 이식 대상에서 제외됨). 사전학습된 경계 분할 능력 덕분에 코너
도메인 데이터가 적어도 수렴이 빠르다.

**타깃 자동 생성**: 마스크 타깃은 preprocessor가 코너 라벨로부터 매번 생성하므로, 어느
단계의 데이터든 코너 CSV만 있으면 별도 마스크 어노테이션 없이 학습에 쓸 수 있다.

## 부록 C. 제약 적합성 분석

F1-F8 제약의 정의는 `README.md` 2절을 따른다.

| 제약 | 대응 |
|---|---|
| F1 (임의 볼록 사각형) | 마스크는 임의 형상을 픽셀 단위로 표현하므로 볼록 사각형을 근사 없이 담는다 |
| F2 (단일 객체, 50% 이상) | 최대 면적 컨투어 선택이 안정적으로 동작하는 근거. 배경 소규모 오검출 조각은 자동 배제 |
| F3 (코너 경계 내부) | 사각형이 테두리에 걸리지 않아 컨투어가 잘리지 않고 온전한 폐곡선으로 추출됨 |
| F4 (실측 소량) | 대규모 분할 사전학습 가중치 이식 + 3단계 학습(부록 B) + 픽셀 단위 조밀 감독으로 소량 데이터 학습 보강 |
| F5 (서브픽셀 정밀도) | 입력 해상도 예측으로 경계 분해능은 좋으나 여전히 픽셀 격자에 묶임. cornerSubPix 후정제 옵션 |
| F6 (CPU 배포) | DeepLabV3-ResNet50는 무거움. LR-ASPP-MobileNetV3 계열로 경량 배포 대안 확보 |
| F7 (조명/글레어 변동) | 광학 증강(ColorJitter, GaussianBlur, GaussianNoise) 적용. 마스크 타깃은 광학 변화에 불변 |
| F8 (가림 없음) | 전경이 단일 연결 영역이라는 가정, 즉 최대 컨투어 하나만 처리하는 단순화의 근거 |
