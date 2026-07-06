# Foundation Adapter (foundation)

## 초록

Foundation Adapter는 수억 장 규모의 이미지로 자기지도(self-supervised) 사전학습된
foundation model(DINOv2, SAM 등)의 backbone을 **동결(frozen)** 상태로 특징
추출기로 쓰고, 그 위에 얹은 경량 head만 학습하여 코너 좌표를 회귀하는 방법론
이다. 학습 가능한 파라미터가 전체의 극히 일부(head)로 제한되므로 수십 장 수준의
few-shot 데이터로도 과적합 없이 학습할 수 있고, backbone이 변하지 않으므로 특징을
미리 계산해 캐싱하면 학습이 매우 빨라진다. 대신 backbone 자체가 무거워 추론
비용이 크므로 실시간 배포용이 아니라, "범용 표현 위에서 도달 가능한 성능
상한이 어디인가"를 재는 탐침(probe) 성격의 방법론이다.

## 1. 서론

### 1.1 배경: foundation model과 자기지도 학습

foundation model은 특정 과제가 아니라 "이미지 일반"을 표현하도록 대규모 데이터로
사전학습된 모델을 가리킨다. 핵심 기술은 자기지도 학습으로, 사람이 만든 라벨 없이
데이터 자체에서 학습 신호를 만들어 낸다. 예를 들어 같은 이미지의 서로 다른
증강(crop, 색 변형)이 같은 표현을 갖도록 학습하면, 모델은 "무엇이 같은 물체
인가"를 라벨 없이 배운다. 이렇게 배운 표현은 특정 과제에 편향되지 않은 범용
시각 특징이며, 분류/분할/깊이 추정 등 다양한 하위 과제에서 소량의 추가 학습만
으로 강한 성능을 보이는 것이 반복적으로 확인되었다.

- **DINOv2**: 자기증류(self-distillation, 학생 네트워크가 EMA로 갱신되는 교사
  네트워크의 출력을 모방) 방식으로 학습된 ViT 계열 모델로, 패치 단위의 조밀한
  특징이 물체 부분/기하 구조에 잘 정렬되어 있다.
- **SAM(Segment Anything Model)**: 10억 개 마스크 규모의 데이터로 학습된
  분할 특화 모델로, 그 ViT 인코더는 물체 경계에 민감한 특징을 제공한다.

### 1.2 왜 동결하는가

표적 데이터가 수십 장뿐이라면, 수억 개 파라미터의 backbone을 파인튜닝하는 것은
극심한 과적합을 부른다. backbone을 동결하고 작은 head만 학습하면, 학습이 탐색
하는 가설 공간이 "고정된 범용 특징의 선형/저용량 조합"으로 제한되어 적은 표본
으로도 일반화가 성립한다. 이는 통계적 학습 이론의 기본 원리(모델 용량이 작을수록
필요한 표본 수가 적다)를 아키텍처로 구현한 것이다. 부수 효과로, backbone이
불변이므로 각 이미지의 특징을 한 번만 계산해 저장해 두면 이후 학습은 특징 벡터
위에서만 이루어져 몇 초 단위로 끝난다.

### 1.3 문제 정의

입력은 컬러 이미지 한 장이고, 출력은 TL(좌상단) -> TR(우상단) -> BR(우하단) ->
BL(좌하단) 시계 방향 순서의 코너 4개 좌표(정규화 [0, 1])이다. raw 출력은 좌표
logits 8개로 직접 회귀와 같고, 이 방법론의 정체성은 특징 추출기가 "동결된
범용 foundation backbone"이라는 점에 있다.

## 2. 이론 및 아키텍처

### 2.1 문제 정식화

모델을 특징 추출기 $\phi$와 head $h$의 합성으로 쓴다.

$$
f(I) = h_{\theta_h}\big(\phi_{\theta_\phi}(I)\big)
$$

동결 학습이란 $\theta_\phi$를 사전학습 값에 고정하고 $\theta_h$만 최적화하는
것이다.

$$
\hat{\theta}_h = \arg\min_{\theta_h} \frac{1}{N} \sum_{n=1}^{N}
\mathcal{L}\big(h_{\theta_h}(\phi(I_n)),\ \mathbf{y}_n\big), \qquad
|\theta_h| \ll |\theta_\phi|
$$

$N$이 매우 작은 few-shot 조건에서, 학습되는 파라미터 수 $|\theta_h|$가 작을수록
과적합 위험이 줄어든다. 극단은 선형 head 하나만 학습하는 linear probing이며,
그 성능은 "표현 $\phi$가 이 과제에 필요한 정보를 얼마나 선형적으로 접근 가능한
형태로 담고 있는가"를 재는 표준 척도로 쓰인다. 본 방법론은 그보다 약간 용량을
높인 경량 head(작은 합성곱/MLP)를 허용하되, backbone 동결이라는 원칙은 유지한다.

### 2.2 모델 구조

**(1) ViT(Vision Transformer) backbone.** foundation model의 표준 구조인 ViT를
개관한다. 입력 이미지를 $P \times P$ 픽셀의 패치들로 자르고(총 $n = HW / P^2$개),
각 패치를 펼쳐 선형 사영으로 $d$차원 토큰을 만든다.

$$
\mathbf{z}_i = E \cdot \mathrm{flatten}(\mathrm{patch}_i) + \mathbf{e}_i^{pos},
\qquad i = 1, \dots, n
$$

$E \in \mathbb{R}^{d \times 3P^2}$는 패치 임베딩 행렬, $\mathbf{e}_i^{pos}$는
위치 정보를 주입하는 위치 임베딩이다 (이 임베딩이 없으면 토큰 순서, 즉 공간
배치 정보가 사라진다). 토큰 행렬 $Z \in \mathbb{R}^{n \times d}$는 transformer
블록을 반복 통과한다. 블록의 핵심인 self-attention은 토큰들을 질의/키/값으로
사영한 뒤

$$
Q = Z W_Q, \quad K = Z W_K, \quad V = Z W_V, \qquad
\mathrm{Attn}(Q, K, V) = \mathrm{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V
$$

모든 토큰 쌍의 유사도($QK^\top$)에 비례해 정보를 섞는다. 합성곱이 고정된 국소
창 안에서만 정보를 모으는 것과 달리, attention은 첫 층부터 이미지 전역의 관계를
직접 계산한다. 각 블록은 attention과 2층 MLP를 잔차 연결과 층 정규화로 감싼
구성이며, 출력은 입력과 같은 개수의 토큰이다. 즉 ViT의 최종 출력은 패치 격자에
대응하는 조밀한 특징 토큰들이고, 이를 격자로 재배열하면 CNN의 특징 맵처럼 쓸 수
있다.

**(2) 동결 backbone + 경량 head.** ViT의 패치 토큰들을 특징 맵으로 재배열한 뒤,
작은 head가 좌표를 회귀한다.

$$
F = \mathrm{reshape}\big(\phi(I)\big) \in \mathbb{R}^{d \times \frac{H}{P} \times \frac{W}{P}}
$$

$$
\hat{\mathbf{y}} = \sigma\Big( \mathrm{FC}\big(\mathrm{GAP}(\mathrm{Conv}_{1\times1}(F))\big) \Big) \in (0, 1)^8
$$

$\mathrm{Conv}_{1\times1}$은 채널 축소(예: $d = 768 \to 256$), GAP는 전역 평균
풀링, FC는 8차원 완전연결, $\sigma$는 원소별 sigmoid이다. 학습 파라미터는 이
head뿐이다.

**전체 shape 흐름** (ViT-B/14, 입력 $518 \times 518$, 배치 크기 $N$):

| 단계 | 연산 | 출력 shape | 학습 여부 |
|---|---|---|---|
| 입력 | - | (N, 3, 518, 518) | - |
| 패치 임베딩 | 14x14 패치 -> 토큰 | (N, 1369, 768) | 동결 |
| Transformer x12 | attention + MLP | (N, 1369, 768) | 동결 |
| 재배열 | 토큰 -> 격자 | (N, 768, 37, 37) | - |
| 1x1 Conv | 채널 축소 | (N, 256, 37, 37) | 학습 |
| GAP + FC | 전역 요약 + 선형 | (N, 8) | 학습 |
| Sigmoid | 원소별 $\sigma$ | (N, 8) | - |

```text
이미지 (3x518x518)
   -> [패치 임베딩 (동결)]        -> 토큰 (1369x768)
   -> [Transformer x12 (동결)]    -> 토큰 (1369x768)   <- 여기까지 캐싱 가능
   -> [재배열]                    -> 특징 맵 (768x37x37)
   -> [1x1 Conv + GAP + FC (학습)] -> logits (8)
   -> [sigmoid]                   -> 정규화 좌표 (8)
```

**(3) 특징 캐싱.** 동결 경계(그림의 "캐싱 가능" 지점)까지의 계산은 이미지마다
한 번만 수행하면 된다. 전체 데이터셋의 특징을 미리 계산해 디스크에 저장하면,
학습 루프는 저장된 특징 -> head의 짧은 경로만 반복하므로 에폭당 수 초로 단축
된다. few-shot 실험(하이퍼파라미터 탐색, 반복 실험)의 회전율을 크게 높이는 실무
이점이다.

### 2.3 손실 함수

head 파라미터에만 그래디언트가 흐르는 좌표 회귀 손실이다. 원소별 오차
$e = \hat{y}_i - y_i$에 SmoothL1($|e| < \beta$에서 $e^2 / 2\beta$, 그 외
$|e| - \beta/2$) 또는 Wing Loss를 적용해 평균한다.

$$
\mathcal{L} = \frac{1}{8} \sum_{i=1}^{8} \ell(\hat{y}_i - y_i), \qquad
\theta_h \leftarrow \theta_h - \eta \, \nabla_{\theta_h} \mathcal{L}, \quad
\theta_\phi \text{ 고정}
$$

동결이 손실 설계에 주는 함의는 다음과 같다.

- **과적합 억제가 구조에 내장**: 정규화(가중치 감쇠, dropout)의 부담이 작다.
  few-shot에서도 강한 정규화 없이 안정적으로 수렴한다.
- **파국적 망각 없음**: backbone이 변하지 않으므로 사전학습 표현이 훼손될 일이
  없다. 파인튜닝 계열이 필요로 하는 보호 장치(층별 학습률, 초기값 앵커)가
  불필요하다.
- **옵티마이저 메모리 절약**: Adam류 옵티마이저는 파라미터당 2개의 통계량을
  유지하는데, 학습 파라미터가 head뿐이므로 그 비용이 무시할 수준이다.

### 2.4 후처리 이론

raw 출력이 좌표 logits $(N, 8)$이므로 후처리는 직접 회귀와 동일한 결정적 변환
이다. 원소별 sigmoid로 (0, 1) 범위를 보장하고,

$$
\hat{\mathbf{y}} = \sigma(\mathbf{z}) \in (0, 1)^8
$$

$(N, 8)$을 $(N, 4, 2)$로 reshape한다. 출력 원소와 코너의 대응이 학습에서 고정
되므로 순서 정렬이 필요 없고, 실패 모드도 없다.

## 3. 결론

Foundation Adapter는 "표현은 빌리고 해독기만 배운다"는 극단적 표본 효율 설계
이다. 장점은 다음과 같다.

- **few-shot 성립**: 학습 파라미터가 경량 head로 제한되어 수십 장 데이터로도
  과적합 없이 학습된다.
- **범용 표현의 품질**: 대규모 자기지도 학습이 만든 특징은 특정 도메인 편향이
  적어, 새로운 도메인(검사 패널)에서도 유효할 가능성이 높다.
- **실험 회전율**: 특징 캐싱으로 학습이 수 초 단위가 되어 반복 실험이 싸다.
- **상한 탐침**: 이 방법론의 성능은 "현존 최상급 범용 표현으로 도달 가능한
  수준"의 근사 상한으로, 다른 방법론의 개선 여지를 판단하는 기준이 된다.

한계는 배포 비용과 적응력의 상한이다. ViT 계열 backbone은 추론이 무거워 CPU
실시간 배포에 부적합하고, backbone이 동결이므로 사전학습 표현에 없는 도메인
특화 정보(fringe 패턴 고유의 미세 단서)는 head가 아무리 학습해도 접근할 수 없다.

이 방법론은 실측 데이터가 극히 적은 초기 단계에서 성능 상한과 데이터 가치를
가늠하는 보험 성격의 선택지이며, 상한이 확인되면 그 성능을 경량 모델로 증류
하거나 파인튜닝 계열로 전환하는 후속 작업의 출발점이 된다.

## 참고 자료

- Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al. "An Image is Worth 16x16
  Words: Transformers for Image Recognition at Scale." ICLR 2021. (ViT)
- Caron, M., Touvron, H., Misra, I., et al. "Emerging Properties in
  Self-Supervised Vision Transformers." ICCV 2021. (DINO)
- Oquab, M., Darcet, T., Moutakanni, T., et al. "DINOv2: Learning Robust Visual
  Features without Supervision." arXiv:2304.07193, 2023.
- Kirillov, A., Mintun, E., Ravi, N., et al. "Segment Anything." ICCV 2023. (SAM)
- Vaswani, A., Shazeer, N., Parmar, N., et al. "Attention Is All You Need."
  NIPS 2017.

## 부록 A. 데이터 인터페이스

- **preprocessor**: 표준 코너 $(N, 4, 2)$를 $(N, 8)$로 flatten한다 (direct와
  동일). 좌표가 이미 [0, 1] 정규화 및 순서 고정 상태이므로 모양 변경 외의 변환은
  없다. 입력 해상도는 backbone의 패치 크기 배수 제약(예: ViT-B/14는 14의 배수)을
  따르도록 factory의 input_size 설정과 맞춘다.
- **postprocessor**: raw 출력 logits $(N, 8)$에 sigmoid를 적용한 뒤 $(N, 4, 2)$로
  reshape한다 (direct와 동일, 2.4절).
- **실패 모드**: 모델이 항상 8개 값을 출력하므로 실패 모드가 구조적으로 없다.
  별도의 fallback이 불필요하다.

## 부록 B. 학습 전략

backbone이 동결이므로 3단계 전략이 "head 학습의 데이터 순서"로 단순화된다.
모든 단계에서 backbone 학습률은 0이다.

| 단계 | 데이터 | 목적 | 학습률 (head) |
|---|---|---|---|
| 1. Pre-training | SmartDoc + MIDV-2020 (공개) | head의 일반 코너 해독 능력 학습 | 1e-3 |
| 2. Domain Adaptation | 합성 fringe 패턴 | 검사 도메인 특징 분포에 head 적응 | 1e-4 |
| 3. Fine-tuning | 실측 데이터 (소량) | 최종 정밀도 확보 | 1e-5 |

**few-shot 축약 실험**: 이 방법론의 목적(F4 극단 대응) 확인을 위해, 단계 1-2를
생략하고 실측 소량 데이터만으로 head를 학습하는 축약 구성을 별도 실험으로
둔다. 축약 구성과 전체 구성의 성능 차이가 "공개/합성 데이터의 기여"를 정량화
한다.

**특징 캐싱**: 각 단계 시작 시 해당 데이터셋의 backbone 특징을 일괄 추출해
저장하고, 학습 루프는 캐싱된 특징 위에서 돈다 (2.2절). 단, 기하 증강(회전,
원근)은 특징이 아닌 이미지에 적용해야 하므로, 증강을 쓰는 단계에서는 캐싱을
증강 표본 단위로 확장하거나 캐싱 없이 학습한다.

## 부록 C. 제약 적합성 분석

F1-F8 제약의 정의는 `README.md` 2절을 따른다.

| 제약 | 대응 |
|---|---|
| F1 (임의 볼록 사각형) | 8개 좌표 자유 회귀로 임의 볼록 사각형 표현 가능 (direct와 동일) |
| F2 (단일 객체, 50% 이상) | 전역 요약 기반 회귀가 안정적으로 동작하는 조건. 물체가 커서 패치 토큰 다수가 물체 정보를 담음 |
| F3 (코너 경계 내부) | sigmoid 치역 $(0, 1)$로 구조적 보장 (2.4절) |
| F4 (실측 소량) | 이 방법론의 존재 이유. 동결 backbone + 경량 head로 few-shot 상한 탐색 (본 프로젝트에서 F4 최강 후보) |
| F5 (서브픽셀 정밀도) | 패치 격자(stride 14)가 거칠어 미세 지역화에 불리. 전역 회귀 한계도 direct와 공유 |
| F6 (CPU 배포) | ViT backbone이 무거워 F6 최하위. 배포 후보가 아닌 상한 탐침 용도로 위치 지정 |
| F7 (조명/글레어 변동) | 대규모 자기지도 표현이 조명 변동에 강건한 것으로 알려져 있고, 광학 증강 병행 |
| F8 (가림 없음) | 전역 회귀라 부분 가림 강건성 요구 자체가 낮고, 본 도메인은 가림이 없어 해당 없음 |
