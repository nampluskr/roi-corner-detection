# docs-plan: 방법론 상세 보고서 작성 계획

이 문서는 `docs/` 폴더에 작성할 12개 방법론 상세 보고서의 문서명, 섹션 수준 목차,
공통 템플릿을 정의한다. 실제 문서 작성 시 이 계획을 따르고, 작성 완료 시 아래
체크리스트를 갱신한다. 방법론 개요와 구현 순서는 `common/roi-corner-detection-models.md`,
제약 F1-F8과 모듈 시그니처는 `README.md`(SSOT)를 참조한다.

## 1. 문서 목록

파일명은 구현 순서 2자리 넘버링 뒤에 방법론 full name의 kebab-case를 붙인
`{NN}_{full-name}.md` 형식을 적용하며 `docs/models/`에 둔다.

| 구현 순서 | 코드 | 문서명 | 작성 상태 |
|---:|---|---|---|
| 1 | `direct` | `models/01_direct-coordinate-regression.md` | [x] |
| 2 | `homography` | `models/02_homography-regression.md` | [x] |
| 3 | `heatmap` | `models/03_heatmap-keypoint-detection.md` | [x] |
| 4 | `seg` | `models/04_segmentation-corner.md` | [x] |
| 5 | `hybrid` | `models/05_dl-classical-cv-hybrid.md` | [x] |
| 6 | `det` | `models/06_bbox-keypoint-detection.md` | [x] |
| 7 | `gcn` | `models/07_polygon-gcn.md` | [x] |
| 8 | `doc` | `models/08_document-pretrained.md` | [x] |
| 9 | `foundation` | `models/09_foundation-adapter.md` | [x] |
| 10 | `line` | `models/10_line-intersection.md` | [x] |
| 11 | `torchseg` | `models/11_torchvision-segmentation.md` | [ ] |
| 12 | `torchdet` | `models/12_torchvision-detection.md` | [ ] |

방법론별 보고서 외에 다음 공통 문서를 `docs/common/`에 둔다. 공통 문서는 넘버링을 적용하지 않는다.

| 문서명 | 내용 | 작성 상태 |
|---|---|---|
| `common/roi-corner-detection-models.md` | 12개 방법론 비교 (표 + 방법론별 요약 + F1-F8 매핑) | [x] |
| `common/roi-corner-detection-metrics.md` | 평가 메트릭 상세 (Polygon IoU, MCD, MaxCD, Reprojection Error) | [x] |
| `common/roi-corner-detection-data-pipeline.md` | 데이터셋 라벨링 방법론(labelme, polygon vs mask, CSV 스키마 근거) + 3단계 학습 데이터 전략 + 합성 Fringe 패턴 생성 수식 | [x] |
| `common/roi-corner-detection-cli-guide.md` | scripts/ CLI 및 experiments/run.py 사용 시나리오 운영 매뉴얼 (학습~비교) | [x] |
| `common/roi-corner-detection-implementation-order.md` | 나머지 9개 방법론 구현 착수 순서 분석 (난이도/속도 기준 재정렬 + 절충안) | [x] |

공통 문서 중 `roi-corner-detection-metrics.md`, `roi-corner-detection-data-pipeline.md`는
방법론 보고서와 동일한 작성 규칙(딥러닝 초보자 대상, 수식 적극 사용, 레거시 비인용)을 따른다.
`roi-corner-detection-cli-guide.md`, `roi-corner-detection-implementation-order.md`는 학술
보고서가 아닌 실행/계획 성격의 문서로, 서술체 docs-rules를 따르되 수식/논문 형식 규칙은
적용하지 않는다.

## 2. 공통 섹션 목차 (템플릿)

모든 보고서는 논문 형식(초록 / 서론 / 주제별 독립 섹션 / 결론 / 참고 자료)을 따른다.
본문은 방법론 자체의 학술적 서술(프로젝트와 무관하게 성립하는 이론)로 한정하고,
이 프로젝트에 특화된 구현/설정/평가 내용은 참고 자료 뒤 부록 A-D로 분리한다.
방법론 특성상 해당 없는 본문 항목은 "해당 없음"으로 명시하고 섹션 자체는 유지한다
(이 규칙은 본문에만 적용하며 부록에는 적용하지 않는다).

```markdown
# <방법론 Full Name> (<코드>)

## 초록
- 방법론 요약, 핵심 특징, 주요 결과 요약 (미평가 시 명시). 번호 없는 섹션

## 1. 서론
- 배경과 동기: 프로젝트 내 위치(우선순위, 카탈로그 출처), 선택 이유
- 문제 정의: 코너 검출 task, 공통 입출력(CSV 스키마, (N, 4, 2) 표준 코너)

## 2. 이론 및 아키텍처
- 독립된 학술논문의 Theory 섹션 수준으로 수식을 포함해 상세히 작성

### 2.1 문제 정식화
- 입출력 정의, 표기법

### 2.2 모델 구조
- 레이어별 정의와 수식, 텐서 shape 흐름, 아키텍처 다이어그램 (텍스트 블록)

### 2.3 손실 함수
- 수식, 각 항의 의미와 가중치, 선택 근거

### 2.4 후처리 이론
- raw 출력 -> 코너 좌표 변환의 수학적 정의

## 3. 결론
- 요약, 이 방법론이 적합한 조건, 향후 작업 (평가 결과는 부록 D 참조)

## 참고 자료
- 공개 논문과 공개 구현만 수록. 번호 없는 섹션

## 부록 A. 데이터 인터페이스
- preprocessor: 표준 코너 (N, 4, 2) -> 학습 타깃 변환 방식
- postprocessor: raw 출력 -> 표준 코너 (N, 4, 2) 변환 방식
- 실패 모드 처리 (예: 4점 미검출 시 fallback)

## 부록 B. 학습 전략
- 3단계 학습(사전학습 -> 합성 적응 -> 실측 파인튜닝) 적용 방식
- 주요 하이퍼파라미터 (scripts/config.py DEFAULTS 기준)

## 부록 C. 제약 적합성 분석
- F1-F8 제약별 영향과 대응 (README.md 2절 기준)

## 부록 D. 평가 결과
- Polygon IoU, MCD, MaxCD, Reprojection Error, latency, 모델 크기
- 장단점, 실패 사례와 완화 방안 논의
```

## 3. 작성 규칙

- 언어는 한국어, 문서 스타일은 CLAUDE.md 6절(ASCII만, em dash/유니코드 화살표/이모지
  금지)을 따른다.
- 대상 독자는 딥러닝 초보자(분류/회귀 구분 정도의 지식 보유)이다. CNN, backbone,
  풀링, 손실 함수, 경사하강 등 기반 개념도 문서 안에서 수식과 함께 설명하여,
  사전 지식 없이 읽을 수 있는 자기완결적 문서로 작성한다.
- 레거시 프로젝트 문서는 작성 시 참고만 하고, 본문에 인용하거나 언급하지 않는다.
  참고 자료 목록에도 포함하지 않는다 (공개 논문/자료만 수록).
- 수식은 LaTeX 표기(ASCII 입력 가능)로 작성한다: 인라인 `$...$`, 블록 `$$...$$`.
- 본론에서 필요시 상세한 수식을 사용한다. 수식 사용은 2절에 한정되지 않는다.
- 2절(이론 및 아키텍처)은 해당 방법론을 처음 접하는 독자가 수식만으로 재구현 가능한
  수준의 자기완결적 서술을 목표로 한다.
- 본문(초록-참고 자료)은 프로젝트와 무관하게 성립하는 학술적 내용만 다루고,
  이 프로젝트 특화 구현/설정/평가는 부록 A-D에만 작성한다.
- 구현/평가 미완료 내용("미평가", "TBD" 등)은 문서에 포함하지 않는다. 평가 결과가
  확보되기 전에는 부록 D를 문서에서 생략(언급 없이)하고, 결과 확보 후 섹션을 추가한다.
- 코드 블록의 시그니처는 README.md(SSOT)와 일치시킨다. 불일치 발견 시 README.md를
  먼저 갱신한 뒤 보고서를 작성한다.
- 문서 간 중복 서술을 피한다. 방법론 간 비교는 `common/roi-corner-detection-models.md`가
  담당하고, 각 보고서는 해당 방법론의 상세만 다룬다.
- 작성 순서는 구현 순서(1 -> 10)를 따르되, 해당 방법론의 구현 Phase 완료 후 작성을
  원칙으로 한다 (PLAN.md Phase 참조).
