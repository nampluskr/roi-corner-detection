---
tags: [roi-corner-detection, planning, implementation-order]
created: 2026-07-10
updated: 2026-07-10
---

# ROI Corner Detection 방법론 구현 순서 분석

## 1. 개요

이 문서는 direct 방법론과 공유 인프라 구현이 끝난 시점에서, 나머지 9개 방법론
(seg, det, heatmap, hybrid, line, doc, homography, foundation, gcn)의 구현 착수
순서를 "구현 난이도와 속도" 기준으로 재분석한 결과이다. README 1절과 PLAN.md 2절의
확정 우선순위는 프로젝트 목적(fringe 패널 도메인, 제약 F1-F8) 적합도 기준으로 정렬한
것이고, 이 문서는 같은 방법론 집합을 구현 비용 관점에서 다시 정렬해 대안 순서를
제시한다.

이 문서의 순서는 구현 착수 순서 제안일 뿐, README/PLAN의 도메인 우선순위 자체를
대체하지 않는다. 채택 시 PLAN.md의 Phase 3 순서만 갱신한다.

## 2. 분석 기준

모든 방법론은 `src/models/<name>/`에 5개 파일(model, preprocessor, postprocessor,
loss, wrapper)만 추가하고, `src/models/base/`의 추상 클래스와 `src/core/`의
Trainer/Evaluator/Predictor, `src/losses/`, `src/metrics/`를 그대로 재사용한다.
factory.py의 `get_wrapper`에는 dispatch 분기 하나만 추가한다. 따라서 방법론 간 구현
난이도를 가르는 실제 변수는 다음 네 가지이다.

- postprocessor 복잡도: raw 출력을 표준 코너 (N, 4, 2)로 변환하는 로직의 난이도
- 새 model 구조: direct의 backbone+FC head 대비 얼마나 다른 구조가 필요한가
- 외부 의존: M-LSD, DocTr, DINOv2, SAM 등 외부 가중치나 패키지 도입 여부
- 데이터 의존: 합성 데이터(`src/data/synthetic.py`)에 대한 의존도

## 3. 방법론별 난이도 비교

각 방법론이 direct 대비 새로 작성해야 하는 요소와 종합 난이도는 다음과 같다.

| 방법 | raw 출력 | postproc 난이도 | 외부의존 | 데이터의존 | 종합 난이도 |
|---|---|---|---|---|---|
| homography | (N, 8) offset | 낮음 (canonical + offset) | 없음 | 보통 | 매우 낮음 |
| heatmap | (N, 4, H, W) | 중간 (soft-argmax, 순수 텐서) | 없음 | 보통 | 중간 |
| seg | (N, 1, H, W) 마스크 | 중간 (findContours + approxPolyDP) | OpenCV(기존) | 보통 | 중간 |
| hybrid | (N, 1, H, W) 마스크 | 중상 (Canny + Hough + cornerSubPix) | OpenCV(기존) | 보통 | 중간 (seg 뒤면 낮아짐) |
| det | (N, A, 5+4) 그리드 | 높음 (그리드 디코딩 + 클래스별 top-1) | 없음 | 큼 | 높음 |
| gcn | 코너 + 반복 정제 | 중간 (최종 반복 출력) | 없음 | 큼 | 높음 (구조 복잡) |
| doc | (N, 8) logits | 낮음 (sigmoid + reshape) | DocTr/DocScanner 패키지 | 낮음 | 낮음 (의존 제외 시) |
| foundation | (N, 8) logits | 낮음 (sigmoid + reshape) | DINOv2/SAM 가중치 | 낮음 | 낮음 (의존 제외 시) |
| line | 직선 세그먼트 | 최상 (4변 그룹화 + 교점 + 퇴화 처리) | M-LSD 가중치(부분) | 보통 | 높음 |

## 4. 재사용 사슬

구현 순서를 정할 때 앞선 방법론의 코드를 뒤 방법론이 재사용하는 세 개의 사슬이
핵심 근거가 된다.

- (N, 8) 좌표 계열: direct -> homography -> foundation -> doc. postprocessor가 direct의
  sigmoid + reshape와 사실상 동일하며(homography만 canonical + offset 변형), direct의
  wrapper와 head 구조를 최대한 재사용한다.
- seg -> hybrid: hybrid는 seg의 세그멘테이션 마스크 model을 그대로 사용하고 후처리만
  Canny + Hough + cornerSubPix로 교체한다. seg 완료 후 hybrid는 postprocessor만 새로
  작성하면 된다.
- heatmap: 독립적이지만 soft-argmax가 순수 텐서 연산이라 외부 의존과 OpenCV 후처리
  실패 모드가 없어 검증이 단순하다.

## 5. 제안 순서 (난이도/속도 기준)

빠르게 끝나고 코드 재사용이 큰 것부터 두고, 외부 의존과 데이터 의존, 복잡한 후처리를
뒤로 미룬 순서이다.

1. homography: direct의 사촌. FC head가 동일하고 preprocessor(canonical + offset)와
   postprocessor만 소량 추가한다. 외부 의존과 합성 의존이 없어 가장 빠른 두 번째 방법론이다.
2. heatmap: 외부 의존이 없고 soft-argmax가 순수 텐서 연산이다. OpenCV 후처리 실패 모드가
   없어 검증이 단순하고, F5(서브픽셀) 강점을 조기에 확보한다.
3. seg: 마스크 model과 OpenCV(findContours, approxPolyDP) 후처리를 구현한다. hybrid의
   선행 조건이다.
4. hybrid: seg의 마스크 model을 재사용하고 후처리만 교체한다. seg 직후에 두면 신규 코드가
   최소가 된다.
5. det: 그리드 박스 디코딩 후처리가 무겁고 합성 데이터 의존이 크다. 외부 의존은 없다.
6. gcn: 반복 정제 구조와 단계별 supervision으로 model이 복잡하고 합성 의존이 크다. 외부
   의존은 없다.
7. doc: 외부 패키지(DocTr/DocScanner) 도입이 필요하다. 후처리 자체는 쉬우나 의존과 도메인
   간극 리스크로 뒤에 둔다.
8. foundation: 외부 가중치(DINOv2/SAM)와 frozen backbone이 필요하다. 후처리는 쉬우나
   무거운 의존과 배포 부적합성 때문에 후반에 둔다. 성능 상한 탐색용이다.
9. line: 외부(M-LSD) 부분 의존과 직선 그룹화, 교점 계산, 퇴화 처리로 후처리 난이도가
   가장 높고 실패 모드가 많아 마지막에 둔다.

결과 순서: homography, heatmap, seg, hybrid, det, gcn, doc, foundation, line

## 6. 확정 순서와의 차이

- 확정 순서(도메인 적합도): seg, det, heatmap, hybrid, line, doc, homography, foundation, gcn
- 제안 순서(난이도/속도): homography, heatmap, seg, hybrid, det, gcn, doc, foundation, line

가장 크게 이동하는 두 방법론은 다음과 같다.

- homography: 도메인 가치는 낮게 평가됐지만(direct 대비 이득이 제한적) 구현이 가장 쉬워
  맨 앞으로 당겨진다.
- line: 경량 방법론이지만 후처리 난이도가 가장 높아 맨 뒤로 밀린다.

seg -> hybrid 인접성은 두 순서 모두 유지된다.

## 7. 절충안 (권장)

난이도만 좇으면 도메인 가치가 높은 seg와 hybrid가 뒤로 밀린다. 가치와 난이도를 함께
고려하려면 seg -> hybrid 쌍과 direct -> homography 사촌 관계를 살린 혼합안이 자연스럽다.

순서: homography, seg, hybrid, heatmap, det, gcn, doc, foundation, line

- homography를 워밍업으로 먼저 둔다(최저 난이도, direct 재사용 최대).
- 그다음 도메인 가치가 높은 seg와 hybrid를 인접 구현한다(마스크 model 공유).
- heatmap으로 서브픽셀 정밀도를 확보한다.
- 데이터 의존이 큰 detect와 gcn을 둔다.
- 외부 의존이 있는 doc, foundation, line을 마지막 묶음으로 둔다.

## 8. 개념 분리 제안

README 1절의 "우선순위"는 도메인 적합도 지표로 유지하고, 실제 구현 착수 순서는 PLAN.md
Phase 3에만 반영해 두 개념을 분리하는 것을 권장한다. 두 지표를 하나의 순서로 합치면
"가치가 높은데 구현이 어려운" 방법론과 "가치는 낮지만 구현이 쉬운" 방법론의 판단이
섞여 혼동을 일으킨다.

## 9. 채택 결과

이 프로젝트는 5절의 난이도/속도 순서를 단일 기준으로 채택했다.

채택 순서: direct -> homography -> heatmap -> seg -> hybrid -> det -> gcn -> doc
-> foundation -> line -> torchseg -> torchdet

8절의 개념 분리 권장과 달리, README/PLAN/CLAUDE의 순서를 이 하나로 통일했다. README 1절의
"우선순위" 열은 "구현 순서" 열로 바꿔 도메인 적합도가 아니라 이 착수 순서를 가리키도록 했다.
direct와 homography는 이미 구현되어 있어, 다음 착수 대상은 heatmap이다.

이후 기존 10개 방법론에 더해 라이브러리 기반 대조군 2개를 11번째 `torchseg`(torchvision
세그멘테이션 모델 end-to-end, `seg` 대조군)와 12번째 `torchdet`(torchvision 검출 모델
end-to-end, `det` 대조군)로 추가했다. 2-9절의 난이도 분석은 기존 10개 방법론을 대상으로
한 것이며, torchseg/torchdet은 각각 seg/detect의 preprocessor/postprocessor를 재사용하고
model만 torchvision 라이브러리 모델로 교체하는 구성이라 신규 후처리 설계 비용이 낮다.
