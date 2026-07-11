# TODO List

## 폴더 및 파일 정리

- [ ] notebooks/ 폴더 정리 -> notebooks/models, dataset/dataloader 평가용
<!-- - [ ] docs/ 폴더내 문서 줄바꿈 정리 -> 강제 줄바꿈 제거 -->

## 코드 수정 공통

- [ ] metrics = {} vs. metrics = None
- [ ] raw_output vs logits
- [ ] trainer tqdm 추가
- [ ] logger 추가

## 모델 구현 및 평가

각 모델별 branch 분기 (최종 12개) 만들어 개별 모델 작성 (CLAUDE.md/AGENTS.md/PLAN.md에
`method/<name>` 브랜치 전략으로 반영 완료)

- [ ] src/utils/plot.py or plotting.py -> show_history()
- [ ] base: 모든 모델에 공통 적용
  - self.metrics = {"xxx": XXXMetric(), ...}
- [ ] 1순위: direct regression
  - rennet18 / resnet34 / resnet50 -> /mnt/d/backbones
  - https://github.com/nampluskr/quad_regression/src 참조
