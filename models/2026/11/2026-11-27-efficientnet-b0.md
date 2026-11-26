# EfficientNet-B0: 7개 stage·16개 MBConv 블록과 균형 있는 모델 확장

조사일: 2026-11-27 (한국 시간). 가중치 다운로드·추론·학습 없이 공개 자료를 분석했다.

## 선정과 식별

Hub API에서 최근 한 달 다운로드 **5,068회**, 누적 좋아요 **56개**를 확인했다. 최상위 트렌딩이라고 주장하지 않으며 사용 실적이 있는 소형 CNN 원본 계열로 선정했다. Suno 음성 생성에서 Google 이미지 분류로 분야·조직을 바꿨다. 기존 models·issues 56개 기록을 모두 읽고 동일 모델·revision·양자화·미러 분석이 없음을 확인했다. [API](https://huggingface.co/api/models/google/efficientnet-b0)

| 항목 | 내용 |
|---|---|
| ID / 개발 | `google/efficientnet-b0` / Google Research Brain Team |
| revision | `3d8d75d0812b65064f0151b456f869c6344bf20c` |
| 기반·용도 | EfficientNet-B0 ImageNet-1k 지도 이미지 분류 |
| 발표 / 조사 | 논문 v1 2019-05-28 / 조사 2026-11-27. 해당 HF 변환본 최초 공개일 확인 불가 |
| 라이선스 | Apache-2.0 |

[고정 카드](https://huggingface.co/google/efficientnet-b0/blob/3d8d75d0812b65064f0151b456f869c6344bf20c/README.md), [논문](https://arxiv.org/abs/1905.11946). 카드는 원 연구팀 대신 HF 팀이 작성했다.

## 구조: attention head 대신 합성곱 stage 읽기

| 항목 | config 확인값 |
|---|---|
| 유형 | Dense CNN, MBConv·squeeze-and-excitation |
| 입력 | RGB 224×224 |
| stage 수 / 반복 수 | 7개 / [1,2,2,3,3,4,1] |
| MBConv 합계 | `1+2+2+3+3+4+1=16`: 계산값. 합성곱 연산 총 층수와 다름 |
| stage 출력 채널 | [16,24,40,80,112,192,320] |
| kernel | [3,3,5,3,5,5,3] |
| 첫 블록 stride | [1,2,2,2,1,2,1] |
| expansion ratio | [1,6,6,6,6,6,6] |
| SE 비율 / activation | 0.25 / Swish |
| 마지막 특징·pooling | 1,280채널 / mean pooling |
| width / depth coefficient | 각각 1.0 / 1.0 |
| dropout / drop connect | 0.2 / 0.2 |
| BatchNorm | epsilon 0.001, momentum 0.99 |
| 총 파라미터 | 논문 B0 약 5.3M. 해당 변환본의 정확한 trainable 수는 직접 집계 안 함 |
| attention·KV head·RoPE·MoE | 해당 없음 |

[고정 config](https://huggingface.co/google/efficientnet-b0/blob/3d8d75d0812b65064f0151b456f869c6344bf20c/config.json), [논문 구조·모델 표](https://arxiv.org/html/1905.11946v5). hidden_dim 1,280은 마지막 이미지 특징 채널이며 LLM hidden dimension처럼 전 층에 고정된 값이 아니다.

MBConv는 채널을 넓힌 표현에서 depthwise 합성곱을 수행하고 다시 좁힌다. SE는 전체 공간에서 얻은 요약으로 채널별 중요도를 조절한다. 공간 attention이나 언어 모델의 GQA와 다르다. B0를 출발점으로 깊이·너비·해상도를 함께 확장하는 것이 compound scaling이며, depth coefficient 1.0은 학습 epoch가 아니다. [논문](https://arxiv.org/html/1905.11946v5)

## 학습 정보의 범위

| 항목 | 논문 ImageNet 설정과 확인 한계 |
|---|---|
| 데이터·목적 | ImageNet 지도 분류. HF 카드는 ImageNet-1k 명시 |
| epoch / step | 이 revision의 실제 종료 epoch·optimizer step 확인 불가 |
| batch / gradient accumulation | 해당 실행값 확인 불가 |
| optimizer | RMSProp, decay 0.9, momentum 0.9 |
| LR / scheduler | 초기 0.256, 2.4 epoch마다 0.97배 |
| weight decay | 1e-5 |
| 정규화·증강 | AutoAugment, stochastic depth, B0 dropout 0.2 |
| 종료 선택 | train에서 25K minival을 분리해 early stopping 후 원 validation 평가 |
| precision / 장비 | HF config float32는 배포 구성. 원 실행 precision·장비 확인 불가 |
| SFT·DPO·RL·LoRA | 오늘 체크포인트의 해당 단계 미공개 |

[논문 §5.2](https://arxiv.org/html/1905.11946v5), [카드](https://huggingface.co/google/efficientnet-b0/blob/3d8d75d0812b65064f0151b456f869c6344bf20c/README.md), [config](https://huggingface.co/google/efficientnet-b0/blob/3d8d75d0812b65064f0151b456f869c6344bf20c/config.json). 논문 recipe를 HF 변환본의 실행 로그로 단정하지 않았다. 2.4 epoch는 LR 감소 간격이며 총 학습 횟수가 아니다.

## 전처리와 평가 해석

고정 processor는 resize 224×224, rescale 1/255, mean [0.485,0.456,0.406], std [0.47853944,0.4732864,0.47434163]이다. 흔히 외우는 ImageNet std와 다르므로 임의로 바꾸지 않는다. crop_size 289가 저장돼 있어도 `do_center_crop=false`여서 기본 활성 crop이라고 읽지 않는다. [고정 전처리](https://huggingface.co/google/efficientnet-b0/blob/3d8d75d0812b65064f0151b456f869c6344bf20c/preprocessor_config.json)

원 논문 B0의 ImageNet 단일 모델·단일 crop top-1은 **77.1%**, top-5는 **93.3%**다. 논문 평가 pipeline의 보고값이며 오늘 HF revision으로 재현한 결과가 아니다. 1,000개 범주의 점수이므로 사용자 도메인·한국어 OCR·물체 위치 탐지 성능으로 일반화하지 않는다. [논문 Table 2](https://arxiv.org/html/1905.11946v5)

기존 [ResNet-50](2026-11-07-resnet-50.md)은 bottleneck 잔차 블록, [ConvNeXt-Tiny](2026-11-13-convnext-tiny-224.md)는 현대화된 합성곱 stage를 다뤘다. B0는 depthwise·SE와 작은 기준 구조를 읽는 사례다. 적은 파라미터·FLOPs만으로 장치별 지연 시간이 더 짧다고 단정하지 않는다.

분석상 이미지 분류의 가벼운 기준 모델이나 전이 학습 출발점에 적합하다. 현재 head는 ImageNet 범주를 출력하며 새 범주는 head 교체·라벨 데이터 학습과 별도 평가가 필요하다. 속도·VRAM·사용자 이미지 정확도는 실측하지 않았다.
