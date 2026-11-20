# DeiT-small: 12층 이미지 Transformer를 300 epoch 학습하기

조사일: 2026-11-20 (한국 시간). 가중치 다운로드나 추론·학습 없이 공개 구성과 학습 코드를 분석했다.

## 선정과 식별

Hugging Face 공개 API에서 최근 한 달 다운로드 **12,878회**, 누적 좋아요 **16개**를 확인했다. 다운로드 실적이 있는 이미지 분류 원본 계열을 선정했으며 인기 최상위나 성능 우위를 뜻하지 않는다. 전날 텍스트 요약에서 이미지 인식으로 분야를 바꿨다. 기존 `models`·`issues` 전체를 읽고 모델 ID·구조 분석의 중복이 없음을 확인했다. [Hub 지표](https://huggingface.co/api/models/facebook/deit-small-patch16-224)

| 항목 | 확인 내용 |
|---|---|
| 모델 / 개발 조직 | `facebook/deit-small-patch16-224` / Facebook AI 연구진 |
| revision | `164deee347853469b97442b3817f22eece80c7e3` |
| 기반·용도 | DeiT-small, ImageNet-1k 지도 학습 이미지 분류 |
| 발표일 / 조사일 | 논문 v1 2020-12-23 / 2026-11-20. 해당 HF 변환본 최초 공개일은 확인 불가 |
| 라이선스 | Apache-2.0 |

[고정 모델 카드](https://huggingface.co/facebook/deit-small-patch16-224/blob/164deee347853469b97442b3817f22eece80c7e3/README.md), [원 논문](https://arxiv.org/abs/2012.12877). 카드는 원 연구팀 대신 HF 팀이 작성했으며 timm에서 변환한 가중치라고 설명한다. 공식 연구소 코드와 HF revision의 실행 이력을 동일시하지 않는다.

## 구조: 패치 196개와 분류 토큰 하나

| 항목 | 값과 해석 |
|---|---|
| 파라미터 | 공식 보고 약 22M. 로딩으로 정확한 수·학습 가능 계수를 재집계하지 않음 |
| 유형 | Dense Transformer encoder + 1,000개 범주 분류 head |
| 입력 | RGB 224×224, 패치 16×16 |
| 패치·토큰 수 | `(224÷16)²=196`개 패치, CLS 포함 197개 토큰: 계산값 |
| Transformer block | 12개 |
| hidden / FFN | 384 / 1,536, GELU |
| attention / KV head | 표준 MHA 6개, KV도 6개. GQA·MQA 아님 |
| head dimension | `384÷6=64`: 계산값 |
| 위치 정보·attention | 학습 가능한 absolute position embedding, 양방향 전역 self-attention |
| dropout / LayerNorm epsilon | hidden·attention dropout 0.0 / 1e-12 (HF config) |
| decoder·VAE·MoE | 없음 |
| distillation token | 이 일반판에는 없음. 이름에 `distilled`가 붙은 별도 변형과 구분 |

[고정 config](https://huggingface.co/facebook/deit-small-patch16-224/blob/164deee347853469b97442b3817f22eece80c7e3/config.json), [공식 모델 정의](https://github.com/facebookresearch/deit/blob/ab5715372db8c6cad5740714b2216d55aeae052e/models.py), [공식 모델 표](https://github.com/facebookresearch/deit/blob/ab5715372db8c6cad5740714b2216d55aeae052e/README.md). 224는 픽셀 해상도이며 LLM의 context length가 아니다.

이미지를 작은 타일로 자른 뒤 각 타일을 384차원 벡터로 바꾼다. 12개 블록에서 타일 사이의 관계를 섞고 CLS 토큰을 분류 head에 넣는다. 패치 위치를 따로 더하므로 단순한 타일 목록과 다르게 공간 배치를 활용할 수 있다. DeiT 연구의 핵심에는 학습 recipe 개선과 증류가 모두 있지만, **오늘 체크포인트는 증류 토큰을 쓰지 않는 일반 small**이다. [모델 설명](https://huggingface.co/facebook/deit-small-patch16-224/blob/164deee347853469b97442b3817f22eece80c7e3/README.md), [공식 코드](https://github.com/facebookresearch/deit/blob/ab5715372db8c6cad5740714b2216d55aeae052e/models.py)

## 학습: 공개 재현 명령과 체크포인트 이력 구분

다음은 고정 공식 코드의 small 재현 명령(4 GPU, 장치별 batch 256)과 기본 인자를 함께 읽은 결과다. 개별 HF 변환본의 원 실행 로그로 확인된 수치라고 주장하지 않는다.

| 항목 | 공식 재현 설정에서 확인한 값 |
|---|---|
| 단계·데이터 | ImageNet 2012 / ImageNet-1k 지도 이미지 분류, 1,000개 범주 |
| epoch | 300 |
| 실제 optimizer update 총수 | 확인 불가. 데이터 분할·sampler·마지막 batch 처리 없이 epoch에서 단정하지 않음 |
| batch | 장치별 256 × GPU 4 = 전역 1,024장: 명령 기준 계산 |
| gradient accumulation | 별도 설정·원 실행 기록 확인 불가 |
| optimizer | AdamW, eps 1e-8, weight decay 0.05 |
| learning rate | 기본 5e-4에 `장치별 batch×world_size÷512` 적용. 위 명령이면 `5e-4×256×4÷512=0.001` |
| scheduler | cosine, warm-up 5 epoch, warmup LR 1e-6, min LR 1e-5 |
| precision | NativeScaler를 사용하는 AMP 학습 경로. HF 원 실행의 정밀도 로그는 확인 불가 |
| 하드웨어 | 재현 명령은 4 GPU. 정확한 GPU 종류·HF 원 실행 장비는 확인 불가 |
| SFT·DPO·RL·LoRA | 해당 적용 단계는 미공개. 언어 모델 후속 학습 단계를 억지로 대응시키지 않음 |

[학습 명령](https://github.com/facebookresearch/deit/blob/ab5715372db8c6cad5740714b2216d55aeae052e/README.md), [인자·LR scaling·학습 루프](https://github.com/facebookresearch/deit/blob/ab5715372db8c6cad5740714b2216d55aeae052e/main.py), [AMP 경로](https://github.com/facebookresearch/deit/blob/ab5715372db8c6cad5740714b2216d55aeae052e/engine.py). 모델 카드의 일반적인 8 GPU·3일 설명을 위 4 GPU 예제의 실측 소요 시간으로 옮기지 않았다. 카드가 사전학습·미세조정이라고 표현하더라도 이 small에 별도 384 해상도 미세조정이나 1,000 epoch 학습을 부여하지 않는다.

**epoch 300은 데이터셋 반복 횟수**이고 block 12는 계산 구조의 깊이다. inference step 수와도 무관하다. 데이터 증강이 있으므로 같은 이미지가 반복되어도 입력 내용이 매번 똑같다고 볼 수 없다. [학습·데이터 코드](https://github.com/facebookresearch/deit/blob/ab5715372db8c6cad5740714b2216d55aeae052e/datasets.py)

## 전처리 불일치와 평가 해석

HF 카드의 문장은 ImageNet mean/std와 256 resize·224 center crop을 설명하지만, 같은 revision의 `preprocessor_config.json`에는 **mean/std 각각 [0.5, 0.5, 0.5], size 224**가 저장되어 있다. 문장만 보고 두 설정을 동일하다고 가정하면 재현 조건을 놓친다. 사용 시 실제 로딩한 전처리기의 resize·crop·normalization 동작을 확인해야 한다. 여기서는 어느 쪽이 정확도를 더 높이는지 실측하지 않았다. [카드](https://huggingface.co/facebook/deit-small-patch16-224/blob/164deee347853469b97442b3817f22eece80c7e3/README.md), [고정 전처리 파일](https://huggingface.co/facebook/deit-small-patch16-224/blob/164deee347853469b97442b3817f22eece80c7e3/preprocessor_config.json)

공식 코드 저장소의 ImageNet validation 평가 안내는 small의 top-1 **79.854%**, top-5 **94.968%**를 보고한다. 카드의 반올림 표는 79.9%·95.0%다. 원본 공식 체크포인트와 공식 평가 pipeline의 보고값이지 HF 변환본을 이번에 재현한 값이 아니다. top-5는 정답이 상위 다섯 후보 안에 드는 비율로, top-1과 구분한다. [공식 평가 명령·출력](https://github.com/facebookresearch/deit/blob/ab5715372db8c6cad5740714b2216d55aeae052e/README.md), [카드](https://huggingface.co/facebook/deit-small-patch16-224/blob/164deee347853469b97442b3817f22eece80c7e3/README.md)

같은 공식 표의 small distilled는 top-1 81.2%를 보고하지만 별도 학습·별도 가중치다. 일반 small에 그 수치를 붙이면 안 된다. 기존 [ConvNeXt-Tiny](2026-11-13-convnext-tiny-224.md)는 합성곱 계층 구조이고 DeiT-small은 패치 간 전역 attention을 쓰므로 이미지 특징을 섞는 방식이 다르다. 크기나 FLOPs만으로 장치별 속도를 단정할 수 없다.

## 활용과 한계

분석상 ImageNet 범주 분류의 기준 모델이나 작은 이미지 encoder의 전이 학습 출발점으로 활용할 수 있다. 그러나 분류 head는 정해진 1,000개 범주를 출력하며 자유로운 이미지 설명·물체 위치 탐지 기능은 없다. 문서·의료·산업 이미지처럼 학습 분포가 다른 입력은 별도 평가가 필요하다. 생성 모델처럼 문장 길이나 sampling 설정을 조절하는 모델도 아니다. 속도·VRAM·사용자 데이터 정확도는 실측하지 않았다.
