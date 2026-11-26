# Bark-small: 세 개의 12층 Transformer와 EnCodec으로 만드는 생성 오디오

조사일: 2026-11-26 (한국 시간). 가중치 다운로드·추론·학습은 수행하지 않았다.

## 선정과 식별

Hugging Face API에서 최근 한 달 다운로드 **17,962회**, 누적 좋아요 **265개**를 확인했다. 성능이나 실제 사용자 수의 증거는 아니다. Microsoft 문서 이해에서 Suno 음성 생성으로 분야·조직을 바꿨다. 기존 models·issues 55개 기록 전체를 읽고 동일 모델·revision·미러·양자화 분석이 없음을 확인했다. EnCodec은 기존 글에서 다뤘지만 Bark의 세 단계 생성 모델 구조 분석은 없었다. [Hub API](https://huggingface.co/api/models/suno/bark-small)

| 항목 | 내용 |
|---|---|
| ID / 개발 | `suno/bark-small` / Suno |
| revision | `1dbd7a128513b8ae4a4e2130fed57b7ac9da5bcd` |
| 기반 | Bark 소형 원본 계열, EnCodec 24kHz codec 포함 |
| 용도 | 텍스트 조건 음성·비언어 소리를 포함하는 생성 오디오 |
| 발표 / 조사 | Bark 최초 2023-04-20, small 옵션 안내 2023-05-01 / 조사 2026-11-26. HF 변환본 최초 공개일 확인 불가 |
| 라이선스 | MIT |

[고정 카드](https://huggingface.co/suno/bark-small/blob/1dbd7a128513b8ae4a4e2130fed57b7ac9da5bcd/README.md), [공식 공개 이력](https://github.com/suno-ai/bark/blob/f4f32d4cd480dfec1c245d258174bc9bde3c2148/README.md). 이번에 확인한 공식 자료는 모델 카드·코드이며 Bark 전용 학습 논문은 확인 불가다.

## 단계별 구조

| 단계 | 구성과 동작 |
|---|---|
| semantic | 텍스트에서 의미 토큰 생성, causal Transformer |
| coarse | 의미 토큰에서 EnCodec 첫 2개 코드북 생성, causal Transformer |
| fine | coarse에서 총 8개 코드북으로 보완, non-causal Transformer |
| 세 Transformer 공통 | 각각 12층, hidden 768, head 12, block_size 1,024, dropout 0.0 |
| head dimension | `768÷12=64`: 계산값, 표준 MHA로 KV도 12개 |
| config의 input/output vocab | semantic 129,600/10,048, coarse 12,096/12,096, fine 1,056/1,056 |
| codec | 24,000Hz mono, codebook_size 1,024, codebook_dim·hidden 128 |
| codec decoder | upsampling ratios [8,5,4,2], LSTM 2층, residual layer 1 |
| 파라미터 수 | HF 카드의 small/large 설명은 Transformer별 약 80M/300M. codec까지 포함한 해당 revision의 정확한 총·trainable 계수는 확인 불가 |
| 위치 정보 | Transformer 구현에서 학습 가능한 position embedding 사용 |
| MoE | 없음 |

[고정 config](https://huggingface.co/suno/bark-small/blob/1dbd7a128513b8ae4a4e2130fed57b7ac9da5bcd/config.json), [카드 Model Details](https://huggingface.co/suno/bark-small/blob/1dbd7a128513b8ae4a4e2130fed57b7ac9da5bcd/README.md). 실제 의미 토큰 10,000개·음향 코드북 1,024개와 special/padding 등을 포함한 config vocabulary 크기를 혼용하지 않는다. 세 모델을 합쳐 “36층 단일 Transformer”라고 설명하지 않는다.

텍스트를 곧바로 파형으로 바꾸는 대신 먼저 소리의 의미를 표현하고, 거친 음향 코드를 만든 다음 세부 코드를 보완한다. 마지막에 codec decoder가 코드를 파형으로 복원한다. fine 단계는 양쪽 문맥을 보는 non-causal 모델이므로 모든 단계가 동일한 autoregressive 생성이라고 설명하면 틀린다. [공식 구조 설명](https://github.com/suno-ai/bark/blob/f4f32d4cd480dfec1c245d258174bc9bde3c2148/model-card.md)

codec stride 곱은 `8×5×4×2=320`이다. 24kHz를 가정하면 `24000÷320=75`프레임/초다. 이 계산은 codec 프레임률이며 semantic 토큰률이나 학습 step이 아니다. block_size 1,024도 곧바로 텍스트 1,024단어 또는 출력 길이 초 단위로 환산할 수 없다. [codec·block 설정](https://huggingface.co/suno/bark-small/blob/1dbd7a128513b8ae4a4e2130fed57b7ac9da5bcd/config.json)

## 학습과 추론의 공개 범위

| 항목 | 확인 결과 |
|---|---|
| 학습 단계 | 텍스트→semantic, semantic→coarse, coarse→fine의 모델 분리 공개 |
| 기반 codec | EnCodec 24kHz. Bark Transformer와 별도 구성 요소 |
| epoch / optimizer step | 원 실행값 미공개 |
| 데이터·토큰·오디오 시간 | 정확한 원 학습 규모 확인 불가 |
| batch / gradient accumulation | 원 실행값 미공개 |
| LR / scheduler / optimizer | 재현 가능한 원 학습 설정 확인 불가 |
| precision / 하드웨어 | config float32는 배포 구성. 원 학습 precision·GPU 종류·수 확인 불가 |
| SFT·DPO·RL·LoRA | 해당 체크포인트 적용 단계 미공개 |

[공식 모델 카드](https://github.com/suno-ai/bark/blob/f4f32d4cd480dfec1c245d258174bc9bde3c2148/model-card.md), [HF 카드](https://huggingface.co/suno/bark-small/blob/1dbd7a128513b8ae4a4e2130fed57b7ac9da5bcd/README.md), [config](https://huggingface.co/suno/bark-small/blob/1dbd7a128513b8ae4a4e2130fed57b7ac9da5bcd/config.json). 공개 추론 코드의 temperature·top-k나 모델 layer 수로 학습 epoch를 추정하지 않았다. 현재 공개 자료만으로 학습 recipe 전체를 재현할 수 있다고 주장하지 않는다.

HF config에 반복되는 generic `max_length=20`, `do_sample=false` 등의 값은 각 단계의 실제 생성 제어·generation config와 함께 읽어야 한다. 카드 예제는 `generate(..., do_sample=True)`를 사용한다. 이런 추론 옵션은 optimizer update나 학습 scheduler가 아니다. [config](https://huggingface.co/suno/bark-small/blob/1dbd7a128513b8ae4a4e2130fed57b7ac9da5bcd/config.json), [사용 예제](https://huggingface.co/suno/bark-small/blob/1dbd7a128513b8ae4a4e2130fed57b7ac9da5bcd/README.md)

## 평가·활용과 제약

확인한 공식 카드에는 small revision에 대응하는 고정 데이터셋·비교 조건의 MOS·WER 정량 평가표가 없다. 따라서 품질 수치는 확인 불가로 남긴다. 카드의 최적화 속도·메모리 절감 소개도 특정 환경의 결과를 확인하지 않고 내 장비 실측으로 옮기지 않았다. 다국어 예시나 웃음 샘플은 사용자 데이터 정확도를 대신하지 못한다. [카드](https://huggingface.co/suno/bark-small/blob/1dbd7a128513b8ae4a4e2130fed57b7ac9da5bcd/README.md)

공식 저장소는 Bark가 일반적인 엄격한 낭독 TTS와 달리 prompt에서 벗어날 수 있는 생성 모델이라고 설명한다. 분석상 표현적인 짧은 음성·효과음 실험에 적합하되 정확한 문장 낭독·발음·길이·음색 일관성은 별도 평가가 필요하다. 장문 분할 시 경계와 화자 일관성도 검증해야 한다. [공식 사용 안내](https://github.com/suno-ai/bark/blob/f4f32d4cd480dfec1c245d258174bc9bde3c2148/README.md)

기존 [SpeechT5 TTS](../10/2026-10-19-speecht5-tts.md)는 텍스트→Mel→vocoder 흐름이고, Bark는 semantic·codec 토큰 계층을 쓴다. 기존 [EnCodec](2026-11-08-encodec-24khz.md)은 소리를 압축·복원하는 codec 자체로, 텍스트에서 코드를 생성하는 Bark 전체와 다르다. 속도·VRAM·한국어 발음 품질은 실측하지 않았다.

구조 보충 근거: [공식 라이브러리 구현](https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/models/bark/modeling_bark.py). 라이브러리 태그는 학습 commit을 뜻하지 않는다.
