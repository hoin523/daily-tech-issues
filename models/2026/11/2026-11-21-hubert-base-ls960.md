# HuBERT-base: 7개 합성곱 층과 12층 Transformer로 배우는 음성 표현

조사일: 2026-11-21 (한국 시간). 공개 문서·설정 분석이며 가중치 다운로드·추론·학습은 수행하지 않았다.

## 선정과 식별

Hugging Face API에서 최근 한 달 다운로드 **163,005회**, 누적 좋아요 **82개**를 확인했다. 사용 관심도를 보여주는 지표이며 성능 증거는 아니다. 이미지 분류에서 음성 표현 학습으로 분야를 바꿨다. 최근 개발 조직은 Facebook에 편중되어 있어 이후 선정에서는 다른 조직을 우선할 필요가 있다. 기존 models·issues 50개 기록을 모두 읽고 동일 모델·revision·미러·양자화에 대한 구조 분석이 없음을 확인했다. [Hub API](https://huggingface.co/api/models/facebook/hubert-base-ls960)

| 항목 | 내용 |
|---|---|
| ID / 개발 | `facebook/hubert-base-ls960` / Facebook AI 연구진 |
| revision | `dba3bb02fda4248b6e082697eee756de8fe8aa8a` |
| 기반·단계 | HuBERT Base, LibriSpeech 960시간 음성 자기지도 사전학습본 |
| 용도 | 음성 프레임 표현 추출, 후속 음성 인식 학습의 출발점 |
| 발표 / 조사 | 논문 2021-06-14 / 조사 2026-11-21. HF 변환본 최초 공개일 확인 불가 |
| 라이선스 | Apache-2.0 |

[고정 카드](https://huggingface.co/facebook/hubert-base-ls960/blob/dba3bb02fda4248b6e082697eee756de8fe8aa8a/README.md), [공식 모델 목록](https://github.com/facebookresearch/fairseq/blob/3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99/examples/hubert/README.md), [논문](https://arxiv.org/abs/2106.07447). 공식 목록에서 Base는 ASR 미세조정되지 않은 모델이다.

## 파형에서 프레임으로, 프레임에서 문맥으로

| 구조 | 확인값 |
|---|---|
| 총 파라미터 | 공식 보고 약 95M. 직접 로딩하여 정확한 trainable 수는 집계하지 않음 |
| 입력 | 단일 파형, 16,000 Hz |
| 합성곱 추출기 | 7층, 각 출력 채널 512, bias 없음 |
| kernel / stride | [10,3,3,3,3,2,2] / [5,2,2,2,2,2,2] |
| Transformer | 양방향 Dense encoder 12층 |
| hidden / FFN | 768 / 3,072, GELU |
| attention / KV heads | 표준 MHA 12개, KV도 12개 |
| head dimension | `768÷12=64`: 계산값 |
| 위치 정보 | kernel 128, group 16의 convolutional positional embedding |
| dropout / LayerDrop | HF hidden·attention·activation 0.1 / LayerDrop 0.1 |
| context | 고정 텍스트 토큰 한도 미기재. 오디오 길이와 전역 attention의 비용이 제약 |
| decoder·MoE | 없음. 기본 HubertModel에는 문자 CTC head 없음 |

[고정 config](https://huggingface.co/facebook/hubert-base-ls960/blob/dba3bb02fda4248b6e082697eee756de8fe8aa8a/config.json), [전처리](https://huggingface.co/facebook/hubert-base-ls960/blob/dba3bb02fda4248b6e082697eee756de8fe8aa8a/preprocessor_config.json), [공식 모델 표](https://github.com/facebookresearch/fairseq/blob/3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99/examples/hubert/README.md). MoE 활성 전문가 파라미터 수는 해당하지 않는다.

합성곱 stride의 곱은 `5×2⁶=320`이다. 따라서 16kHz에서 프레임 간 간격은 `320÷16000=20ms`, 대략 초당 50프레임이다. 경계·kernel 때문에 출력 개수를 정확히 `입력 길이÷320`이라고 계산하지는 않는다. 12층 encoder가 각 프레임을 다른 시점의 소리와 함께 해석한다. [합성곱 구성](https://huggingface.co/facebook/hubert-base-ls960/blob/dba3bb02fda4248b6e082697eee756de8fe8aa8a/config.json)

학습에서는 미리 군집화한 음성 특징의 번호를 가린 구간에서 맞힌다. 군집 번호는 사람이 적은 단어가 아니다. 그래서 이 사전학습본의 벡터를 바로 문장으로 읽을 수 없다. `vocab_size=32`·tokenizer_class·CTC 관련 값이 config에 남아 있어도 카드가 명시한 tokenizer 부재와 기본 `HubertModel` 구조를 무시하면 안 된다. [카드](https://huggingface.co/facebook/hubert-base-ls960/blob/dba3bb02fda4248b6e082697eee756de8fe8aa8a/README.md), [논문](https://arxiv.org/abs/2106.07447)

## 학습 예제와 실제 변환본의 차이

공식 README는 음성 목록과 프레임별 k-means label을 준비한 뒤 학습하도록 안내한다. 다음은 고정 `hubert_base_librispeech.yaml`의 **공개 재현 설정**이다. 개별 HF revision의 전체 실행 로그나 모든 반복 단계의 실제 학습량으로 단정하지 않는다.

| 학습 항목 | 공개 설정 |
|---|---|
| 데이터·단계 | LibriSpeech 960시간 자기지도 사전학습. 원본 목록은 Base에 미세조정 없음으로 표시 |
| epoch | 명시값 미공개 |
| step | `max_update=400000`. epoch 또는 전체 반복 학습 누적 step과 동일시하지 않음 |
| batch budget | 장치별 `max_tokens=1400000`, world size 32 |
| 입력 제한 | min_sample_size 32,000 / max_sample_size 250,000 샘플, random crop |
| gradient accumulation | 이 YAML에 명시값 미공개; 실제 실행값 확인 불가 |
| optimizer | Adam, β=(0.9,0.98), eps=1e-6, weight decay=0.01 |
| learning rate / scheduler | 5e-4, polynomial decay, warm-up 32,000 updates |
| precision / 장비 | FP16, 분산 world size 32, node당 process 8. GPU 종류 확인 불가 |
| loss | masked prediction 가중치 1, unmasked prediction 가중치 0 |
| SFT·DPO·RL·LoRA | 이 사전학습본의 해당 단계는 보고되지 않음 |

[고정 학습 YAML](https://github.com/facebookresearch/fairseq/blob/3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99/examples/hubert/config/pretrain/hubert_base_librispeech.yaml), [학습·CTC 후속 학습 안내](https://github.com/facebookresearch/fairseq/blob/3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99/examples/hubert/README.md).

`max_tokens`는 여기서 문자 token 개수가 아니라 오디오 샘플 예산으로 읽어야 한다. 16kHz 가정에서 `1,400,000÷16,000=87.5초`는 장치별 예산의 시간 환산이며, 실제 매 update에 소비한 고유 음성량이 아니다. crop·batch 구성·누적 여부 없이 960시간과 나누어 epoch를 추정하지 않았다.

또한 공식 학습 YAML은 `mask_prob=0.80`, `encoder_layerdrop=0.05`, `normalize=false`를 사용하지만 HF config·전처리는 각각 mask_time_prob 0.05, LayerDrop 0.1, do_normalize true다. 서로 다른 구현·필드 의미 및 변환 설정을 조사하지 않고 같은 실행 recipe라고 합치면 안 된다. 이 차이의 정확한 변환 경위는 확인 불가다. [YAML](https://github.com/facebookresearch/fairseq/blob/3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99/examples/hubert/config/pretrain/hubert_base_librispeech.yaml), [HF config](https://huggingface.co/facebook/hubert-base-ls960/blob/dba3bb02fda4248b6e082697eee756de8fe8aa8a/config.json), [HF 전처리](https://huggingface.co/facebook/hubert-base-ls960/blob/dba3bb02fda4248b6e082697eee756de8fe8aa8a/preprocessor_config.json)

## 평가·비교·활용

원 논문은 LibriSpeech의 dev/test clean·other 분할과 여러 라벨 데이터 규모에서 후속 ASR을 평가한다. 문자 오류가 아닌 WER를 읽을 때도 미세조정 라벨 시간, 모델 크기, language model·decoder 조건을 함께 비교해야 한다. 공식 README도 Viterbi, KenLM, Fairseq-LM decoding을 별도로 안내한다. 오늘 Base 원본만으로 얻은 WER는 없으며, 논문의 1B 모델 개선 수치를 95M Base에 옮기지 않았다. [논문](https://arxiv.org/abs/2106.07447), [decoding 조건](https://github.com/facebookresearch/fairseq/blob/3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99/examples/hubert/README.md)

기존 [Wav2Vec2 Base-960h](../10/2026-10-12-wav2vec2-base-960h.md) 분석은 CTC 음성 인식 체크포인트를 다뤘다. 오늘 HuBERT는 문자 head 없이 음성 표현을 배우는 사전학습본이므로 바로 전사 결과를 비교할 수 없다. HuBERT의 군집 label 예측이라는 학습 목표도 구분한다.

분석상 음성 분류·검색·ASR 후속 학습의 encoder 출발점으로 쓸 수 있다. 16kHz 샘플링 조건을 지키고, 다른 언어·잡음·마이크 환경에서는 별도 검증해야 한다. 샘플링레이트 숫자만 바꾸는 것은 resampling이 아니다. 긴 오디오의 attention 비용과 분할 시 경계 문맥 손실도 고려해야 한다. 속도·VRAM·사용자 음성 정확도는 실측하지 않았다.
