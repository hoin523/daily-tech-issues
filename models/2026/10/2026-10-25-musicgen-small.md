# MusicGen-small: 24층 음악 Transformer와 4개 오디오 코드북

조사일: 2026-10-25 KST. 공개 카드·설정·논문·코드를 분석했으며 가중치 다운로드, 음악 생성, 학습은 수행하지 않았다.

## 선정과 식별

모델 ID는 `facebook/musicgen-small`, 개발 조직은 Meta FAIR다. revision은 `4c8334b02c6ec4e8664a91979669a501ec497792`다. 조사 시 [Hub API](https://huggingface.co/api/models/facebook/musicgen-small)는 다운로드 158,867, 좋아요 523을 반환했고 [모델 페이지](https://huggingface.co/facebook/musicgen-small)는 다운로드 집계 기간을 last month로 표시한다. 이 지표는 사용 관심도이며 음악 품질의 증거가 아니다.

기존 models·issues 전체 기록을 읽어 MusicGen 및 같은 가중치의 변형 분석이 없음을 확인했다. 최근 텍스트 모델과 달리 음악 생성 분야를 선정했다. [SpeechT5 분석](2026-10-19-speecht5-tts.md)은 음성 합성으로, 음악 토큰 생성과 목적이 다르다.

용도는 영어 텍스트 설명이나 오디오 prompt에 따른 음악 생성이다. 기반 구성 요소는 T5-base 텍스트 encoder와 32kHz EnCodec이며, 다른 LLM의 fine-tuning 모델이라고 분류하지 않는다. 카드의 학습 시기는 2023년 4~5월이다. [논문 최초 제출](https://arxiv.org/abs/2306.05284)은 2023-06-08이며 이 날짜를 체크포인트의 정확한 배포일로 단정하지 않는다. 해당 배포일은 이번 조사에서 확인 불가다. [고정 카드](https://huggingface.co/facebook/musicgen-small/blob/4c8334b02c6ec4e8664a91979669a501ec497792/README.md).

가중치 라이선스는 **CC-BY-NC 4.0**, 원 AudioCraft 코드는 MIT다. 공개 다운로드 가능성과 상업 이용 가능성을 구분해야 한다. [카드 라이선스](https://huggingface.co/facebook/musicgen-small/blob/4c8334b02c6ec4e8664a91979669a501ec497792/README.md).

## 구조: 300M이라는 이름에 포함되지 않는 구성 요소

[고정 config](https://huggingface.co/facebook/musicgen-small/blob/4c8334b02c6ec4e8664a91979669a501ec497792/config.json)에는 음악 decoder, T5, EnCodec 설정이 함께 들어 있다. 카드의 300M은 음악 언어 모델 크기 표기이며 패키지 전체와 다르다. Hub API의 safetensors 메타데이터는 전체 591,087,174개 F32 파라미터를 보고한다. 이를 모두 음악 decoder의 학습 파라미터라고 해석하지 않았다.

| 구성 | 확인값 | 역할 |
|---|---|---|
| 음악 decoder | 24 layers, hidden 1,024, FFN 4,096 | 이전 오디오 토큰에서 다음 토큰 예측 |
| attention | 16 heads, 표준 MHA; 별도 GQA 설정 없음 | causal self-attention과 텍스트 cross-attention |
| head dimension | 1,024 ÷ 16 = 64, 균등 분할 계산 | head별 폭 |
| 코드북 | 4개, 각각 2,048개 항목 | RVQ의 서로 다른 잔차 표현 |
| 음악 위치 표현 | sinusoidal embedding | 코드 패턴의 시간 위치 |
| decoder 위치 설정 | max_position_embeddings=2,048 | 텍스트 길이나 학습 step 수와 무관 |
| T5 text encoder | 12 layers, width 768, 12 heads, d_kv=64, FFN 3,072 | 영어 설명을 조건 벡터로 변환 |
| EnCodec | mono 32,000Hz, hidden_size 128, 초기 filters 64 | 오디오와 이산 코드 사이 변환 |
| codec 배율 / LSTM | [8,5,4,4] / 2 layers | 시간 해상도 변환·시계열 처리 |
| 활성화 | 배포 decoder config는 GELU | 논문 본문의 ReLU 설명과 구분 |
| MoE·활성 전문가 | 해당 없음 | dense 구성 |

구조 표는 [config](https://huggingface.co/facebook/musicgen-small/blob/4c8334b02c6ec4e8664a91979669a501ec497792/config.json)와 [Transformers v4.31.0 구현](https://github.com/huggingface/transformers/blob/v4.31.0/src/transformers/models/musicgen/modeling_musicgen.py)을 대조했다. 구현에는 sinusoidal embedding과 cross-attention이 있으며, config의 일반 플래그 `add_cross_attention=false`만 보고 cross-attention이 없다고 판정하지 않았다. T5 설정에 decoder 관련 값도 남아 있지만 MusicGen에서는 텍스트 **encoder**를 사용한다. [버전 고정 문서](https://huggingface.co/docs/transformers/v4.31.0/en/model_doc/musicgen).

EnCodec의 시간 배율은 `8×5×4×4=640`이고 `32,000÷640=50 frames/s`다. 논문은 이를 5층 비인과 codec으로 설명한다. 이 층 수를 내부 모든 합성곱·잔차·LSTM 연산의 총 개수로 바꾸어 세지 않았다. 학습 시 오디오를 코드로 압축하고, 생성 시에는 예측 코드를 codec decoder가 파형으로 복원한다. [논문 §3.1](https://arxiv.org/html/2306.05284v2#S3.SS1).

### 네 코드북을 어떻게 함께 생성하는가

RVQ의 첫 코드북은 큰 정보를 담고, 다음 코드북은 앞선 양자화가 남긴 잔차를 보완한다. 네 스트림을 완전히 직렬로 늘어놓으면 초당 `50×4=200` 토큰 위치를 순차 생성해야 한다. MusicGen은 `[0,1,2,3]` 지연을 두어 서로 다른 시점의 코드를 한 단계에서 예측한다. 따라서 초당 약 50회의 autoregressive 단계로 진행하며 시작·끝의 지연 정렬 처리는 별도다. ‘한 번에 모든 곡을 생성’하거나 ‘음악 1초를 실제 1초에 생성’한다는 뜻은 아니다. [고정 코드북 설정](https://github.com/facebookresearch/audiocraft/blob/896ec7c47f5e5d1e5aa1e4b260c4405328bf009d/config/model/lm/musicgen_lm.yaml), [논문 §2.2](https://arxiv.org/html/2306.05284v2#S2.SS2).

논문은 학습에 30초 crop을 사용한다고 설명한다. config의 2,048 위치를 50으로 나눠 새로운 지원 길이를 보장하지 않는다. Small은 melody 전용 체크포인트와 달리 chromagram 기반 멜로디 제어 모델이 아니며, 오디오 prompt를 이어가는 기능과 멜로디 조건 기능을 구분한다. [MusicGen 문서](https://huggingface.co/docs/transformers/v4.31.0/en/model_doc/musicgen).

## 학습: 실제 반복 횟수와 루프의 epoch 설정

[논문 v2 §3.1~3.2](https://arxiv.org/html/2306.05284v2#S3.SS1)은 약 20,000시간의 라이선스 음악에서 30초를 임의 추출해 음악 LM을 학습했다고 설명한다. 배포 카드는 공개 모델용 데이터에 HT-Demucs로 보컬을 제거했다고 덧붙인다. 따라서 논문 평가와 배포 체크포인트 평가를 같은 것으로 취급하지 않는다. [고정 카드](https://huggingface.co/facebook/musicgen-small/blob/4c8334b02c6ec4e8664a91979669a501ec497792/README.md).

| 항목 | 300M 음악 LM 학습 | 구성 요소의 별도 사전학습 |
|---|---|---|
| optimizer steps | 논문 1,000,000 | T5·codec 전용 전체 step 확인 불가 |
| epoch | 공개 recipe 500, 아래 의미 구분 필요 | 확인 불가 |
| batch | 논문 192 examples | 확인 불가 |
| gradient accumulation | 실제 실행 설정 확인 불가 | 확인 불가 |
| optimizer | AdamW 계열, small은 D-Adaptation 사용 | 확인 불가 |
| betas / weight decay / clip | (0.9,0.95) / 0.1 / 1.0 | 확인 불가 |
| LR / scheduler | 자동 step-size; recipe lr=1 / cosine | 확인 불가 |
| warmup / EMA | 4,000 steps / decay 0.99 | 확인 불가 |
| precision / 하드웨어 | mixed precision float16 / 32 GPUs | 이번 조사에서 상세 확인 불가 |
| GPU 기종 | 이번에 확인한 논문 구간·recipe로 확인 불가 | 확인 불가 |
| SFT·DPO·RL·LoRA | 별도 적용 근거 확인 불가 | 확인 불가 |

논문 수치와 별개로, 조사한 AudioCraft commit `896ec7c47f5e5d1e5aa1e4b260c4405328bf009d`의 [base recipe](https://github.com/facebookresearch/audiocraft/blob/896ec7c47f5e5d1e5aa1e4b260c4405328bf009d/config/solver/musicgen/musicgen_base_32khz.yaml)는 `epochs=500`, `optimizer=dadam`, `lr=1`을 둔다. 상속되는 [기본 recipe](https://github.com/facebookresearch/audiocraft/blob/896ec7c47f5e5d1e5aa1e4b260c4405328bf009d/config/solver/musicgen/default.yaml)는 `updates_per_epoch=2000`이다. 구성상 `500×2000=1,000,000` 업데이트로 논문 규모와 맞지만, 고정 업데이트 단위의 루프 epoch를 20,000시간 전체의 완전한 반복 횟수라고 해석하면 안 된다. 또한 D-Adaptation의 lr=1을 일반 AdamW의 고정 유효 학습률 1로 읽지 않는다. 공개 recipe가 이 Hub revision의 원 실행 인자 전체라는 보장도 없다.

텍스트 encoder는 [공식 conditioner 설정](https://github.com/facebookresearch/audiocraft/blob/896ec7c47f5e5d1e5aa1e4b260c4405328bf009d/config/conditioner/text2music.yaml)에서 `finetune=false`다. 이 설정의 CFG dropout 0.3은 논문 v2의 0.2와 달라 현재 recipe와 논문 실험을 구분했다. EnCodec의 별도 사전학습 설정을 음악 LM의 100만 step으로 채우지 않았다.

가정 계산으로 `1,000,000×192×30초=5.76×10^9초=160만 시간`의 crop 처리가 된다. 매 step에서 batch와 crop 길이가 같고 실패·재시작을 제외한다는 가정이며, 고유 음악 시간이나 검증된 epoch 수가 아니다. 추론의 약 50 steps/s와 학습 optimizer steps, 구조의 24 layers도 서로 다른 단위다.

## 공개 평가와 활용 판단

[배포 카드](https://huggingface.co/facebook/musicgen-small/blob/4c8334b02c6ec4e8664a91979669a501ec497792/README.md)는 MusicCaps에서 small의 FAD 4.88, KLD 1.42, text consistency 0.27을 보고한다. FAD는 VGGish 특징 분포, KLD는 PaSST 라벨 분포, text consistency는 CLAP 오디오·텍스트 임베딩을 이용한다. FAD·KLD는 낮을수록, text consistency는 높을수록 좋다는 평가 방향이며 서로 단위가 달라 점수를 더해 종합 순위를 만들지 않는다. 이번 분석의 실측값은 없다.

동일 카드에서 medium은 FAD 5.14·KLD 1.38·text consistency 0.28이다. Small이 한 지표에서 낮다고 전반적인 청취 품질이 더 좋다고 결론 내릴 수 없다. 논문은 MusicCaps의 약 5.5K개 10초 sample 중 불균형 전체 집합에서 객관 지표를, 장르 균형 subset에서 청취 평가를 설명한다. 논문의 원 모델과 보컬 제거 데이터로 학습한 공개 모델의 수치도 구별해야 한다. [논문 §3.2~3.3](https://arxiv.org/html/2306.05284v2#S3.SS2), [고정 카드](https://huggingface.co/facebook/musicgen-small/blob/4c8334b02c6ec4e8664a91979669a501ec497792/README.md).

음악 생성 구조 연구와 짧은 instrumental 아이디어 실험에 적합하다. 영어 prompt 중심이고 사실적인 보컬, 다양한 문화·장르에 고른 품질, 무음으로 끝나지 않는 생성을 보장하지 않는다. SpeechT5처럼 문장을 정확히 읽는 TTS의 대체 모델도 아니다. 실제 사용 판단에는 라이선스와 직접 청취 평가가 필요하다. [카드의 용도·한계](https://huggingface.co/facebook/musicgen-small/blob/4c8334b02c6ec4e8664a91979669a501ec497792/README.md).

전체 메타데이터의 F32 가중치만 계산하면 `591,087,174×4=2,364,348,696 bytes`, 약 2.36GB(십진 단위)다. 300M 음악 LM만의 크기와 다르며 VRAM 실측이나 최소 요구량이 아니다. KV cache·조건 encoder·codec·임시 activation과 guidance 실행 방식이 메모리에 추가로 영향을 준다.
