# TinyLlama-1.1B-Chat v1.0: 22층 GQA와 사전학습·SFT·DPO의 구분

조사일: 2026-11-15 (한국 시간). 공개 자료 분석이며 가중치 다운로드·학습·추론은 실행하지 않았다.

## 선정과 식별

Hugging Face API에서 최근 한 달 다운로드 **946,602회**, 누적 좋아요 **1,912개**를 확인했다. 오디오 구성 요소에 이어 소형 대화 모델을 선택했다. 지표는 사용 관심도이며 품질이나 최신 트렌딩 순위를 뜻하지 않는다. [API](https://huggingface.co/api/models/TinyLlama/TinyLlama-1.1B-Chat-v1.0), [모델 페이지](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)

| 항목 | 확인 내용 |
|---|---|
| 모델 / 개발 | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` / TinyLlama 연구진 |
| revision | `fe8a4ea1ffedaf415f4da2f062534de366a451e6` |
| 기반 모델 | `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` |
| 라이선스 / 주 언어 | Apache-2.0 / 영어 |
| 발표 | TinyLlama 논문 v1 2024-01-04. 이 Chat v1.0 가중치의 정확한 최초 공개일은 이번 조사에서 확인 불가. |
| 용도 | causal decoder-only 대화 모델, Llama 계열 구조·tokenizer 활용 |

근거: [고정 카드](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/blob/fe8a4ea1ffedaf415f4da2f062534de366a451e6/README.md), [논문](https://arxiv.org/html/2401.02385v1). Llama와 구조가 호환된다는 설명을 Llama 2 가중치 자체를 축소한 모델이라는 뜻으로 해석하지 않았다.

기존 `models`·`issues` 전체를 읽었다. OLMo 글에서 관련 모델로 언급됐지만 TinyLlama의 구조 분석은 없었다. 동일 revision·양자화·미러 모델의 중복 분석이 아니다.

## 구조와 KV 공유

| 항목 | 확인값 |
|---|---|
| 저장 파라미터 원소 | Hub BF16 집계 1,100,048,384개. 직접 가중치를 로드해 trainable 계수를 재집계한 값은 아님. |
| layer / hidden | 22 / 2,048 |
| attention Q heads | 32 |
| KV heads | 4, GQA |
| head dimension | `2048÷32=64`, 계산값 |
| Q:KV 그룹 | `32÷4=8`, 하나의 KV head를 8개 Q head가 공유 |
| FFN intermediate | 5,632, SiLU gate를 사용하는 SwiGLU |
| context | max position 2,048 |
| 위치 표현 | RoPE, theta 10,000, rope_scaling 없음 |
| 정규화 | RMSNorm epsilon 1e-5 |
| 어휘 | 32,000 |
| embedding / 출력 | `tie_word_embeddings=false` |
| 기타 | attention bias 없음, MoE·이미지 encoder 없음 |

근거: [고정 config](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/blob/fe8a4ea1ffedaf415f4da2f062534de366a451e6/config.json), [논문 §2.2](https://arxiv.org/html/2401.02385v1), [API](https://huggingface.co/api/models/TinyLlama/TinyLlama-1.1B-Chat-v1.0).

각 층은 과거 문맥을 읽고 다음 토큰의 표현을 만든다. GQA는 query 방향을 32개 유지하면서 key·value는 4개로 줄여 공유한다. 따라서 MHA의 KV 32개와 같은 캐시 크기로 계산하면 안 된다. 하지만 8배 공유가 모델 전체 메모리나 속도의 8배 개선을 뜻하지는 않는다. [구성](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/blob/fe8a4ea1ffedaf415f4da2f062534de366a451e6/config.json)

예를 들어 batch 1·길이 2,048·모든 22층의 KV를 BF16으로 저장한다는 단순 가정이면 `2(K,V)×22×2048×4×64×2byte=46,137,344byte`, 즉 **44MiB**다. cache 텐서만의 이론값이며 가중치·activation·allocator·서빙 구현 비용을 제외한다. VRAM 실측값이 아니다.

## 단계별 학습

| 항목 | 사전학습 | SFT | DPO |
|---|---|---|---|
| 데이터 | SlimPajama·StarCoder 계열 | UltraChat 변형 | UltraFeedback 선호 쌍 |
| 규모 | 약 3T 누적 토큰 | `ultrachat_200k`가 카드에 등재, 실제 사용 표본 수 확인 불가 | 카드 설명상 64k prompts 규모, 사용 split·실제 쌍 수 확인 불가 |
| epoch | 논문상 약 1T corpus의 약 3회 반복 | 미공개 | 미공개 |
| step | 기반 ID의 `1431k`는 이름 정보. 실행 로그를 검증한 정확한 update 수로 쓰지 않음. | 미공개 | 미공개 |
| optimizer | 논문 AdamW, β1=0.9·β2=0.95 | 실행 대응값 확인 불가 | 실행 대응값 확인 불가 |
| LR / schedule | 최대4e-4·최소4e-5 cosine, warm-up2,000step | 확인 불가 | 확인 불가 |
| batch / accumulation | 논문 batch 약2M tokens, accumulation 확인 불가 | 확인 불가 | 확인 불가 |
| precision / 장비 | 학습 precision 확인 불가 / 논문16 A100-40GB | 확인 불가 | 확인 불가 |

근거: [고정 카드](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/blob/fe8a4ea1ffedaf415f4da2f062534de366a451e6/README.md), [논문 §2.4](https://arxiv.org/html/2401.02385v1). 저장 BF16을 원 학습 전체 precision의 증거로 쓰지 않았다. 논문은 warm-up 뒤 LR가 잠시 최대값에 머문 설정 버그도 밝히므로 이상적인 cosine 곡선이 실행 전체에 정확히 적용됐다고 단정하지 않는다.

SFT는 대화 예시를 따라 답하는 학습이고, DPO는 같은 prompt의 선호·비선호 응답 쌍을 활용해 응답 선호를 조정한다. 카드가 TRL DPOTrainer와 Zephyr recipe를 따른다고 설명해도 Zephyr의 모든 숫자를 TinyLlama의 실행값으로 복사할 근거는 아니다. [카드](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/blob/fe8a4ea1ffedaf415f4da2f062534de366a451e6/README.md)

공식 저장소의 [sft/script.sh](https://github.com/jzhang38/TinyLlama/blob/bf122247c486b6b897050e98cbb7bedae8eeba73/sft/script.sh)는 V0.1·V0.2 및 OpenAssistant 기반 예제다. 그곳의 5 epoch·LR1e-5를 오늘의 UltraChat·DPO 기반 v1.0 설정으로 적용하지 않았다. stage별 정확한 재현 로그는 확인하지 못했다.

## 평가와 활용 범위

논문은 Language Model Evaluation Harness로 HellaSwag·ARC·PIQA 등 zero-shot 상식 추론을 평가한다. 이는 사전학습 모델 연구의 결과이므로 Chat v1.0 revision의 대화 품질 점수로 옮기지 않았다. MMLU·BBH는 shot 조건도 다르다. [논문 §3](https://arxiv.org/html/2401.02385v1)

작은 대화 모델의 로컬 실험과 미세조정 출발점으로 검토할 수 있다는 것이 이번 분석의 해석이다. 1.1B라는 크기와 DPO 단계가 사실 정확도·한국어 품질·장문 추론을 보장하지 않는다. 2,048 위치와 영어 중심 데이터 조건도 고려해야 한다. [config](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/blob/fe8a4ea1ffedaf415f4da2f062534de366a451e6/config.json), [카드](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/blob/fe8a4ea1ffedaf415f4da2f062534de366a451e6/README.md)

카드의 role 기반 chat template을 사용해 system·user·assistant를 구분해야 한다. 일반 텍스트만 이어붙인 입력과는 학습 형식이 다를 수 있다. [사용 예제](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/blob/fe8a4ea1ffedaf415f4da2f062534de366a451e6/README.md#how-to-use)

[GPT-Neo 125M](2026-11-10-gpt-neo-125m.md)은 기본 LM과 global/local attention을 살펴본 사례다. 오늘 모델은 GQA와 SFT·DPO를 함께 살펴보는 사례여서 layer 수·파라미터 수만으로 사용 목적을 비교할 수 없다.

출처·구성·중복을 검토했다. 추론 속도·VRAM·생성 품질을 실측하지 않았으며 계산값과 원문 보고값을 구분했다.
