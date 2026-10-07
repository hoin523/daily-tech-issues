# Whisper large-v3: 음성을 읽는 32층 encoder와 문장을 쓰는 32층 decoder

조사일: 2026-10-07 KST. 공개 설정과 설명을 분석했으며 음성 추론·학습은 실행하지 않았다.

## 선정과 식별

모델 ID: `openai/whisper-large-v3`. revision: `06f233fe06e710322aca913c1bc4249a0d71fce1`. [Hub API](https://huggingface.co/api/models/openai/whisper-large-v3) 조회값은 다운로드 4,046,310, 좋아요 6,570이다. 다운로드 응답의 집계 기간은 확인 불가이며 누적값이라고 단정하지 않는다. 기존 models·issues 전체 기록과 중복되지 않고 앞선 LLM·임베딩 분석과 다른 음성 분야를 선정했다.

용도는 다국어 음성 전사와 영어로의 음성 번역이다. Hub 라이선스 표기는 Apache-2.0이다. 별도 기반 체크포인트에서 fine-tuning됐다는 근거는 확인 불가이며 large-v2를 이용해 만든 의사 라벨은 데이터 생성 관계다. [고정 모델 카드](https://huggingface.co/openai/whisper-large-v3/blob/06f233fe06e710322aca913c1bc4249a0d71fce1/README.md).

[OpenAI 공식 카드](https://github.com/openai/whisper/blob/main/model-card.md)는 large-v3 발표를 2023년 11월로 기록한다. 정확한 일자는 이번 조사에서 확인하지 않았다.

## 구조

아래 값은 고정 revision의 [config.json](https://huggingface.co/openai/whisper-large-v3/blob/06f233fe06e710322aca913c1bc4249a0d71fce1/config.json)과 [preprocessor_config.json](https://huggingface.co/openai/whisper-large-v3/blob/06f233fe06e710322aca913c1bc4249a0d71fce1/preprocessor_config.json)을 직접 확인했다.

| 항목 | 값·의미 |
|---|---|
| 구조 | WhisperForConditionalGeneration, encoder-decoder |
| 파라미터 | large 계열 약 1,550M, 공식 코드 카드 기준 |
| 레이어 | encoder 32 + decoder 32 |
| hidden dimension | 1,280 |
| attention heads | encoder·decoder 각각 20 |
| head dimension | 1,280 ÷ 20 = 64, 표준 균등 분할 가정 계산 |
| FFN dimension | 양쪽 모두 5,120 |
| activation | GELU |
| 오디오 | 16,000 Hz, 30초 단위 |
| 입력 특징 | 128 Mel bins, 최대 3,000 frames |
| FFT / hop | 400 / 160 samples |
| encoder 위치 상한 | 1,500 |
| decoder 위치 상한 | 448 tokens |
| vocab_size | 51,866 |
| 저장 dtype | float16, 학습 precision과 별개 |
| MoE·GQA | 전문가 구성 없음; 별도 KV heads 설정 없음 |

파라미터 수의 출처는 [공식 코드 카드](https://github.com/openai/whisper/blob/main/model-card.md)다. Dense 구성에서 MoE식 활성 파라미터 수는 적용하지 않는다. `num_hidden_layers=32`만 보고 전체를 32층으로 세면 encoder·decoder 구분을 놓친다.

입력 파형을 시간·주파수 표현으로 바꾼 뒤 encoder가 음성 정보를 읽는다. decoder는 이전 텍스트와 encoder 출력을 함께 참조하며 다음 토큰을 생성한다. 30초 입력 단위와 decoder의 448-token 위치 상한은 서로 다른 축이므로 ‘448초를 듣는다’고 해석할 수 없다. 긴 파일은 별도 분할·연결 전략이 필요하다. [모델 카드](https://huggingface.co/openai/whisper-large-v3/blob/06f233fe06e710322aca913c1bc4249a0d71fce1/README.md).

위치 임베딩의 구체적인 구현은 이번에 고정한 config만으로 확인 불가다. RoPE 같은 LLM 설정을 임의로 적용하지 않는다.

## 학습

[large-v3 카드](https://huggingface.co/openai/whisper-large-v3/blob/06f233fe06e710322aca913c1bc4249a0d71fce1/README.md)는 약한 라벨 음성 100만 시간과 large-v2로 의사 라벨을 붙인 음성 400만 시간을 혼합해 **2.0 epochs** 학습했다고 명시한다.

| 항목 | large-v3 학습 |
|---|---|
| 데이터 | 약한 라벨 1M + 의사 라벨 4M hours |
| epoch | 2.0 |
| optimizer step | 확인 불가 |
| batch / gradient accumulation | 확인 불가 |
| learning rate / scheduler / optimizer | v3 전용 설정 확인 불가 |
| 학습 precision / 하드웨어 | v3 전용 설정 확인 불가 |
| 별도 SFT·DPO·RL·LoRA | 적용 근거 확인 불가 |

2 epochs는 혼합 데이터 반복 횟수이고 32 layers는 구조 깊이다. Step 수는 batch와 샘플링·길이 처리 방식 없이는 역산할 수 없다. 추론 예제의 batch_size나 torch_dtype는 학습 설정이 아니다. 초기 Whisper 논문의 학습량이나 optimizer를 v3 설정으로 그대로 복사하지 않는다.

## 비교와 평가 한계

large-v2 대비 입력 특징은 80에서 128 Mel bins로 늘었고 광둥어 토큰이 추가됐다. 공개 카드의 오류 감소 주장은 저자 측 평가이며 이번 분석에서 개별 데이터셋·언어별 점수와 디코딩 조건을 직접 대조하지 못해 성능 수치를 순위로 재현하지 않았다. [모델 카드](https://huggingface.co/openai/whisper-large-v3/blob/06f233fe06e710322aca913c1bc4249a0d71fce1/README.md).

실제 전사 비교는 같은 음성 데이터와 정규화·언어 지정·디코딩 조건에서 WER 또는 CER를 측정해야 한다. 한국어에서는 띄어쓰기 처리도 점수에 영향을 줄 수 있다. [공식 카드](https://github.com/openai/whisper/blob/main/model-card.md)는 언어·환경별 정확도 차이, 말하지 않은 내용 생성과 반복 오류 가능성을 설명한다.

## 사용 판단

녹음 전사·자막 생성 후보로 적합하지만 화자 분리나 실시간 응답 품질을 이 모델 하나의 공개 설명으로 보장하지 않는다. 전날의 MiniLM은 텍스트를 검색 벡터로 변환하며 이 모델은 음성을 텍스트로 변환한다. 두 기능을 파이프라인으로 연결할 수는 있어도 같은 평가 지표로 우열을 비교할 수 없다.

FP16 가중치만 가정하면 `1.55×10^9 × 2 bytes ≈ 3.10 GB`다. 공식 계열 파라미터 반올림 값에 기반한 계산이며 VRAM 실측이 아니다. activation·캐시·배치·런타임 메모리는 추가된다. 가중치 다운로드와 유료 연산은 수행하지 않았다.
