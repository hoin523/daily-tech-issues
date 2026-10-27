# Nomic Embed Text v1.5: 12층 encoder와 잘라 쓸 수 있는 Matryoshka 벡터

조사일: 2026-10-28 KST. 가중치 다운로드·추론·학습 없이 공개 자료를 분석했다.

## 선정·식별

모델 ID는 `nomic-ai/nomic-embed-text-v1.5`, revision은 `e9b6763023c676ca8431644204f50c2b100d9aab`다. [Hub API](https://huggingface.co/api/models/nomic-ai/nomic-embed-text-v1.5) 조회에서 다운로드 12,203,372·좋아요 986을 확인했고 [페이지](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)는 다운로드 기간을 last month로 표시한다. 다운로드는 사용 관심도이며 검색 품질의 증거가 아니다. 기존 models·issues 전체 파일을 읽어 동일 모델·계열의 구조 분석이 없음을 확인했다. 새로운 개발 조직 Nomic의 임베딩 모델을 선정했다.

영어 검색·군집·분류용 텍스트 encoder이고 Apache-2.0 라이선스다. 카드상 기반은 `nomic-ai/nomic-bert-2048`이며 v1의 학습 파이프라인에 Matryoshka 표현 학습을 적용한 계열이다. 조사한 v1.5 이름의 공개 recipe는 `nomic-ai/nomic-embed-text-v1-unsupervised`에서 시작한다. 따라서 이미 지도학습을 마친 v1 가중치를 반드시 이어 학습했다고 단정하지 않았다. [고정 카드](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/blob/e9b6763023c676ca8431644204f50c2b100d9aab/README.md), [고정 recipe](https://github.com/nomic-ai/contrastors/blob/613ddfd37309e538cceadb05b1e6423e7b09f603/src/contrastors/configs/train/contrastive_matryoshka.yaml).

v1 기술 보고서의 날짜를 v1.5 발표일로 쓰지 않았다. 카드가 연결한 v1.5 발표 블로그는 이번 조사에서 접근하지 못해 정확한 발표일은 **확인 불가**다. 조사일·HF revision과 구분한다.

## 구조와 임베딩 생성 경로

아래 구조값은 [고정 config](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/blob/e9b6763023c676ca8431644204f50c2b100d9aab/config.json), [문장 길이 설정](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/blob/e9b6763023c676ca8431644204f50c2b100d9aab/sentence_bert_config.json), [pooling](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/blob/e9b6763023c676ca8431644204f50c2b100d9aab/1_Pooling/config.json), [모듈 구성](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/blob/e9b6763023c676ca8431644204f50c2b100d9aab/modules.json)을 확인했다.

| 항목 | 값·의미 |
|---|---|
| 종류 | NomicBertModel, 비인과적 dense encoder |
| 총 파라미터 | 136,731,648, API safetensors 메타데이터 |
| 레이어 / hidden | 12 / 768 |
| FFN intermediate | config intermediate_size·n_inner 모두 3,072 |
| attention heads / head dimension | 12 / 64, config 명시 |
| KV heads | 별도 KV head 축소값은 config에 없음; MHA 계열 |
| 활성화 | activation_function=swiglu, hidden_act=silu |
| 정규화 | LayerNorm, use_rms_norm=false, prenorm=false |
| 위치 표현 | RoPE, theta=1,000, rotary fraction=1.0 |
| 위치·문장 길이 표기 | max_position_embeddings·max_trained_positions=2,048; n_positions 및 문장 인코딩 길이=8,192 |
| 어휘 슬롯 | 30,528, pad_vocab_size_multiple=64 |
| 출력 | mean pooling으로 768차원; 후처리해 512·256·128·64차원 사용 |
| MoE | 해당 모델 config에 전문가 라우팅 구성 없음 |
| 저장 dtype | float32, 학습 precision과 구분 |

원격 코드 매핑은 다른 저장소 `nomic-ai/nomic-bert-2048`를 가리킨다. [조회한 코드 revision](https://huggingface.co/nomic-ai/nomic-bert-2048/blob/7710840340a098cfb869c4f65e87cf2b1b70caca/modeling_hf_nomic_bert.py)은 읽기만 했으며 실행하지 않았다. 본 모델 revision만 고정해도 외부 코드까지 자동으로 같은 버전이 되는 것은 아니다. config의 intermediate 필드를 gated MLP의 모든 내부 행렬 폭으로 동일시하지 않고, 구현·런타임에 따라 해석이 필요한 설정값으로 표기했다.

입력 토큰들은 양방향 문맥을 반영한 뒤 padding을 제외하고 평균된다. 검색 문서는 `search_document:`, 질문은 `search_query:`를 붙인다. 군집과 분류에는 각각 `clustering:`·`classification:`을 사용한다. 같은 문장이라도 목적에 맞는 가까움의 의미가 달라 학습 시 접두사로 구분한다. [고정 카드](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/blob/e9b6763023c676ca8431644204f50c2b100d9aab/README.md).

### Matryoshka는 레이어를 덜 실행하는 기능이 아니다

카드의 경로는 **mean pooling → 전체 768차원 LayerNorm → 앞쪽 d개 선택 → L2 정규화**다. Matryoshka 학습은 앞부분만 남겨도 유용한 의미가 담기게 여러 차원에서 손실을 적용한다. 공개 recipe에는 `[64,128,256,512,768]`와 동일 가중치가 있다. 768차원을 임의로 순열한 뒤 앞부분만 쓰거나, 잘라낸 다음에 전체 벡터 LayerNorm을 대신하면 카드의 경로와 달라진다. [카드 코드](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/blob/e9b6763023c676ca8431644204f50c2b100d9aab/README.md), [recipe](https://github.com/nomic-ai/contrastors/blob/613ddfd37309e538cceadb05b1e6423e7b09f603/src/contrastors/configs/train/contrastive_matryoshka.yaml).

출력 256차원을 골라도 encoder의 hidden width는 768이고 12층은 그대로다. 벡터 저장·비교 비용을 줄이는 선택이지 encoder FLOPs나 가중치 크기를 1/3로 줄인다는 뜻은 아니다. modules.json은 Transformer와 Pooling 두 개뿐이며 차원 절단·정규화가 모두 자동 모듈로 등록됐다고 가정하지 않았다. [모듈 구성](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/blob/e9b6763023c676ca8431644204f50c2b100d9aab/modules.json).

### 2,048과 8,192를 구분하기

고정 카드의 장문 예제는 tokenizer 길이를 8,192로 바꾸고 dynamic RoPE, factor=2.0을 명시한다. 반면 고정 config의 rope_parameters는 default이며 rotary_scaling_factor는 null이다. 따라서 문장 설정의 max_seq_length=8192만 보고 장문 확장 설정이 모든 로딩 경로에서 자동 적용된다고 단정하지 않았다. 긴 입력을 사용할 때는 실제 tokenizer truncation·RoPE 설정을 함께 확인해야 한다. [카드](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/blob/e9b6763023c676ca8431644204f50c2b100d9aab/README.md), [config](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/blob/e9b6763023c676ca8431644204f50c2b100d9aab/config.json).

## 학습: v1 보고서와 v1.5 recipe의 적용 범위

계열 파이프라인은 masked language modeling(MLM) 기반 모델 → 약하게 관련된 쌍의 대조 사전학습 → 고품질 쌍·hard negative의 지도 대조학습이다. v1.5의 공개 recipe는 마지막 단계에서 여러 출력 차원의 손실을 적용한다. 다음 표는 실제 실행 로그가 아닌 **조회한 recipe 값**이며 해당 HF revision의 실행 인자 전체와 완전히 대응한다고 보장하지 않는다. [recipe](https://github.com/nomic-ai/contrastors/blob/613ddfd37309e538cceadb05b1e6423e7b09f603/src/contrastors/configs/train/contrastive_matryoshka.yaml).

| 항목 | v1.5 Matryoshka 지도 대조학습 recipe |
|---|---|
| 초기 모델 | nomic-embed-text-v1-unsupervised |
| epoch | num_epochs=1 |
| optimizer steps | 최종 총 step은 확인 불가; save_every=4500은 저장 간격 |
| batch / negatives | batch_size=256 / 쌍당 negatives=7 |
| 입력 길이 | seq_len=512 |
| learning rate / weight decay | 2e-5 / 0.01 |
| scheduler / warmup | linear / 400 steps |
| beta1 / beta2 / gradient norm | 0.9 / 0.999 / 1.0 |
| optimizer | recipe에 종류 명시 없음; 실제 실행 선택 확인 불가 |
| gradient accumulation | 확인 불가; chunk_size=32를 accumulation으로 해석하지 않음 |
| precision / 하드웨어 | v1.5 실제 실행값 확인 불가; use_fp8=false만으로 BF16·FP16을 확정하지 않음 |
| 데이터 | finetune_triplets.yaml 참조; v1.5 실행의 정확한 유효 샘플·토큰 수 확인 불가 |
| SFT·DPO·RL·LoRA | 생성형 SFT·선호학습·LoRA의 별도 적용 근거 확인 불가 |

[기술 보고서](https://arxiv.org/html/2402.01613v2)는 제목·초록상 **v1** 보고서다. v1의 약지도 단계는 1 epoch·batch 16,384·AdamW LR 2e-4, 지도 단계는 1 epoch·batch 256·LR 2e-5를 설명한다. 지도 단계 데이터는 약 1.6M datapoints이며 부록은 전체 v1 학습 자원으로 8×H100을 설명한다. 이를 v1.5의 전 단계 실측 이력으로 복사하지 않았다. v1.5 카드의 5M 시각화 sample 역시 전체 학습량이 아니다.

MLM 기반 학습의 epoch·총 step·배치·optimizer·학습률·정밀도·하드웨어를 이번 v1.5 checkpoint의 이력에 맞춰 검증하지 못해 **확인 불가**로 남긴다. 1 epoch recipe는 해당 학습 데이터 반복 설정이고, 12 layers는 구조 깊이, 출력 256은 벡터 차원이다. README 메타데이터의 `epoch_0_model`도 epoch 0번 학습했다는 근거가 아니다.

## 공개 평가·사용 판단

고정 카드의 MTEB 집계는 768차원 62.28, 512차원 61.96, 256차원 61.04, 128차원 59.34, 64차원 56.10이다. 같은 카드의 v1 768차원은 62.39다. v1.5의 장점은 모든 크기에서 점수가 높다는 주장이 아니라 저장 차원을 선택할 수 있다는 점이다. 이는 저자 공개값이며 이번 조사에서 재측정하지 않았다. [차원 비교표](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/blob/e9b6763023c676ca8431644204f50c2b100d9aab/README.md).

MTEB는 검색뿐 아니라 분류·군집·STS 등 여러 작업을 합친 결과다. 차원별 동일 평가 버전·prompt·전체 프로토콜의 대응까지 이번에 독립 재현하지 못했으며 최신 리더보드 순위라고 쓰지 않았다. v1 보고서는 FEVER·HotpotQA 등 BEIR 학습 split 사용과 ablation을 구분한다. 해당 보고서 점수를 모두 보지 않은 데이터에 대한 zero-shot이라고 단정하면 안 된다. [보고서 §4.2~5](https://arxiv.org/html/2402.01613v2).

영어 RAG 인덱스·군집·문서 유사도에 적합한 후보이며 한국어는 같은 데이터에서 다국어 모델과 비교해야 한다. [multilingual-e5 분석](2026-10-17-multilingual-e5-large.md)과 용도가 겹치지만 언어 범위·차원 선택·prefix가 달라 그대로 교체할 수 없다. 카드의 multimodal 설명은 별도 vision encoder가 같은 공간에 정렬됐다는 의미이며 이 텍스트 모델 하나가 이미지를 직접 입력받는다는 뜻은 아니다. [카드](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/blob/e9b6763023c676ca8431644204f50c2b100d9aab/README.md).

계산상 백만 문서의 F32 벡터 본체는 `1,000,000×768×4=3.072GB`, 256차원은 `1,000,000×256×4=1.024GB`다. 검색 인덱스 메타데이터·그래프는 제외한 십진 단위 계산이며 실측이 아니다. 모델 F32 가중치는 `136,731,648×4≈546.9MB`로 차원 절단만으로 줄지 않는다. 런타임·activation·배치 메모리도 별도다.
