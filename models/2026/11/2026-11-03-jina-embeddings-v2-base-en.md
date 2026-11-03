# Jina Embeddings v2 base-en: 12층 양방향 ALiBi와 장문 평균 풀링

조사일: 2026-11-03 KST. 가중치 다운로드·추론·학습 없이 공개 자료를 분석했다.

## 선정과 식별

모델 ID는 `jinaai/jina-embeddings-v2-base-en`, revision은 `322d4d7e2f35e84137961a65af894fda0385eb7a`다. [Hub API](https://huggingface.co/api/models/jinaai/jina-embeddings-v2-base-en)는 다운로드 357,283·좋아요 739를 반환했고 [페이지](https://huggingface.co/jinaai/jina-embeddings-v2-base-en)는 다운로드 기간을 last month로 표시했다. 인기도는 검색 품질의 증거가 아니다. 기존 models·issues 전체 파일을 읽어 동일 모델·가중치 변형 분석이 없음을 확인했다. 이번에는 Jina AI의 영어 임베딩 모델을 선정했다.

용도는 문장·문서 검색, 군집화와 유사도이며 라이선스는 Apache-2.0이다. 기반 JinaBERT는 C4로 사전학습됐고 이후 문장 쌍·hard negative 대조학습을 거친다. 생성형 instruct 모델이 아니다. [고정 카드](https://huggingface.co/jinaai/jina-embeddings-v2-base-en/blob/322d4d7e2f35e84137961a65af894fda0385eb7a/README.md). 연구 보고서의 최초 제출은 [arXiv 기록](https://arxiv.org/abs/2310.19923) 기준 2023-10-30이며 HF 가중치의 정확한 최초 배포일은 이번 조사에서 확인 불가다.

## 구조

[고정 config](https://huggingface.co/jinaai/jina-embeddings-v2-base-en/blob/322d4d7e2f35e84137961a65af894fda0385eb7a/config.json), [pooling](https://huggingface.co/jinaai/jina-embeddings-v2-base-en/blob/322d4d7e2f35e84137961a65af894fda0385eb7a/1_Pooling/config.json), [모듈 구성](https://huggingface.co/jinaai/jina-embeddings-v2-base-en/blob/322d4d7e2f35e84137961a65af894fda0385eb7a/modules.json) 및 [고정 외부 구현](https://huggingface.co/jinaai/jina-bert-implementation/blob/f3ec4cf7de7e561007f27c9efc7148b0bd713f81/modeling_bert.py)을 대조했다. Config가 외부 `jinaai/jina-bert-implementation` 코드를 가리키므로 모델 revision 고정만으로 그 코드 revision까지 고정되는 것은 아니다. 코드는 읽기만 했고 실행하지 않았다.

| 항목 | 확인값·해석 |
|---|---|
| 구조 | JinaBERT 기반 양방향 dense encoder |
| 총 파라미터 | 137,368,320, API의 F16 메타데이터 |
| 레이어 / hidden | 12 / 768 |
| attention Q·K·V heads | 각각12, 구현의 별도 Q·K·V projection 확인 |
| head dimension | 768÷12=64, 구현과 일치하는 계산 |
| FFN intermediate | 3,072 |
| FFN 종류 | GEGLU; 합동 projection 6,144를 두 3,072 갈래로 분할 |
| 정규화 | post-LayerNorm, eps=1e-12 |
| 위치 처리 | symmetric bidirectional ALiBi |
| 최대 길이 | config model_max_length·max_position_embeddings=8,192 |
| 어휘 슬롯 | 30,528 |
| 출력 경로 | mean pooling → Normalize, 768차원 |
| MoE·GQA | 전문가 라우팅·KV head 축소 없음 |
| dtype | config는 float32, API 저장 tensor는 F16; 두 표기를 구분 |

Config의 architectures는 `JinaBertForMaskedLM`로 남아 있지만 AutoModel 매핑과 모듈 구성은 임베딩용 JinaBertModel·평균 풀링·정규화 경로를 제공한다. 이름 하나만 보고 임베딩 출력이 MLM 어휘 확률이라고 결론 내리지 않았다. [config](https://huggingface.co/jinaai/jina-embeddings-v2-base-en/blob/322d4d7e2f35e84137961a65af894fda0385eb7a/config.json), [모듈 구성](https://huggingface.co/jinaai/jina-embeddings-v2-base-en/blob/322d4d7e2f35e84137961a65af894fda0385eb7a/modules.json).

GEGLU는 GELU를 적용한 갈래와 다른 갈래를 곱해 정보를 선택하고 768차원으로 다시 투영한다. 3,072라는 필드를 합동 projection 폭 6,144와 혼동하면 파라미터 구조를 잘못 읽는다. Padding을 제외한 토큰 평균을 구하고 정규화해야 공개 문장 임베딩 경로에 맞는다. CLS 토큰만 그대로 저장하는 방식과 다르다. [구현](https://huggingface.co/jinaai/jina-bert-implementation/blob/f3ec4cf7de7e561007f27c9efc7148b0bd713f81/modeling_bert.py), [카드](https://huggingface.co/jinaai/jina-embeddings-v2-base-en/blob/322d4d7e2f35e84137961a65af894fda0385eb7a/README.md).

## ALiBi가 장문을 다루는 방식과 한계

ALiBi는 학습 가능한 절대 위치 벡터를 더하는 대신 attention score에 토큰 거리의 선형 bias를 더한다. 이 encoder는 거리 bias를 양방향 대칭으로 적용하고 head마다 다른 기울기를 사용한다. 가까운 위치를 선호하되 먼 위치를 완전히 차단하는 local window는 아니다. [논문 §4.1](https://arxiv.org/html/2310.19923v2#S4.SS1).

카드는 학습 입력은 512 tokens, 추론 지원은 8,192로 구분한다. ‘8K 지원’은 8K 길이로 사전학습했다는 뜻이 아니다. 또한 위치 extrapolation이 가능해도 dense attention의 메모리 비용이 사라지지는 않는다. 입력 512→8192는 길이 16배, 단순 attention score 원소 수는 `16²=256`배가 되는 계산이다. 실제 메모리·시간은 커널·배치에 따라 달라지며 이 배수를 실측 VRAM 증가율로 쓰지 않았다.

[최근 Nomic 분석](../10/2026-10-28-nomic-embed-text-v1-5.md)은 RoPE 확장과 Matryoshka 출력 차원 선택을 설명했다. 이 모델의 핵심은 ALiBi와 고정 768차원 평균 표현이며 Nomic식 차원 절단 절차가 같은 품질로 적용된다는 근거는 없다.

## 학습 단계별 공개값

[논문 §4~5](https://arxiv.org/html/2310.19923v2#S4)는 MLM backbone, 약지도 문장 쌍, 지도 hard negative의 세 단계를 설명한다. 표는 보고서 설정이며 해당 HF revision의 원 실행 로그를 직접 재현한 것은 아니다.

| 항목 | C4 backbone MLM | 약지도 대조학습 | 지도 hard-negative 학습 |
|---|---|---|---|
| 데이터 | 약365M 문서·170B tokens의 C4 원 코퍼스 설명 | 약40개 출처 문장 쌍 | MSMarco·NQ·NLI 등 |
| 목적 | 30% whole-word masking, NSP 제외 | 양방향 InfoNCE, temperature0.05 | 양성1·음성15를 포함한 대조학습 |
| 입력 길이 | 최대512, 문서 첫512만 사용 | 최대512 | 최대512 |
| epoch | 정확한 값 확인 불가 | 확인 불가 | 확인 불가 |
| optimizer step | 보고서100,000-step 학습 스케줄 | 총 step 확인 불가 | 총 step 확인 불가 |
| batch | global4,096 documents | 해당 단계 확정값 확인 불가 | 해당 단계 확정값 확인 불가 |
| optimizer / LR | AdamW / base peak6e-4 | 단계별 설정 확인 불가 | 단계별 설정 확인 불가 |
| scheduler | 10,000-step warmup·linear decay | 확인 불가 | 확인 불가 |
| betas·epsilon·WD | (0.9,0.98)·1e-6·0.01 | 확인 불가 | 확인 불가 |
| gradient accumulation | 확인 불가 | 확인 불가 | 확인 불가 |
| precision | FP16 dynamic mixed precision | mixed precision, 정확한 dtype 확인 불가 | mixed precision, 정확한 dtype 확인 불가 |
| 하드웨어 | 해당 base 실행 GPU 수·기종 확인 불가 | 확인 불가 | 확인 불가 |

C4 전체 170B tokens를 이 모델이 실제 소비한 토큰 수로 쓰지 않았다. 보고서는 문서당 첫512만 쓰고 한 문서를 여러 chunk로 나누지 않는다고 설명하므로 전체 코퍼스 규모·실제 학습량은 다르다. 카드의 400M 이상 문장 쌍·hard negative collection도 고유한 실제 처리량이나 epoch와 동일시하지 않는다. [논문 §4.2~4.3](https://arxiv.org/html/2310.19923v2#S4), [카드](https://huggingface.co/jinaai/jina-embeddings-v2-base-en/blob/322d4d7e2f35e84137961a65af894fda0385eb7a/README.md).

단계별 세부 미확인 값은 **확인 불가**로 남겼으며 모든 자료에 미공개라고 단정하지 않았다. 논문의 GLUE 평가용 10 epochs는 작업별 분류 fine-tuning 프로토콜이므로 이 임베딩 checkpoint의 학습 epoch로 넣지 않았다. 별도 생성형 SFT·DPO·RL·LoRA 적용 근거도 확인 불가다. Config의 attention dropout0.0과 논문 학습 설명의0.1을 원 학습 이력이 완전히 동일하다는 증거로 취급하지 않았다.

## 평가와 활용 판단

보고서 MTEB 표3의 Jina base-v2는 평균60.38, retrieval nDCG@10 집계47.87, STS Spearman 집계80.7이다. all-MiniLM-L6-v2의 평균은56.26이다. 저자 공개값이며 이번 조사 실측이나 최신 순위가 아니다. 집계별 지표가 달라 한 점수만으로 모든 영어 검색 환경의 우위를 보장하지 않는다. [논문 §6.2.1](https://arxiv.org/html/2310.19923v2#S6.SS2.SSS1).

보고서는 긴 문서 평가를 위해 NarrativeQA·PatentClustering·WikiCitiesClustering 등을 추가하며 입력 상한을 바꿔 truncation 영향을 비교한다. 다만 장문 비교에서 e5-base-v2에 권장 prefix를 사용하지 않았다고 각주로 명시한다. 이 조건의 비교를 모든 모델의 최적 설정끼리 평가한 결과라고 쓰지 않았다. 또한 짧은 MTEB 집계가 장문 품질을 완전히 설명하지 않는다. [논문 §6.2.2 및 그림3 각주](https://arxiv.org/html/2310.19923v2#S6.SS2.SSS2).

영어 긴 문서 검색·유사도·군집화를 위한 후보이며, 단일 벡터가 문서의 세부 근거를 모두 보존하는 것은 아니다. Chunking과 전체 문서 인코딩을 실제 recall@k·지연 시간으로 비교하는 것이 좋다. 한국어 중심 업무에는 별도의 다국어 모델과 검증해야 한다. [multilingual-e5 분석](../10/2026-10-17-multilingual-e5-large.md)처럼 언어 범위가 다른 모델과 체크포인트 크기만으로 비교하지 않는다.

API의 F16 항목을 기준으로 가중치 저장량은 `137,368,320×2=274,736,640 bytes`, 약274.7MB(십진 단위)다. VRAM 실측·최소 요구량이 아니며 장문 attention·activation·배치·런타임 메모리는 별도다.
