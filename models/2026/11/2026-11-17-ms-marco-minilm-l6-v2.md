# MS MARCO MiniLM-L6 v2: 6층 cross-encoder로 검색 후보 다시 정렬하기

조사일: 2026-11-17 (한국 시간). 공개 카드·구성·공식 문서·코드를 분석했으며 가중치 다운로드·학습·추론은 실행하지 않았다.

## 선정과 식별

Hugging Face API에서 최근 한 달 다운로드 **57,878,163회**, 누적 좋아요 **446개**를 확인했다. 이미지 생성에 이어 검색의 재정렬 단계를 선택했다. 다운로드는 성능 증거나 현재 트렌딩 순위가 아니다. [API](https://huggingface.co/api/models/cross-encoder/ms-marco-MiniLM-L6-v2), [모델 페이지](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)

| 항목 | 확인 내용 |
|---|---|
| ID / 배포 | `cross-encoder/ms-marco-MiniLM-L6-v2` / Sentence Transformers Cross-Encoders |
| revision | `233902d25c440f23af6f7d6e94d2946bac0bee0a` |
| 기반 | 카드 메타데이터의 base_model은 `cross-encoder/ms-marco-MiniLM-L12-v2`. 구체적인 초기화·증류 실행 경로는 확인 불가. |
| 용도 / 언어 | query·passage 쌍에 대한 관련성 점수, 영어 |
| 라이선스 | Apache-2.0 |
| 발표 / 조사 | 기반 MiniLM 논문 최초 제출 2020-02-25 / 조사 2026-11-17. 이 reranker v2 자체의 최초 공개일은 확인 불가. |

근거: [고정 카드](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2/blob/233902d25c440f23af6f7d6e94d2946bac0bee0a/README.md), [MiniLM 원 논문](https://arxiv.org/abs/2002.10957).

기존 `models`·`issues` 전체를 읽었으며 동일 reranker·revision·미러·양자화 분석은 없었다. 기존 all-MiniLM 문장 임베딩과 구조 계보는 관련되지만 출력·학습 과제가 다른 모델이다.

## 구조: 두 텍스트를 함께 읽고 점수 하나 출력

| 항목 | 확인값 |
|---|---|
| 유형 | BERT 계열 양방향 encoder + sequence classification head |
| layer / hidden | 6 / 384 |
| attention / KV heads | 표준 MHA Q·K·V 각각 12개 |
| head dimension | `384÷12=32`, 계산값 |
| FFN / 활성화 | 1,536 / GELU |
| 입력 길이 | 최대 position 512, query·passage·특수 토큰의 합계 |
| 위치 / segment | absolute position embedding / token type vocabulary 2 |
| vocabulary | 30,522 |
| dropout / LN epsilon | attention·hidden 각각0.1 / 1e-12 |
| 출력 | 단일 LABEL_0 logit, 기본 CrossEncoder activation 설정 Identity |
| 저장 텐서 규모 | API F32 원소 22,713,601개와 I64 512개. I64 buffer를 학습 파라미터에 더하지 않음. trainable 수는 직접 집계하지 않았다. |
| MoE·생성 decoder | 없음 |

근거: [고정 config](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2/blob/233902d25c440f23af6f7d6e94d2946bac0bee0a/config.json), [BERT sequence classifier 공식 구현](https://github.com/huggingface/transformers/blob/v4.4.2/src/transformers/models/bert/modeling_bert.py), [API](https://huggingface.co/api/models/cross-encoder/ms-marco-MiniLM-L6-v2). config의 `_name_or_path`에는 L12 흔적이 있지만 실제 깊이는 `num_hidden_layers=6`으로 확인했다. 이름 문자열보다 구조 필드를 기준으로 삼았다.

query와 passage를 한 시퀀스로 입력하므로 각 layer의 attention이 두 텍스트 사이의 관련 단서를 함께 읽는다. pooled 표현을 분류 head가 점수 하나로 바꾼다. 문서마다 독립된 검색 벡터를 미리 만들어 두는 bi-encoder와 다르다. [카드 입력 예제](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2/blob/233902d25c440f23af6f7d6e94d2946bac0bee0a/README.md), [공식 설명](https://sbert.net/examples/cross_encoder/training/ms_marco/README.html)

점수는 기본 설정에서 0~1 확률로 제한되지 않는다. 카드 예제도 양수·음수 점수를 반환한다. sigmoid를 적용해 범위를 바꿀 수는 있지만 그것만으로 확률이 보정되지는 않는다. 같은 query의 후보를 내림차순 정렬하는 용도로 해석하는 것이 적절하다. [카드](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2/blob/233902d25c440f23af6f7d6e94d2946bac0bee0a/README.md), [출력 activation](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2/blob/233902d25c440f23af6f7d6e94d2946bac0bee0a/config.json)

512는 각 텍스트별 한도가 아니라 쌍을 구성한 전체 길이다. 긴 passage는 truncation으로 중요한 뒷부분을 잃을 수 있다. layer 6개는 네트워크 깊이이며 epoch와 무관하다.

## 학습 이력과 공개 범위

| 단계·항목 | 확인한 내용 |
|---|---|
| 기반 MiniLM | teacher의 attention 지식을 작은 학생으로 전달하는 연구 계열. 이 reranker의 원 사전학습 실행 조건은 확인 불가. |
| ranking 학습 | 카드상 MS MARCO Passage Ranking |
| 데이터 규모 | 공식 문서는 원 데이터셋을 약 500k 검색 query로 설명. 실제 이 모델에 사용한 쌍·negative 수는 확인 불가. |
| epoch / 총 step | 고정 카드에서 미공개 |
| batch / accumulation | 해당 실행 대응값 확인 불가 |
| optimizer / LR / scheduler | 해당 실행 대응값 확인 불가 |
| precision / 학습 하드웨어 | 해당 실행 대응값 확인 불가 |
| SFT·DPO·RL·LoRA | LLM 대화 SFT·DPO 등의 단계로 보고되지 않음. 지도 ranking 학습과 구분. |

근거: [고정 카드](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2/blob/233902d25c440f23af6f7d6e94d2946bac0bee0a/README.md), [MiniLM 논문](https://arxiv.org/abs/2002.10957), [공식 MS MARCO 학습 문서](https://sbert.net/examples/cross_encoder/training/ms_marco/README.html).

현재 학습 문서는 BCE·ListNet·LambdaLoss 등 여러 재현 예제를 제공한다. 그중 한 예제의 기본값을 과거 v2 가중치의 확정 recipe로 복사하지 않았다. 분류 head라는 사실만으로 실제 원 학습 loss까지 확정할 수 없다. 원 모델에 연결된 실행 로그·고정 recipe는 이번 조사에서 확인하지 못했다.

## 평가와 활용 범위

고정 카드의 보고값은 TREC DL 2019 **NDCG@10 74.30**, MS MARCO Dev **MRR@10 39.01**이다. NDCG와 MRR은 순위 품질의 서로 다른 지표이며 분류 정확도가 아니다. 같은 표의 L12-v2는 각각74.31·39.02로 기록되지만 다른 데이터·한국어·장문에서도 차이가 같다는 보장은 없다. [고정 평가 표](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2/blob/233902d25c440f23af6f7d6e94d2946bac0bee0a/README.md#performance)

카드는 V100에서 초당1,800문서라는 속도도 보고한다. 이번 환경의 처리량이 아니고 batch·입력 길이 등 세부 조건을 완전히 확인하지 못했으므로 운영 속도 보장에 사용하지 않는다. [카드](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2/blob/233902d25c440f23af6f7d6e94d2946bac0bee0a/README.md#performance)

BM25나 [GTE-small](2026-11-09-gte-small.md) 같은 임베딩 검색기로 후보를 좁힌 뒤 rerank하는 데 활용할 수 있다. 후보가 K개면 query·passage K쌍을 평가하는 계산이 필요하다는 것이 구조에서 도출한 해석이다. 후보 검색에서 빠진 문서를 reranker가 찾아낼 수는 없다. [공식 사용 설명](https://sbert.net/examples/cross_encoder/training/ms_marco/README.html)

[all-MiniLM-L6-v2](../10/2026-10-06-all-minilm-l6-v2.md)와 layer·hidden이 비슷해도 해당 모델의 문장 벡터를 이 모델의 단일 점수로 대체할 수 없다. 질문 관련성과 답변의 사실 정확성도 다른 문제다. 이번에는 검색 품질·VRAM·속도를 실측하지 않았다.

출처·값·중복 여부를 검토하고 계산·해석과 원문 보고값을 구분했다.
