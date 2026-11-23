# BGE-small-en-v1.5: 12층 BERT와 CLS pooling으로 만드는 384차원 검색 벡터

조사일: 2026-11-23 (한국 시간). 공개 자료 분석이며 가중치 다운로드·추론·학습은 수행하지 않았다.

## 선정과 식별

Hub API에서 최근 한 달 다운로드 **43,139,469회**, 누적 좋아요 **683개**를 확인했다. 사용 관심도이며 품질이나 사용자 수의 증거는 아니다. Qwen 텍스트 생성에서 BAAI의 임베딩으로 분야·조직을 바꿨다. 기존 models·issues 52개 기록을 모두 읽고 동일 모델·revision·미러·양자화의 분석이 없음을 확인했다. 기존 BGE-M3와는 다른 영어 소형 dense 모델이다. [Hub API](https://huggingface.co/api/models/BAAI/bge-small-en-v1.5)

| 항목 | 내용 |
|---|---|
| 모델 / 조직 | `BAAI/bge-small-en-v1.5` / Beijing Academy of Artificial Intelligence |
| revision | `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` |
| 기반 | BGE-small-en 계열, BERT 구조. 정확한 초기 기반 체크포인트 ID 확인 불가 |
| 용도·언어 | 영어 dense retrieval·의미 유사도 |
| 발표 / 조사 | 카드의 v1.5 발표 2023-09-12 / 조사 2026-11-23 |
| 라이선스 | 카드 License 절은 FlagEmbedding MIT와 무료 상업 사용을 명시. 이 revision의 독립 LICENSE 파일은 확인 불가 |

[고정 카드](https://huggingface.co/BAAI/bge-small-en-v1.5/blob/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/README.md), [config](https://huggingface.co/BAAI/bge-small-en-v1.5/blob/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/config.json). 논문 공개일과 모델 업데이트 날짜를 구분한다. 관련 [C-Pack 논문](https://arxiv.org/abs/2309.07597)은 중국어 중심 자원과 영어 확장도 다루며, 중국어 모델의 학습값을 영어 small v1.5에 그대로 적용하지 않는다.

## 구조와 표현 생성

| 항목 | 확인값 |
|---|---|
| 저장 텐서 수 | Hub F32 33,360,000개 + I64 512개. 정수 원소를 학습 파라미터로 합산하지 않음 |
| 유형 | Dense 양방향 BERT encoder, 12층 |
| hidden / FFN | 384 / 1,536, GELU |
| attention / KV heads | 표준 MHA 12개, KV도 12개 |
| head dimension | `384÷12=32`: 계산값 |
| 입력 길이 | 최대 512 position, sentence-transformers max_seq_length도 512 |
| 어휘 / token types | 30,522 / 2 |
| 위치 정보 | 학습 가능한 absolute position embedding |
| dropout / LayerNorm | hidden·attention 0.1 / epsilon 1e-12 |
| pooling | CLS token 사용, mean·max pooling 사용 안 함 |
| 문장 벡터 | 384차원, 유사도 예제는 L2 정규화 후 내적 |
| MoE·decoder·이미지 encoder | 없음 |

[config](https://huggingface.co/BAAI/bge-small-en-v1.5/blob/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/config.json), [pooling](https://huggingface.co/BAAI/bge-small-en-v1.5/blob/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/1_Pooling/config.json), [입력 길이](https://huggingface.co/BAAI/bge-small-en-v1.5/blob/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/sentence_bert_config.json), [사용 코드](https://huggingface.co/BAAI/bge-small-en-v1.5/blob/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/README.md), [메타데이터](https://huggingface.co/api/models/BAAI/bge-small-en-v1.5). 저장 텐서 메타데이터를 확인했으며 실제 trainable 계수는 로딩으로 집계하지 않았다.

각 토큰은 문장 양쪽의 문맥을 참고한다. 마지막 층의 CLS 벡터를 문장 대표로 꺼내 정규화하면 방향을 비교할 수 있다. 문서를 미리 벡터화해 두고 질문 벡터와 가까운 문서를 찾는 방식이다. 문장 두 개를 한 번에 읽는 cross-encoder와 달리 문서 벡터를 재사용할 수 있다. [카드의 pooling·검색·reranker 설명](https://huggingface.co/BAAI/bge-small-en-v1.5/blob/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/README.md)

단일 벡터를 F32로 저장한다고 가정하면 `384×4=1536 bytes`다. 이는 벡터 값만의 계산이며 인덱스·문서 metadata·프로그램 메모리를 포함하지 않는다. 임의로 차원을 잘라도 품질을 유지하는 Matryoshka 학습이라고 보고된 모델은 아니다.

## 학습 공개 범위

| 항목 | 확인한 범위 |
|---|---|
| 사전학습 목적 | 카드: RetroMAE 기반 텍스트 복원 |
| 검색 학습 | 카드: 대규모 pair 데이터에 대한 contrastive learning |
| 데이터 규모·구성 | 해당 영어 small v1.5 실행에 대응하는 정확한 pair·토큰 수 확인 불가 |
| epoch / optimizer step | 개별 체크포인트 명시값 확인 불가 |
| batch / gradient accumulation | 개별 실행값 확인 불가 |
| LR / scheduler / optimizer | 원 실행 설정 확인 불가 |
| precision / 하드웨어 | 배포 F32. 실제 학습 정밀도·GPU 종류·수 확인 불가 |
| SFT·DPO·RL·LoRA | 해당 단계 적용은 미공개. contrastive fine-tuning을 대화 SFT와 혼용하지 않음 |

[카드 Train 절](https://huggingface.co/BAAI/bge-small-en-v1.5/blob/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/README.md), [관련 논문](https://arxiv.org/abs/2309.07597). 일반 후속 학습 예제는 제공되지만 이를 이 revision의 실제 실행 로그로 취급하지 않았다. 12층·512 position에서 epoch를 추정하지 않는다.

복원 사전학습은 입력 문맥을 배우고, contrastive 학습은 관련 문장 쌍을 가깝게 만드는 별도 목적이다. 카드도 복원 사전학습만 끝난 모델을 바로 유사도 계산에 쓰지 말고 contrastive 학습이 필요하다고 설명한다. FAQ의 temperature 0.01 설명은 BGE 유사도 분포 문제의 맥락이며 v1.5 전체 실행 설정을 복원할 근거로 쓰지 않았다. [Train·FAQ](https://huggingface.co/BAAI/bge-small-en-v1.5/blob/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/README.md)

## 검색 지침과 평가 해석

짧은 질문으로 긴 문서를 검색할 때 카드 권장 영어 query prefix는 `Represent this sentence for searching relevant passages: `이다. 문서에는 붙이지 않는다. v1.5는 prefix 없이도 사용할 수 있도록 개선됐다고 설명하지만 실제 서비스 데이터에서 두 조건을 비교해야 한다. [모델 목록·FAQ](https://huggingface.co/BAAI/bge-small-en-v1.5/blob/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/README.md)

유사도 0.8을 정답 확률 80%로 읽으면 안 된다. 카드는 절대 점수보다 상대 순위가 중요하며 임계값은 사용자 데이터 분포에서 정하라고 안내한다. 정규화 여부, query prefix, truncation과 pooling 설정을 고정해야 검색 결과를 비교할 수 있다. [FAQ·사용 코드](https://huggingface.co/BAAI/bge-small-en-v1.5/blob/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/README.md)

고정 카드의 당시 MTEB 표는 영어 56개 task 평균 **62.17**, retrieval 15개 평균 **51.68**을 보고한다. 최신 leaderboard 순위나 정답률 62.17%라는 뜻은 아니다. 개별 metadata 예로 ArguAna test의 nDCG@10은 **59.55**지만 dataset revision은 None으로 표시되어 재현 추적에 한계가 있다. 전체 집계와 개별 지표는 구분해야 한다. 모두 개발 측 보고값이며 이번에 재평가하지 않았다. [평가 표·metadata](https://huggingface.co/BAAI/bge-small-en-v1.5/blob/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/README.md)

같은 표의 base v1.5는 768차원·평균 63.55다. 이 표에서는 small보다 높지만 서비스별 품질·속도를 보장하지 않는다. 기존 [GTE-small](2026-11-09-gte-small.md)과 출력 차원이 같아도 벡터 공간은 다르므로 인덱스를 섞어 쓰지 않는다. 기존 [BGE-M3](../10/2026-10-11-bge-m3.md)의 다국어·장문·복수 검색 방식도 오늘 모델의 기능으로 옮기지 않는다.

분석상 영어 RAG 후보 검색·유사 문서 탐색에 활용할 수 있다. 512 position을 넘는 문서는 chunking과 경계 처리 검증이 필요하며 한국어 검색 품질은 별도 평가해야 한다. 속도·VRAM·사용자 데이터 정확도는 실측하지 않았다.
