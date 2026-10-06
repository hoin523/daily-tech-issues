# all-MiniLM-L6-v2: 문장을 384차원 벡터로 압축하는 모델

조사일: 2026-10-06 KST. 실행·학습 없이 공식 자료와 설정 파일을 분석했다.

## 선정과 식별

모델 ID는 `sentence-transformers/all-MiniLM-L6-v2`, revision은 `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`이다. [모델 API](https://huggingface.co/api/models/sentence-transformers/all-MiniLM-L6-v2)에서 다운로드 235,658,929, 좋아요 6,190을 확인했다. 웹 페이지는 다운로드 기간을 ‘last month’로 표시한다. 이는 채택 신호이며 품질 순위가 아니다. 기존 models·issues 전체 기록에 동일 모델 분석은 없다.

용도는 영어 문장 유사도·검색·군집화이며 라이선스는 Apache-2.0이다. 정확한 최초 발표일은 이번 조사에서 확인 불가다. 기반 모델은 `nreimers/MiniLM-L6-H384-uncased`로, 원본 카드에 따르면 12-layer MiniLM에서 한 층씩 건너뛰어 6개 층을 남긴 모델이다. [기반 모델 카드](https://huggingface.co/nreimers/MiniLM-L6-H384-uncased).

## 구조

표의 설정값은 고정 revision의 [config.json](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/config.json), [문장 길이 설정](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/sentence_bert_config.json), [pooling 설정](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/1_Pooling/config.json), [모듈 구성](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/modules.json)을 직접 확인했다.

| 항목 | 값 |
|---|---|
| 종류 | BertModel 기반 dense encoder |
| 파라미터 | 약 22.7M, Hub 표시값 |
| Transformer 레이어 | 6 |
| hidden / FFN intermediate | 384 / 1,536 |
| attention heads | 12, 별도 GQA KV 설정 없음 |
| head dimension | 384 ÷ 12 = 32, 표준 BERT 분할 가정의 계산값 |
| 위치 표현 | absolute, 최대 위치 슬롯 512 |
| 문장 인코딩 기본 길이 | 256 word pieces, 초과분 잘림 |
| 어휘 슬롯 | 30,522 |
| 활성화 | GELU |
| 후단 | mean pooling → Normalize |
| 출력 | 384차원 문장 벡터 |
| MoE·멀티모달 | 해당 없음 |

파라미터 표시값의 출처는 [모델 페이지](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)다. Dense 모델이라 전문가를 선택하는 활성 파라미터 구분은 적용하지 않는다.

각 단어 조각은 encoder에서 양방향 문맥을 반영한다. 이후 padding을 제외한 토큰 표현을 평균내고 길이가 1인 벡터로 정규화한다. 따라서 단순 AutoModel 출력이나 CLS 토큰을 그대로 저장하면 공개된 문장 인코딩 파이프라인과 달라진다. [고정 pooling 및 모듈 구성](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/modules.json).

512는 위치 임베딩 슬롯, 256은 배포 파이프라인 기본 입력 길이다. 학습 카드의 128-token 길이는 또 다른 설정이며 서로 치환할 수 없다.

## 학습

후처리는 문장 쌍을 구별하는 대조학습이다. 배치 안에서 실제 짝이 가까워지도록 학습한다. 아래 수치는 [고정 모델 카드](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/README.md)의 학습 설명이다.

| 항목 | 문장 임베딩 fine-tuning | 기반 사전학습 |
|---|---|---|
| epoch | 미공개; step으로 대체해 추정하지 않음 | 확인 불가 |
| step | 100,000 | 확인 불가 |
| 데이터 | 약 11.7억 학습 튜플, 가중 샘플링 | 확인 불가 |
| batch | 1,024, TPU core당 128 | 확인 불가 |
| optimizer / LR | AdamW / 2e-5 | 확인 불가 |
| warmup | 500 steps | 확인 불가 |
| 학습 입력 길이 | 128 tokens | 확인 불가 |
| 하드웨어 | TPU v3-8 | 확인 불가 |
| gradient accumulation / precision | 확인 불가 | 확인 불가 |
| DPO·RL·LoRA | 적용 근거 확인 불가 | 적용 근거 확인 불가 |

[저장소 train_script.py](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/train_script.py)에는 linear warmup scheduler가 있지만 예시 명령은 1,000,000 steps를 포함한다. 공개 카드의 실제 모델 설명인 100,000 steps와 구분한다. 스크립트 기본값·예시가 이 체크포인트의 실제 실행 인자라는 증거는 없다.

계산상 `100,000 × 1,024 = 102,400,000` 쌍 처리에 해당한다. 이는 매 step의 global batch가 그대로 유지됐다고 가정한 값이다. 가중 샘플링과 재사용 때문에 고유 쌍 수나 정확한 epoch가 아니다.

## 비교·평가·활용

[Sentence Transformers 공식 문서](https://sbert.net/docs/sentence_transformer/pretrained_models.html)는 all-mpnet-base-v2를 품질 중심 선택지로, 이 모델을 빠른 대안으로 설명한다. 문서의 속도 비교는 여기서 재측정하지 않았고 동일 하드웨어·배치·입력 길이 조건도 이번 조사에서 확인 불가다. 따라서 ‘항상 몇 배 빠르다’는 배포 보장으로 사용하지 않는다. 본 분석에서는 개별 평가 데이터셋·프로토콜을 직접 대조하지 못해 점수 순위를 싣지 않았다.

전날 분석한 Qwen3-8B는 답변을 생성하는 decoder이며, 이 모델은 검색용 벡터를 만드는 encoder다. 두 모델은 결과물과 평가 목적이 달라 생성 벤치마크로 우열을 비교할 수 없다. [Qwen3-8B 분석](2026-10-05-qwen3-8b.md).

짧은 영어 문서의 검색 인덱스나 유사 문장 묶기부터 검증하기 좋다. 한국어 중심·긴 문서에서는 다국어 모델과 chunking 전략을 함께 비교하고 검색 recall@k 등 실제 목표 지표를 측정해야 한다. 다운로드 수는 이 검증을 대신하지 않는다.

F32 가중치만 가정하면 `22.7M × 4 bytes ≈ 90.8 MB`다. 이는 계산값이며 메모리 실측이 아니다. 토크나이저·activation·배치와 검색 인덱스 메모리는 별도다. 가중치 다운로드나 추론은 수행하지 않았다.
