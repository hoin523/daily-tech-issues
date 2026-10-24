# ModernBERT-base: 22층에서 지역·전역 attention을 번갈아 쓰는 장문 encoder

조사일: 2026-10-24 KST. 가중치를 내려받거나 추론·학습을 실행하지 않고 공개 설정·논문·구현을 분석했다.

## 선정과 식별

- 모델 ID: `answerdotai/ModernBERT-base`. Answer.AI·LightOn 등의 공동 개발이며, 영어·코드용 masked language model(MLM)이다.
- 고정 revision: `8949b909ec900327062f0ebf497f51aef5e6f0c8`.
- 발표: [개발팀 공개 글](https://huggingface.co/blog/modernbert)은 2024-12-19, [논문 v1](https://arxiv.org/html/2412.13663v1)은 2024-12-18이다. Hub 저장소 생성일과 공개 발표일을 구분했다.
- 기반: 다른 공개 체크포인트의 fine-tuning이 아니라 base는 무작위 초기화에서 학습했다. large가 base를 이용해 초기화되는 관계와 반대다. [논문 §2.2.2](https://arxiv.org/html/2412.13663v1#S2.SS2.SSS2).
- 라이선스: Apache-2.0. [고정 모델 카드](https://huggingface.co/answerdotai/ModernBERT-base/blob/8949b909ec900327062f0ebf497f51aef5e6f0c8/README.md).
- 조사 시 [Hub API](https://huggingface.co/api/models/answerdotai/ModernBERT-base)는 다운로드 3,599,085·좋아요 1,145를 반환했다. [모델 페이지](https://huggingface.co/answerdotai/ModernBERT-base)의 다운로드 기간은 last month다. 다운로드는 채택 신호이며 성능이나 고유 사용자 수가 아니다.

기존 models·issues 전체 파일을 읽어 동일 모델·revision 및 같은 계열의 구조 분석이 없음을 확인했다. 최근 생성 모델 분석에 이어 영어 encoder와 새로운 개발 조직을 선정했다.

## 구조: 숫자와 실제 구현을 함께 읽기

아래 값은 [고정 config.json](https://huggingface.co/answerdotai/ModernBERT-base/blob/8949b909ec900327062f0ebf497f51aef5e6f0c8/config.json), [고정 모델 카드](https://huggingface.co/answerdotai/ModernBERT-base/blob/8949b909ec900327062f0ebf497f51aef5e6f0c8/README.md), [Transformers v4.48.0 구현](https://github.com/huggingface/transformers/blob/v4.48.0/src/transformers/models/modernbert/modeling_modernbert.py)을 대조했다. 구현 버전은 고정했지만 원 학습 코드와 동일한 실행 환경이라는 뜻은 아니다.

| 항목 | 확인값·의미 |
|---|---|
| 종류 | ModernBertForMaskedLM, 양방향 dense encoder |
| 총 파라미터 | 149,655,232, Hub API safetensors 메타데이터; 카드에서는 약 149M |
| 레이어 / hidden | 22 / 768 |
| attention Q / K / V heads | 각각 12, 표준 MHA; 구현의 합동 QKV projection 확인 |
| head dimension | 768 ÷ 12 = 64, 구현과 일치하는 계산 |
| MLP intermediate | 1,152; 입력 projection은 두 갈래를 위해 2,304로 확장 |
| 활성화·정규화 | GeGLU / Pre-LayerNorm |
| 입력 길이 | 최대 8,192 tokens |
| 지역 attention 설정 | local_attention=128, 구현은 좌·우 각각 64 범위로 전달 |
| 전역 attention 간격 | 3 layers; 구현상 0부터 세는 layer_id가 3의 배수일 때 |
| RoPE theta | 전역 160,000 / 지역 10,000 |
| 어휘 슬롯 / 임베딩 공유 | 50,368 / 입력·출력 가중치 공유 |
| 저장 dtype | float32; 학습 precision과 별개 |
| MoE·멀티모달 | 해당 없음; 활성 전문가 수의 구분도 적용하지 않음 |

구현 기준 전역 층 인덱스는 `0,3,6,9,12,15,18,21`로 8개, 지역 층은 `22−8=14`개다. 이는 코드의 modulo 조건에서 유도한 값이다. 지역 창의 경계·자기 토큰 포함 방식에 따라 실제 허용 위치 개수가 달라질 수 있어 설정값 128을 그대로 ‘정확히 128개 토큰만 본다’고 쓰지 않는다. [attention 구현](https://github.com/huggingface/transformers/blob/v4.48.0/src/transformers/models/modernbert/modeling_modernbert.py).

각 토큰은 앞뒤 문맥을 함께 읽는다. 지역 층은 주변 단어 관계를 다루고 주기적인 전역 층은 멀리 떨어진 부분을 연결한다. 모든 층에서 문서 전체의 관계를 계산하는 부담을 줄이는 설계다. MLP는 2,304차원 출력을 1,152차원 두 갈래로 나누고 한쪽에 GELU를 적용해 다른 쪽과 곱한 뒤 768차원으로 돌려보낸다. `intermediate_size=1152`와 논문의 GLU expansion 2304는 서로 다른 지점을 센다. [논문 §2.1](https://arxiv.org/html/2412.13663v1#S2.SS1), [MLP 구현](https://github.com/huggingface/transformers/blob/v4.48.0/src/transformers/models/modernbert/modeling_modernbert.py).

주의할 설정이 있다. config에 `position_embedding_type="absolute"`가 남아 있지만 논문은 RoPE를 명시하고 위 버전의 attention 구현도 rotary embedding을 생성한다. 따라서 이 필드 하나로 학습된 absolute 위치 임베딩 모델이라고 판정하지 않았다. Unpadding은 padding 계산을 줄이지만 문서 경계를 무시해 배치의 다른 문서까지 자유롭게 참조한다는 뜻은 아니다. [논문 §2.1.2](https://arxiv.org/html/2412.13663v1#S2.SS1.SSS2), [구현](https://github.com/huggingface/transformers/blob/v4.48.0/src/transformers/models/modernbert/modeling_modernbert.py).

## 학습: epoch 대신 공개된 토큰·배치·학습률을 구분

학습 목표는 입력의 30%를 마스킹하는 MLM이며 next-sentence prediction은 사용하지 않는다. 영어 웹·코드·과학 문헌 등이 포함된다. 아래는 **base** 수치다. [논문 §2.2 및 부록 A 표 3](https://arxiv.org/html/2412.13663v1#A1).

| 항목 | 짧은 문맥 사전학습 | 장문 확장 1단계 | 장문 확장 2단계 |
|---|---|---|---|
| 처리 토큰 | 표 3: 1.719T, 본문은 약 1.7T | 250B | 50B |
| 최대 길이 | 1,024 | 8,192 | 8,192 |
| global batch | 768→4,608으로 증가 | 72 | 72 |
| microbatch | 96 | 12 | 12 |
| learning rate | 8e-4, 3B-token warmup | 3e-4 | 3e-4에서 감쇠 |
| scheduler | warmup 뒤 일정 LR | 일정 LR | 1−sqrt decay |
| 데이터 변화 | 기본 혼합 데이터 | 장문용 혼합 | 고품질 출처 비중 확대 |
| epoch | 확인 불가 | 확인 불가 | 확인 불가 |
| 총 optimizer steps | 확인 불가 | 확인 불가 | 확인 불가 |
| gradient accumulation | 실제 실행 인자 확인 불가 | 확인 불가 | 확인 불가 |

공통 optimizer는 StableAdamW, betas=(0.90,0.98), epsilon=1e-6, weight decay=1e-5이고 하드웨어는 8×H100, DistributedDataParallel이다. Batch warmup은 50B tokens에 걸쳐 진행되며 LR warmup의 3B와 다르다. 학습 precision은 이번에 확인한 카드·논문 표로 확정할 수 없어 **확인 불가**로 남긴다. Microbatch와 GPU 수만으로 accumulation을 단정하지 않았으며, 변화하는 배치·packing·재시작 이력이 있어 총 step과 epoch도 단순 나눗셈으로 만들지 않았다. [논문 부록 A·C](https://arxiv.org/html/2412.13663v1#A1).

카드의 ‘2T tokens’는 반올림된 규모이며 부록의 단계 합 `1.719T+0.250T+0.050T=2.019T`와 표기 정밀도가 다르다. 이는 고유 데이터 크기나 반복 횟수가 아니다. 최종 base는 annealing 중 좋은 3개 체크포인트와 마지막 체크포인트를 평균했다. 이 공개 체크포인트에 별도 SFT·DPO·RL·LoRA를 수행했다는 근거는 확인 불가다. [논문 부록 A.4](https://arxiv.org/html/2412.13663v1#A1.SS4).

22 layers는 구조 깊이, optimizer step은 가중치 갱신 횟수, epoch는 데이터 전체 반복 횟수다. 8,192는 입력 토큰 길이이고 이 값들에서 epoch를 추정할 수 없다.

## 평가를 읽는 법과 활용 판단

[고정 카드 평가표](https://huggingface.co/answerdotai/ModernBERT-base/blob/8949b909ec900327062f0ebf497f51aef5e6f0c8/README.md)의 base 비교에서 GLUE는 ModernBERT 88.4, BERT 84.7이다. BEIR은 같은 ModernBERT라도 DPR 방식 41.6, ColBERT 방식 51.3으로 다르다. 이것은 저자 보고 결과이며 이 저장소에서 추론해 얻은 점수가 아니다. 검색 표현 방식·추가 학습이 다른 결과를 사전학습 체크포인트의 무조건적인 검색 성능으로 해석하지 않는다.

[논문 §3.1](https://arxiv.org/html/2412.13663v1#S3.SS1)은 GLUE의 각 작업에서 fine-tuning hyperparameter 탐색을 설명한다. MLDR_ID는 해당 데이터로 학습한 조건, MLDR_OOD는 다른 조건이며 두 열의 점수를 섞으면 안 된다. 한국어·실제 업무 문서 품질, 지연 시간과 VRAM은 별도 실측이 필요하다. 공식 평가에 포함된 CodeSearchNet·StackQA도 모든 코드베이스에 대한 품질 보장이 아니다.

[DistilBERT 분석](2026-10-18-distilbert-base-uncased.md)의 6층 단문 영어 encoder와 비교하면, ModernBERT는 22층·장문·지역/전역 혼합 구조에 초점이 있다. 층 수 증가만으로 더 빠르거나 모든 분류에서 더 좋다고 결론 낼 수 없다. [ModernBERT 카드](https://huggingface.co/answerdotai/ModernBERT-base/blob/8949b909ec900327062f0ebf497f51aef5e6f0c8/README.md)는 검색·분류·QA에 작업별 fine-tuning을 안내한다. 이 원본 MLM을 준비된 문장 임베딩 모델처럼 바로 대체하는 데에는 검증이 필요하다.

실용적인 후보는 긴 영어 문서 분류, 코드 검색용 backbone, 마스킹 예측이다. 영어·코드 중심 학습이므로 한국어 검색에서는 다국어 모델과 같은 데이터로 비교해야 한다. F32 가중치만 계산하면 `149,655,232×4=598,620,928 bytes`, 약 598.6MB(십진 단위)다. 계산상 저장 용량이며 실행 VRAM 요구량이나 속도 실측이 아니다. Activation·attention·런타임과 배치 길이가 추가 메모리를 요구한다.
