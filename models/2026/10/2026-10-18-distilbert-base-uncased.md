# DistilBERT base uncased: BERT 교사를 압축한 6층 영어 encoder

조사일: 2026-10-18 KST. ID `distilbert/distilbert-base-uncased`, revision `12040accade4e8a0f71eabdb258fecc2e7e948be`.

## 선정과 식별

Hugging Face 연구진이 BERT-base-uncased를 교사로 증류한 영어 모델이다. 라이선스는 Apache-2.0이다. 원 논문은 2019년 공개됐으며 정확한 최초 배포 체크포인트 일자는 이번 조사에서 확인 불가다. [고정 카드](https://huggingface.co/distilbert/distilbert-base-uncased/blob/12040accade4e8a0f71eabdb258fecc2e7e948be/README.md), [원 논문](https://arxiv.org/abs/1910.01108)

조사 당시 최근 한 달 다운로드 7,301,484회·좋아요 1,489개였다. 좋아요 집계 기간은 확인 불가이며 인기와 성능은 구분한다. [모델 페이지](https://huggingface.co/distilbert/distilbert-base-uncased), [API](https://huggingface.co/api/models/distilbert/distilbert-base-uncased)

모든 기존 models·issues 기록을 읽어 동일 모델 분석이 없음을 확인했다. MiniLM과 작은 encoder라는 공통점이 있지만 다른 증류 학습과 출력 목적을 가진 모델이다. 가중치 다운로드·유료 연산·추론·학습은 수행하지 않았다.

## 구조

표는 [고정 config.json](https://huggingface.co/distilbert/distilbert-base-uncased/blob/12040accade4e8a0f71eabdb258fecc2e7e948be/config.json)과 API의 safetensors 집계에 근거한다.

| 항목 | 값 |
|---|---|
| 배포 architecture | DistilBertForMaskedLM |
| 파라미터 집계 | 66,985,530 F32 원소, 약 67M; MLM head 포함 배포 집계 |
| Transformer layers | 6 |
| hidden / FFN dimension | 768 / 3,072 |
| attention heads | 12, 별도 GQA KV heads 설정 없음 |
| head dimension | 768÷12=64, 균등 분할 가정 계산 |
| 최대 위치 슬롯 | 512 |
| 위치 표현 | 학습 가능한 absolute, sinusoidal_pos_embds=false |
| activation | GELU |
| vocab | 30,522 |
| dropout / attention dropout | 0.1 / 0.1 |
| MoE·활성 전문가 | 없음; MoE식 활성 파라미터 구분 해당 없음 |
| 멀티모달 | 없음 |

BERT의 token-type embedding과 pooler를 제거하고 깊이를 절반으로 줄였다. 학생 초기화는 교사의 한 층씩 건너뛴 가중치를 활용한다. [원 논문 §3](https://arxiv.org/html/1910.01108v4#S3)

문장의 좌우 문맥을 함께 읽어 `[MASK]` 위치의 단어 점수를 예측한다. 다음 단어를 순서대로 생성하는 대화 LLM과 목적이 다르다. Lowercasing을 사용하므로 대소문자 정보가 사라진다. 512는 특수 토큰을 포함한 입력 위치 예산이며 단어 512개를 보장하지 않는다. [카드의 사용·전처리 설명](https://huggingface.co/distilbert/distilbert-base-uncased/blob/12040accade4e8a0f71eabdb258fecc2e7e948be/README.md)

## 학습과 증류

교사의 단어 확률 분포를 따르는 distillation loss, 정답 마스크 토큰을 맞히는 MLM loss, hidden state 방향을 맞추는 cosine loss를 결합한다. 교사가 높은 확률의 정답 외에 주는 대안 단어 정보도 학습 신호가 된다. [논문 §2–3](https://arxiv.org/html/1910.01108v4)

| 항목 | 사전학습·증류 확인 상태 |
|---|---|
| 데이터 | English Wikipedia + BookCorpus |
| 데이터 구성 | 카드의 BookCorpus는 11,038권; 실제 소비 토큰 수 확인 불가 |
| epoch | 확인 불가 |
| optimizer steps | 확인 불가 |
| batch | 논문 최대 4K examples, gradient accumulation 활용 |
| accumulation 횟수 | 확인 불가 |
| learning rate / scheduler / optimizer | 해당 체크포인트 실제 실행값 확인 불가 |
| 학습 precision | 확인 불가 |
| hardware / 시간 | 8 V100 16GB, 약 90시간 |
| 별도 SFT·DPO·RL·LoRA | 이 기본 체크포인트 적용 근거 확인 불가 |

근거는 [논문 학습 설명](https://arxiv.org/html/1910.01108v4#S3)과 [카드](https://huggingface.co/distilbert/distilbert-base-uncased/blob/12040accade4e8a0f71eabdb258fecc2e7e948be/README.md)다. 공식 [v4.10.0 증류 설정 파일](https://github.com/huggingface/transformers/blob/v4.10.0/examples/research_projects/distillation/training_configs/distilbert-base-uncased.json)도 확인했지만 구조 설정 파일을 실제 학습 실행 로그로 간주하지 않았다. 실행 인자·데이터 순회 기록 없이 epoch를 역산하지 않는다. 6 layers와 90시간은 각각 깊이와 시간이며 epoch가 아니다.

카드의 intended uses에는 next-sentence prediction 문구가 있지만 논문은 NSP 목표 없이 증류했다고 명시한다. 이 배포 architecture도 MaskedLM이다. 따라서 기본 모델에 NSP head가 준비됐다고 안내하지 않는다. 문서의 BERT식 전처리 설명과 DistilBERT 고유 학습 목표를 구분해야 한다.

## 평가와 비교

논문 GLUE development 평가에서 BERT-base macro score 79.5, DistilBERT 77.0을 보고한다. 각 과제 fine-tuning 후 서로 다른 seed 5회 결과의 중앙값이며, 기본 MLM 모델을 그대로 분류기에 넣어 얻은 점수가 아니다. 현재 모델 순위나 모든 과제에서 성능 97% 유지라는 보장으로 해석하지 않는다. [논문 표 1](https://arxiv.org/html/1910.01108v4#S4)

속도 주장도 당시 CPU·batch 조건의 측정이다. 여기서는 직접 측정하지 않았으므로 현대 GPU·모바일에서도 동일 향상률을 보장하지 않는다. 평가 시 같은 head·fine-tuning 데이터·길이·장치를 맞춰야 한다.

기존 [MiniLM 분석](2026-10-06-all-minilm-l6-v2.md)은 6층·384차원과 평균 풀링·정규화로 문장 벡터를 만든다. DistilBERT는 6층·768차원의 token 표현과 MLM head가 중심이다. 층수가 같아도 검색용 의미 벡터 품질이 같지 않으며 그대로 상호 교체할 수 없다.

## 활용과 제약

영어 masked-token 실험이나 분류·개체명 인식·추출형 QA용 fine-tuning 출발점으로 검토할 수 있다. 자동으로 고품질 문장 검색 벡터를 주는 모델은 아니며 한국어 중심 업무에는 별도 다국어 모델을 평가해야 한다. 교사·학습 데이터의 편향도 이어질 수 있다. [카드 제한 사항](https://huggingface.co/distilbert/distilbert-base-uncased/blob/12040accade4e8a0f71eabdb258fecc2e7e948be/README.md)

F32 가중치만 가정하면 `66,985,530×4≈267.9 MB`(십진)다. 이는 계산값이며 최소 VRAM·실측이 아니다. Activation·batch·실행 버퍼·추가 downstream head는 별도다.
