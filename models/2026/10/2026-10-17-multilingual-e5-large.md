# multilingual-e5-large: 24층 인코더와 평균 풀링으로 만드는 다국어 검색 벡터

조사일: 2026-10-17 KST. 모델 ID `intfloat/multilingual-e5-large`, revision `3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3`.

## 선정과 식별

E5 연구진의 다국어 텍스트 임베딩 모델이다. 기반 모델은 XLM-RoBERTa-large이며 라이선스는 MIT다. 정확한 최초 체크포인트 발표일은 확인 불가이고, 기술 보고서 최초 공개일 2024-02-08과 구분한다. [고정 모델 카드](https://huggingface.co/intfloat/multilingual-e5-large/blob/3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3/README.md), [보고서 기록](https://arxiv.org/abs/2402.05672)

조사 당시 최근 한 달 다운로드 6,787,644회·좋아요 1,274개를 확인했다. 좋아요 집계 기간은 확인 불가이며 다운로드는 성능이나 고유 이용자 수가 아니다. [모델 페이지](https://huggingface.co/intfloat/multilingual-e5-large), [API](https://huggingface.co/api/models/intfloat/multilingual-e5-large)

전체 기존 models·issues를 읽어 동일 모델 분석이 없음을 확인했다. 앞선 BGE-M3와 기반 계열은 관련되지만 별도 학습과 출력 규칙을 가진 원본 모델이며 양자화·미러 변형이 아니다. 가중치 다운로드·추론·학습은 수행하지 않았다.

## 구조

아래 값은 고정 [config.json](https://huggingface.co/intfloat/multilingual-e5-large/blob/3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3/config.json), [문장 길이](https://huggingface.co/intfloat/multilingual-e5-large/blob/3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3/sentence_bert_config.json), [pooling 설정](https://huggingface.co/intfloat/multilingual-e5-large/blob/3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3/1_Pooling/config.json)에 근거한다.

| 항목 | 값 |
|---|---|
| 유형 | XLMRobertaModel, 양방향 dense encoder |
| Transformer layers | 24 |
| hidden / FFN dimension | 1,024 / 4,096 |
| attention heads | 16; 별도 GQA KV heads 설정 없음 |
| head dimension | 1,024÷16=64, 균등 분할 가정 계산 |
| 위치 표현 | absolute, 514 위치 슬롯 |
| 파이프라인 입력 길이 | 최대 512 토큰 |
| 어휘 슬롯 | 250,002 |
| activation / LayerNorm epsilon | GELU / 1e-5 |
| 출력 | mean pooling 후 정규화한 1,024차원 벡터 |
| 저장 dtype | float32, 실제 학습 precision과 별개 |
| MoE·멀티모달 | 없음; MoE식 활성 파라미터 구분 해당 없음 |

API는 F32 원소 559,890,432개와 I64 원소 514개를 함께 보고한다. 따라서 합계 559,890,946개를 모두 학습 가능한 파라미터라고 부르지 않는다. 모델은 약 560M 규모이고, 정확한 trainable parameter 집계는 직접 인스턴스화하지 않아 확인 불가다. [조회 API](https://huggingface.co/api/models/intfloat/multilingual-e5-large)

514 위치 슬롯은 유효 텍스트 514토큰 지원과 다르다. 배포 파이프라인은 512토큰으로 잘라 처리한다. 패딩을 제외한 토큰 표현을 평균내고 L2 정규화한다. CLS 하나를 그대로 저장하면 이 배포 방식과 달라진다. [카드의 평균 풀링 코드](https://huggingface.co/intfloat/multilingual-e5-large/blob/3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3/README.md)

검색 시 한국어에도 질의는 `query: `, 문서는 `passage: ` 접두어를 붙인다. 예를 들어 `query: 환불 절차`와 `passage: 상품 수령 후 ...`를 각각 인코딩한다. 이 접두어는 장식이 아니라 학습 때 사용한 역할 신호다. 벡터 유사도는 관련성 점수이며 정답일 확률은 아니다. [공식 사용 규칙](https://huggingface.co/intfloat/multilingual-e5-large/blob/3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3/README.md)

## 학습

XLM-R 기반 위에 약한 감독 대조 사전학습과 지도 fine-tuning을 수행한다. [기술 보고서 부록 A](https://arxiv.org/html/2402.05672v1#A1)와 [모델 카드](https://huggingface.co/intfloat/multilingual-e5-large/blob/3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3/README.md)를 기준으로 large 모델의 값을 구분했다.

| 항목 | 대조 사전학습 | 지도 fine-tuning |
|---|---|---|
| 초기화 | XLM-RoBERTa-large | 앞 단계 모델 |
| 데이터 | 번역·제목/본문·질문/답변 등 다국어 쌍 | MS MARCO·NQ·NLI·MIRACL·Mr. TyDi 등 학습 split |
| epoch | 확인 불가 | 보고서 2 |
| optimizer step | 확인 불가 | 실제 완료 로그 확인 불가 |
| learning rate | 1e-4 | 1e-5 |
| batch | 이번 조사에서 검증하지 못함 | 512 |
| gradient accumulation | 확인 불가 | 확인 불가 |
| scheduler / optimizer | 확인 불가 | 확인 불가 |
| precision / hardware | 확인 불가 | 확인 불가 |
| DPO·RL·LoRA | 적용 근거 확인 불가 | 적용 근거 확인 불가 |

카드의 사전학습 데이터 표는 mC4 1B·NLLB 2.4B 쌍 등을 제시한다. 이 데이터 풀 크기를 실제 소비량이나 고유 학습 토큰 수로 치환하지 않는다. 위 fine-tuning은 임베딩 학습이며 생성 LLM의 instruction SFT와 같은 출력 목적이 아니다.

보고서에는 별도 large-instruct 모델도 등장한다. 그 모델의 GPT-4 합성 데이터 혼합을 이번 일반 large 체크포인트에 적용했다고 쓰지 않았다. 2 epochs는 지도 학습 단계의 반복 횟수이며 24 layers와 다른 값이다. Step은 업데이트 횟수로 batch·샘플링 정보 없이 정확하게 역산하지 않는다.

모델 카드가 연결한 [공식 평가 코드 unilm/e5](https://github.com/microsoft/unilm/tree/master/e5)는 재현 경로다. 이번에 체크포인트의 완전한 학습 실행 설정·로그를 확보한 것은 아니며 미검증 값은 확인 불가로 표시했다.

## 평가와 관련 모델 비교

보고서는 MIRACL development set 16개 언어 평균으로 large의 nDCG@10 66.5, Recall@100 94.3을 보고한다. 같은 표의 한국어는 66.5와 93.4다. 이는 보고서의 평가 수치이며 이번에 재측정하지 않았다. [보고서 표 4·6](https://arxiv.org/html/2402.05672v1)

언어 평균과 한국어 점수를 혼동하지 않으며, 검색 지표를 생성 답변 정확도로 읽지 않는다. Corpus·질의·관련성 라벨·prefix·chunking 조건을 맞춘 비교가 필요하다. 학습에 사용한 MIRACL training split과 평가 development split도 구분해야 한다.

기존 [BGE-M3 분석](2026-10-11-bge-m3.md)은 24층·1,024차원이라는 공통점이 있지만 CLS 기반 dense와 sparse·multi-vector를 함께 지원하고 최대 8,192토큰을 설정한다. 이번 E5는 mean pooling·512토큰·dense 벡터 검색이 핵심이다. 같은 기반 구조만으로 벡터를 서로 섞거나 성능을 동일시할 수 없다.

## 활용과 제약

한국어 포함 다국어 문서 검색·유사 문장·RAG 검색 단계에서 검토할 수 있다. 긴 문서는 512토큰에서 잘리므로 chunking을 설계해야 한다. 저자원 언어에서는 품질 저하가 있을 수 있으며 실제 업무 질의로 검증해야 한다. [제한 사항](https://huggingface.co/intfloat/multilingual-e5-large/blob/3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3/README.md)

F32 원소 수만 기준으로 `559,890,432×4≈2.24 GB`(십진)다. 이는 가중치 규모 계산이며 최소 VRAM·실측이 아니다. Activation·배치·검색 인덱스가 추가된다. 속도·한국어 검색 품질을 실측하지 않았다.
