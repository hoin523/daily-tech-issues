# BART-large-CNN: 12층 encoder와 12층 decoder로 영어 기사를 요약하기

조사일: 2026-11-19 (한국 시간). 공개 자료 분석이며 가중치 다운로드·추론·학습은 수행하지 않았다.

## 선정과 식별

Hugging Face API에서 최근 한 달 다운로드 **1,052,661회**, 누적 좋아요 **1,653개**를 확인했다. 문서 이미지 이해에 이어 텍스트 요약 분야를 선택했다. 사용 관심도이며 품질이나 트렌딩 순위의 증거는 아니다. [API](https://huggingface.co/api/models/facebook/bart-large-cnn), [모델 페이지](https://huggingface.co/facebook/bart-large-cnn)

| 항목 | 확인 내용 |
|---|---|
| 모델 / 개발 | `facebook/bart-large-cnn` / Facebook AI BART 연구진 |
| revision | `37f520fa929c961707657b28798b30c003dd100b` |
| 기반 / 단계 | 영어 BART-large 사전학습 후 CNN/DailyMail 지도 요약 미세조정 |
| 발표 / 조사 | 논문 v1 2019-10-29 / 조사 2026-11-19. HF 변환 가중치 최초 배포일은 확인 불가. |
| 라이선스 / 언어 | MIT / 영어 |

근거: [고정 카드](https://huggingface.co/facebook/bart-large-cnn/blob/37f520fa929c961707657b28798b30c003dd100b/README.md), [원 논문](https://arxiv.org/html/1910.13461v1). HF 카드는 원 연구팀 직접 작성본이 아닌 Hugging Face 팀 작성본이다. 기존 `models`·`issues` 전체를 읽었으며 동일 모델·revision·미러·양자화 분석은 없었다.

## 구조와 생성 설정

| 항목 | 확인값 |
|---|---|
| 저장 텐서 | Hub F32 원소 406,290,432개. trainable 계수는 가중치 로딩으로 직접 재집계하지 않음. |
| 유형 | 양방향 encoder + causal decoder의 seq2seq, Dense |
| encoder / decoder layer | 각각12, 합계24개 Transformer layer |
| hidden | 1,024 |
| attention / KV heads | encoder·decoder 각각16개 표준 MHA, KV도16개 |
| head dimension | `1024÷16=64`, 계산값 |
| FFN | 양쪽4,096, GELU |
| 입력·위치 | 최대1,024 위치, 학습 가능한 absolute positional embedding |
| 어휘 | 50,264 |
| dropout | hidden0.1, attention·activation0.0, LayerDrop0.0 |
| decoder 연결 | 이전 출력에 self-attention, encoder 출력에 cross-attention |
| MoE·이미지 encoder | 없음 |
| 요약 생성 기본값 | beam4, min_length56·max_length142, length_penalty2.0, no-repeat trigram3 |

근거: [고정 config](https://huggingface.co/facebook/bart-large-cnn/blob/37f520fa929c961707657b28798b30c003dd100b/config.json), [공식 BART 라이브러리 구현](https://github.com/huggingface/transformers/blob/v4.7.0/src/transformers/models/bart/modeling_bart.py), [API](https://huggingface.co/api/models/facebook/bart-large-cnn). 라이브러리 태그는 구조 확인용이며 원 학습 commit이 아니다.

encoder는 기사 전체를 앞뒤로 읽고, decoder는 원문 표현과 이미 쓴 요약을 참고해 다음 토큰을 예측한다. 원문 문장만 고르는 추출식 요약과 달리 새 문장을 생성할 수 있다. 자연스러운 문장이 반드시 원문에 충실하다는 뜻은 아니다. [논문 구조·요약 설명](https://arxiv.org/html/1910.13461v1)

config에 `num_hidden_layers=12`도 있지만 encoder12와 decoder12를 합쳐 모델 전체가12층이라고 설명하지 않았다. `max_length=142`는 생성 기본 한도이며 네트워크의 최대 position1,024와 구분한다. beam4도 학습 epoch가 아니다. `_num_labels=3` 흔적이 있다고 이 체크포인트를3개 범주 분류 모델로 읽지 않는다. [config](https://huggingface.co/facebook/bart-large-cnn/blob/37f520fa929c961707657b28798b30c003dd100b/config.json)

## 사전학습과 요약 미세조정

| 항목 | BART-large 사전학습 | 공식 CNN/DM 미세조정 예제 |
|---|---|---|
| 데이터·목적 | 약160GB 영어 문서, 손상된 원문 복원 | CNN/DailyMail 기사→요약 |
| step | 논문500,000 | 예제 total updates20,000 |
| epoch | 이번 조사에서 명시값 확인 불가 | 예제 epoch 값 미공개 |
| batch | 논문8,000 | 장치별 max_tokens2,048, accumulation4, GPU8개 |
| optimizer | 해당 절에서 상세 확인 불가 | Adam, β1=0.9·β2=0.999, eps1e-8 |
| LR / scheduler | 해당 실행 상세 확인 불가 | 3e-5, warm-up500updates, polynomial decay |
| loss·규제 | text infilling·문장 순서 섞기 | label-smoothed cross entropy0.1, weight decay0.01 |
| precision / 장비 | 해당 실행 조건 확인 불가 | FP16, 예제 V10032GB 8개 |

근거: [논문 §5.1](https://arxiv.org/html/1910.13461v1), [고정 공식 fairseq 학습 안내](https://github.com/facebookresearch/fairseq/blob/3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99/examples/bart/README.summarization.md). 오른쪽은 공개 재현 recipe이며 개별 HF revision의 실행 로그까지 확인한 값은 아니다. SFT 성격의 지도 요약 학습이지 DPO·RL·LoRA가 적용됐다고 보고된 모델은 아니다.

문서 일부를 mask로 바꾸고 문장 순서를 섞은 뒤 복원하는 사전학습과 기사에서 짧은 요약을 만드는 미세조정은 목적·출력이 다르다. 사전학습의500,000step을 요약 학습step이라고 쓰지 않았다. [원 논문](https://arxiv.org/html/1910.13461v1)

공식 예제의 `max_tokens`는 장치별 token budget이며 고정된 기사 개수가 아니다. `8×2048×4=65,536`은 설정상 token budget 곱셈일 뿐 매 update의 정확한 유효 입력량이 아니다. 실제 packing·padding·길이 조건 없이 이를 고정 batch 문서 수나 epoch로 바꾸지 않는다. [학습 명령](https://github.com/facebookresearch/fairseq/blob/3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99/examples/bart/README.summarization.md)

## 평가를 읽을 때 주의할 점

원 논문 CNN/DailyMail 표의 ROUGE-1/2/L은 **44.16/21.28/40.90**이다. 논문 조건의 요약 결과이며 해당 HF revision으로 재현한 수치는 아니다. [논문 Table 3](https://arxiv.org/html/1910.13461v1)

HF 카드 메타데이터에는 별도 ROUGE 결과가 있으나 dataset split이 **train**으로 표시되어 있다. 이를 검증 없이 test 성능이라고 옮기지 않았다. ROUGE-L과 ROUGE-LSUM도 구분해야 한다. 논문 생성 조건과 현재 config의 beam 수 역시 일치한다고 가정하지 않았다. [고정 카드](https://huggingface.co/facebook/bart-large-cnn/blob/37f520fa929c961707657b28798b30c003dd100b/README.md), [config](https://huggingface.co/facebook/bart-large-cnn/blob/37f520fa929c961707657b28798b30c003dd100b/config.json)

영어 뉴스 기사 요약의 기준 모델이나 요약 학습 실험에 활용할 수 있다는 것이 이번 분석의 해석이다. 1,024 위치를 넘는 원문은 전처리 전략이 필요하며 잘린 정보가 요약에서 사라질 수 있다. ROUGE는 표현 겹침을 보는 지표여서 사실 오류·수치 왜곡을 모두 검출하지 못한다. 출력의 인명·날짜·숫자는 원문과 대조해야 한다.

[ByT5-small](2026-11-12-byt5-small.md)은 byte 기반 사전학습본이고, 오늘 모델은 subword 기반 영어 요약 미세조정본이다. [Donut-base](2026-11-18-donut-base.md)는 이미지 encoder에 BART 계열 decoder를 붙인 별도 모델이므로 이 텍스트 요약기의 입출력과 혼용하지 않는다.

출처·구성·중복 여부를 검토했다. 속도·VRAM·새 문서 요약 품질은 실측하지 않았으며 공식 보고값과 구조 해석을 구분했다.
