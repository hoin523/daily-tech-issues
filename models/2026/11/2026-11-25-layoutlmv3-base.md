# LayoutLMv3-base: 텍스트·좌표·이미지 패치를 함께 읽는 12층 문서 encoder

조사일: 2026-11-25 (한국 시간). 공개 자료 분석이며 가중치 다운로드·추론·학습은 수행하지 않았다.

## 선정과 식별

Hub API의 최근 한 달 다운로드 **617,500회**, 누적 좋아요 **526개**를 확인했다. 사용 관심도이며 정확도나 사용자 수의 증거는 아니다. 이미지 생성에서 Microsoft의 문서 멀티모달 모델로 분야·조직을 바꿨다. 기존 models·issues 54개 기록을 모두 읽고 동일 모델·revision·미러·양자화 분석이 없음을 확인했다. [API](https://huggingface.co/api/models/microsoft/layoutlmv3-base)

| 항목 | 내용 |
|---|---|
| ID / 개발 | `microsoft/layoutlmv3-base` / Microsoft Research·공동 연구진 |
| revision | `cfbbbff0762e6aab37086fdd4739ad14fe7d5db4` |
| 기반 | RoBERTa로 초기화한 문서 멀티모달 사전학습 encoder |
| 발표 / 조사 | 관련 논문 2022년, 확인한 v3 2022-07-19 / 조사 2026-11-25. 이 HF revision 최초 공개일 확인 불가 |
| 용도·언어 | 문서 이해 후속 학습, 영어 |
| 라이선스 | CC BY-NC-SA 4.0. 비상업·동일조건변경허락 조건을 포함 |

[고정 카드](https://huggingface.co/microsoft/layoutlmv3-base/blob/cfbbbff0762e6aab37086fdd4739ad14fe7d5db4/README.md), [논문](https://arxiv.org/html/2204.08387v3). 기본 사전학습본이며 특정 양식의 필드를 바로 추출하는 완성된 task head 모델로 취급하지 않는다.

## 구조

| 항목 | 확인값 |
|---|---|
| 총 파라미터 | 논문 Base 약 133M, 직접 trainable 계수 재집계 안 함 |
| Transformer | 양방향 Dense encoder 12층, hidden 768 |
| attention / KV heads | 표준 MHA 12개, KV도 12개 |
| head dimension | `768÷12=64`: 계산값 |
| FFN | 3,072, GELU |
| 텍스트 어휘·위치 | vocab 50,265, max_position_embeddings 514 |
| 이미지 | RGB 224×224, patch 16 (라이브러리 기본값·논문 확인) |
| 이미지 토큰 | `(224÷16)²=196`개 패치: 계산값. visual CLS를 포함하면 197개 |
| 좌표 임베딩 | max_2d_position_embeddings 1,024, coordinate_size·shape_size 각각 128 |
| attention bias | 1D relative·2D spatial 모두 활성, bins 각각 32·64 |
| dropout / LayerNorm | hidden·attention 0.1 / epsilon 1e-5 |
| decoder·MoE | 없음 |

[고정 config](https://huggingface.co/microsoft/layoutlmv3-base/blob/cfbbbff0762e6aab37086fdd4739ad14fe7d5db4/config.json), [논문 구조](https://arxiv.org/html/2204.08387v3), [공식 라이브러리 구성 코드](https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/models/layoutlmv3/configuration_layoutlmv3.py). 텍스트 position 테이블의 514를 514개 유효 단어 한도라고 해석하지 않는다. 논문 텍스트 sequence 길이는 512이고 이미지 토큰은 별도로 결합된다.

텍스트만 읽으면 “합계”라는 단어는 알 수 있어도 영수증에서 어디에 적혔는지 놓치기 쉽다. 이 모델은 OCR 단어와 상자의 위치, 이미지 패치까지 함께 attention에 넣어 관계를 학습한다. 이미지 패치를 사용한다는 이유로 OCR이 필요 없는 모델이라고 설명하면 안 된다. 고정 전처리 파일은 `apply_ocr=true`다. [카드](https://huggingface.co/microsoft/layoutlmv3-base/blob/cfbbbff0762e6aab37086fdd4739ad14fe7d5db4/README.md), [전처리](https://huggingface.co/microsoft/layoutlmv3-base/blob/cfbbbff0762e6aab37086fdd4739ad14fe7d5db4/preprocessor_config.json)

## 사전학습과 후속 학습

| 항목 | 논문 사전학습 보고값 |
|---|---|
| 데이터 | IIT-CDIP 문서 이미지 약 1,100만 개 |
| 목적 | MLM·MIM·word-patch alignment |
| optimizer update | 500,000 step |
| epoch | 명시값 미공개. step과 데이터 개수만으로 단정하지 않음 |
| batch | 2,048 |
| gradient accumulation | 사용한다고 명시, 정확한 누적 횟수 확인 불가 |
| optimizer | Adam, β=(0.9,0.98), weight decay 0.01 |
| Base LR | 1e-4, 처음 4.8% step 선형 warm-up |
| warm-up 계산 | `500000×0.048=24000` step: 계산값 |
| precision·장비 | mixed precision·분산 학습. 정확한 precision 구성·GPU 종류·수 확인 불가 |
| SFT·DPO·RL·LoRA | 이 기본 revision에 해당 언어 모델 후속 단계 적용은 미공개 |

[논문 §3.1–3.2](https://arxiv.org/html/2204.08387v3). 논문 보고값이며 HF revision의 실행 로그를 직접 확보한 것은 아니다. warm-up 이후 전체 LR 곡선은 이번 조사에서 확인 불가다.

텍스트를 가려 복원하고, 이미지 패치의 이산 토큰을 맞히며, 단어에 대응하는 패치가 가려졌는지도 예측한다. 이미지 tokenizer는 사전학습 label 생성에 쓰이며 문서 답변을 생성하는 decoder와 다르다. [논문 학습 목표](https://arxiv.org/html/2204.08387v3)

논문의 FUNSD 후속 학습 예제는 1,000 step·batch 16·LR 1e-5다. 이는 사전학습 500,000 step과 별개다. task별 head와 라벨 데이터가 추가되므로 기본 encoder에 그 결과를 그대로 부여하지 않는다. [논문 §3.3](https://arxiv.org/html/2204.08387v3)

## 평가와 사용 시 제약

논문 Base의 FUNSD F1은 **90.29**, DocVQA ANLS는 **78.76**이다. 각각 task별 미세조정 결과이며 서로 같은 지표가 아니다. FUNSD는 train 149·test 50문서, segment-level 위치 사용도 비교 조건이다. 이 결과를 오늘 HF 기본 encoder의 무학습 추출 정확도라고 쓰지 않았다. [논문 Table 1·§3.3](https://arxiv.org/html/2204.08387v3)

전처리는 resize 224, RGB mean/std 각각 0.5, OCR 활성으로 저장돼 있다. 실제 OCR 단어·좌표와 tokenizer 정렬이 어긋나면 이미지가 정상이어도 잘못된 입력이 된다. 2D 임베딩 테이블 1,024와 이미지 픽셀 좌표계를 혼용하지 말고 processor가 요구하는 정규화 좌표를 확인해야 한다. [전처리](https://huggingface.co/microsoft/layoutlmv3-base/blob/cfbbbff0762e6aab37086fdd4739ad14fe7d5db4/preprocessor_config.json), [config](https://huggingface.co/microsoft/layoutlmv3-base/blob/cfbbbff0762e6aab37086fdd4739ad14fe7d5db4/config.json)

기존 [Donut-base](2026-11-18-donut-base.md)는 OCR 없는 이미지 encoder·텍스트 decoder 방식이다. 오늘 모델은 OCR·위치·이미지의 표현을 합치는 encoder로, 입력 준비와 후속 task head가 다르다. 분석상 양식 분류·정보 추출·문서 QA 학습의 출발점으로 활용할 수 있다. 한국어·새 서식·저화질 OCR에서는 별도 평가가 필요하다. 속도·VRAM·사용자 문서 정확도는 실측하지 않았다.
