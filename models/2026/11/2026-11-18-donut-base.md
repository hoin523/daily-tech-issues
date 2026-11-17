# Donut-base: OCR 엔진 없이 문서를 읽는 20개 Swin 블록과 4층 decoder

조사일: 2026-11-18 (한국 시간). 공개 자료 분석이며 가중치 다운로드·학습·추론은 수행하지 않았다.

## 선정과 식별

Hugging Face API에서 최근 한 달 다운로드 **58,037회**, 누적 좋아요 **259개**를 확인했다. 검색 재정렬에 이어 문서 이미지 이해 분야와 새로운 개발 조직을 선택했다. 사용 관심도이며 성능이나 최신 트렌딩 순위의 증거가 아니다. [API](https://huggingface.co/api/models/naver-clova-ix/donut-base), [모델 페이지](https://huggingface.co/naver-clova-ix/donut-base)

| 항목 | 확인 내용 |
|---|---|
| ID / 개발 | `naver-clova-ix/donut-base` / NAVER CLOVA 연구진 |
| HF revision | `a959cf33c20e09215873e338299c900f57047c61` |
| 공식 코드 revision | `4cfcf972560e1a0f26eb3e294c8fc88a0d336626`, 조사 시점 코드. 원 학습 commit은 확인 불가. |
| 기반 | Swin 계열 이미지 encoder + multilingual BART 계열 text decoder |
| 배포 단계 | **pre-trained only**, downstream 미세조정 필요 |
| 발표 / 조사 | 논문 최초 v1 2021-11-30 / 조사 2026-11-18. HF 가중치 자체 최초 공개일은 확인 불가. |
| 라이선스 | MIT |

근거: [고정 카드](https://huggingface.co/naver-clova-ix/donut-base/blob/a959cf33c20e09215873e338299c900f57047c61/README.md), [공식 저장소](https://raw.githubusercontent.com/clovaai/donut/4cfcf972560e1a0f26eb3e294c8fc88a0d336626/README.md), [논문 v1](https://arxiv.org/html/2111.15664v1). HF 카드는 원 개발팀 작성본이 아니라 Hugging Face 팀 작성본이다. v1 논문의 초기 실험 설정과 최종 base 배포 설정이 같다고 가정하지 않았다.

기존 `models`·`issues` 전체를 읽었으며 동일 모델·revision·미러·양자화 분석은 없었다.

## 구조: stage와 block, decoder layer 구분

| 구성 | 확인값 |
|---|---|
| 전체 파라미터 | 이 revision의 정밀한 trainable 총량은 직접 확인하지 못함. Dense, MoE 없음. |
| encoder | Donut-Swin, 4개 stage, 깊이 `[2,2,14,2]` → 총20개 block |
| stage 채널 | embed128에서 두 배씩 증가 → `[128,256,512,1024]`, 구성·코드에서 유도 |
| encoder heads / KV | stage별 Q·K·V 각각 `[4,8,16,32]`, 표준 MHA |
| encoder head dimension | 각 stage 채널÷head=32, 계산값 |
| patch / window | 4×4 픽셀 / 10×10 패치 window |
| 이미지 크기 | config `[2560,1920]` |
| encoder FFN | mlp_ratio4, stage별 `[512,1024,2048,4096]`, 계산값 |
| 위치 정보 | encoder absolute embedding 비사용, Swin relative position·shifted window |
| decoder | MBART causal decoder, 실제 `decoder_layers=4` |
| decoder hidden / heads | 1,024 / self·cross attention16개, head dimension64 |
| decoder FFN / 위치 | 4,096 / 최대1,536 위치, MBART 계열 위치 embedding |
| 어휘 / 연결 | 57,525 / encoder 특징에 cross-attention |

근거: [고정 config](https://huggingface.co/naver-clova-ix/donut-base/blob/a959cf33c20e09215873e338299c900f57047c61/config.json), [공식 SwinEncoder·BARTDecoder](https://github.com/clovaai/donut/blob/4cfcf972560e1a0f26eb3e294c8fc88a0d336626/donut/model.py). decoder 설정의 `encoder_layers=12`, `num_hidden_layers=12` 흔적을 그대로 decoder 깊이라고 읽지 않았다. 생성 코드가 사용하는 값은 decoder_layers다.

encoder는 작은 패치에서 시작해 window 안의 정보를 섞고 patch merging으로 해상도를 줄인다. decoder는 이미지 특징과 이전 출력 토큰을 참고해 다음 토큰을 생성한다. 이미지 전체를 먼저 OCR 텍스트와 좌표로 변환해 외부 모델에 넘기는 별도 OCR 엔진이 필요 없다는 의미이지, 문자 인식 능력을 학습하지 않는다는 뜻은 아니다. [공식 모델 구현](https://github.com/clovaai/donut/blob/4cfcf972560e1a0f26eb3e294c8fc88a0d336626/donut/model.py)

크기를 config대로 맞추고 4배 patchification·세 번의2배 merging을 적용하면 최종 공간 격자는 `2560÷32 × 1920÷32 = 80×60`, 즉4,800개 위치라는 계산이 된다. 실제 전처리 resize·padding 조건을 따져야 하며 이 숫자를 문서 글자 수로 해석하지 않는다. [구성](https://huggingface.co/naver-clova-ix/donut-base/blob/a959cf33c20e09215873e338299c900f57047c61/config.json), [전처리 구현](https://github.com/clovaai/donut/blob/4cfcf972560e1a0f26eb3e294c8fc88a0d336626/donut/model.py)

## 학습 단계와 공개 범위

| 항목 | base 사전학습에 대해 확인한 내용 |
|---|---|
| 데이터 | 공식 저장소: IIT-CDIP 약11M + SynthDoG 영어·중국어·일본어·한국어 각0.5M |
| 목적 | 문서 이미지에서 텍스트를 생성하는 사전학습, downstream 구조화 작업과 구분 |
| epoch / 총 step | 이 HF revision과 대응하는 실행값 확인 불가 |
| batch / accumulation | 실행 대응값 확인 불가 |
| optimizer / LR / scheduler | 실행 대응값 확인 불가 |
| precision / 하드웨어 | 해당 HF revision의 실행 조건 확인 불가 |
| downstream 지도 학습 | 이 공개본에는 포함되지 않음. 과제별 별도 미세조정 필요. |
| DPO·RL·LoRA | 이 base 배포본의 단계로 보고되지 않음 |

근거: [공식 base 설명·데이터 목록](https://raw.githubusercontent.com/clovaai/donut/4cfcf972560e1a0f26eb3e294c8fc88a0d336626/README.md), [고정 카드](https://huggingface.co/naver-clova-ix/donut-base/blob/a959cf33c20e09215873e338299c900f57047c61/README.md). 데이터 개수를 더해13M이라고 계산할 수 있지만 유일한 이미지 수·반복 학습량을 독립 검증한 값은 아니다.

공식 코드의 기본 초기화는 `swin_base_patch4_window12_384`와 `hyunwoongko/asian-bart-ecjk`를 참조한다. 이는 코드의 초기화 경로이며 최종 HF 가중치가 어떤 upstream revision을 사용했는지까지 확인한 것은 아니다. 기반 사전학습의 epoch·step도 확인 불가로 남겼다. [모델 초기화 코드](https://github.com/clovaai/donut/blob/4cfcf972560e1a0f26eb3e294c8fc88a0d336626/donut/model.py)

공식 config 폴더에는 CORD·DocVQA·RVL-CDIP 등의 downstream 설정이 있다. 그 과제별 epoch·batch를 pre-trained-only base의 사전학습 조건으로 옮기지 않았다. [공식 config 폴더](https://github.com/clovaai/donut/tree/4cfcf972560e1a0f26eb3e294c8fc88a0d336626/config)

## 평가와 활용 범위

논문은 문서 분류·파싱·DocVQA를 별도로 평가하며, 공식 저장소도 CORD 등의 **finetuned 모델** 점수를 구분한다. 해당 수치를 오늘 base 체크포인트를 그대로 사용한 점수로 기록하지 않았다. JSON 출력 형식이나 task prompt도 과제별 학습과 연결되어야 한다. [논문 실험](https://arxiv.org/html/2111.15664v1), [공식 모델 표](https://raw.githubusercontent.com/clovaai/donut/4cfcf972560e1a0f26eb3e294c8fc88a0d336626/README.md)

문서 파싱·분류·질의응답용 미세조정의 출발점으로 검토할 수 있다는 것이 이번 분석의 해석이다. 사전학습본만으로 영수증 금액과 항목을 신뢰성 있게 추출한다고 보장할 수 없다. 작은 글자·길이가 긴 문서·훈련과 다른 레이아웃에서는 결과를 확인해야 하며, 문법적으로 맞는 생성 텍스트가 원본 사실과 같다는 보장도 없다.

[Swin-Tiny](../10/2026-10-30-swin-tiny-patch4-window7-224.md)는 이미지 분류 구조를 다뤘지만 Donut은 더 큰 문서 격자와 생성 decoder를 결합한다. [BLIP 이미지 captioning](../10/2026-10-15-blip-image-captioning-base.md)도 image-to-text이지만 학습 목적·출력 schema가 다르다. 입력과 출력 모양만으로 모델을 서로 대체할 수는 없다.

출처·구성·중복을 검토했다. 실제 문서 정확도·VRAM·속도는 측정하지 않았으며 구조 계산과 공개 사실을 구분했다.
