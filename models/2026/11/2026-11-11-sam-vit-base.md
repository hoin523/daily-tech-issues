# SAM ViT-B: 이미지 12층과 양방향 mask decoder 2층으로 만드는 분할 마스크

조사일: 2026-11-11 (한국 시간). 공개 구성·논문·코드 분석이며 가중치 다운로드·학습·추론은 수행하지 않았다.

## 선정과 식별

Hugging Face API에서 최근 한 달 다운로드 **499,523회**, 누적 좋아요 **179개**를 확인했다. 텍스트 생성·검색에 이어 이미지 분할을 선택했다. 이 수치는 사용 관심도이며 정확도나 최신 트렌딩 순위의 증거가 아니다. [API](https://huggingface.co/api/models/facebook/sam-vit-base), [모델 페이지](https://huggingface.co/facebook/sam-vit-base)

| 항목 | 확인 내용 |
|---|---|
| ID / 개발 | `facebook/sam-vit-base` / Meta AI Research, FAIR |
| HF revision | `70c1a07f894ebb5b307fd9eaaee97b9dfc16068f` |
| 공식 코드 revision | `dca509fe793f601edb92606367a655c15ac00fdf`, 조사 시점 구현. 원 학습 commit은 확인 불가. |
| 용도 | 점·상자·이전 mask 등의 공간적 prompt를 받아 이미지 마스크 예측 |
| 기반 | MAE 사전학습 ViT 계열 이미지 encoder를 사용하는 SAM의 ViT-B 변형. MAE 원 체크포인트 revision은 확인 불가. |
| 발표 / 조사 | 원 논문 v1 2023-04-05 / 조사 2026-11-11. 이 HF 변환 가중치의 최초 배포일은 확인 불가. |
| 라이선스 | Apache-2.0 |

근거: [고정 카드](https://huggingface.co/facebook/sam-vit-base/blob/70c1a07f894ebb5b307fd9eaaee97b9dfc16068f/README.md), [공식 생성 코드](https://github.com/facebookresearch/segment-anything/blob/dca509fe793f601edb92606367a655c15ac00fdf/segment_anything/build_sam.py), [논문](https://arxiv.org/html/2304.02643v1). HF 카드는 개발팀 직접 작성본이 아니라 Hugging Face 팀이 작성한 설명이다.

기존 `models`·`issues` 전체 기록을 읽었으며 같은 SAM 체크포인트·미러·양자화 변형의 분석은 없었다. 다른 시각 encoder와 기반 개념이 겹쳐도 prompt encoder·mask decoder가 있는 별도 작업 모델이다.

## 구성 요소별 구조

| 구성 | 값·역할 |
|---|---|
| 저장 텐서 규모 | API F32 원소 93,735,728개. buffer를 제외한 trainable 수는 직접 집계하지 않았다. |
| image encoder | ViT-B, 12층, hidden 768, Q·K·V 각각 12 heads |
| image head dimension | `768÷12=64`, 계산값 |
| 입력·패치 | 1024×1024 입력 격자, 16×16 패치 → 64×64 특징 격자 |
| attention 공간 | window 14×14 패치, 0부터 센 `[2,5,8,11]` layer는 global |
| image 위치 표현 | absolute position embedding + relative position attention |
| encoder MLP | 공식 생성 코드 ratio 4 → 3,072 |
| neck | image 특징을 256채널로 변환 |
| prompt encoder | hidden 256, 점·상자 embedding 및 mask 입력 처리 |
| mask decoder | 2개 two-way Transformer block + 마지막 token-to-image attention |
| decoder hidden / FFN | 256 / 2,048, 8 heads |
| decoder head dimension | self-attention `256÷8=32`, downsample=2인 cross-attention `(256÷2)÷8=16` |
| mask 후보 | `num_multimask_outputs=3`, 모호한 prompt에 여러 후보와 품질 점수 |
| IoU 예측 head | 3층 MLP, hidden 256 |
| MoE·VAE·언어 생성 | 없음. prompt는 이 공개 모델에서 텍스트 지시를 의미하지 않는다. |

근거: [고정 config](https://huggingface.co/facebook/sam-vit-base/blob/70c1a07f894ebb5b307fd9eaaee97b9dfc16068f/config.json), [공식 ViT-B 생성](https://github.com/facebookresearch/segment-anything/blob/dca509fe793f601edb92606367a655c15ac00fdf/segment_anything/build_sam.py), [two-way attention](https://github.com/facebookresearch/segment-anything/blob/dca509fe793f601edb92606367a655c15ac00fdf/segment_anything/modeling/transformer.py), [mask decoder](https://github.com/facebookresearch/segment-anything/blob/dca509fe793f601edb92606367a655c15ac00fdf/segment_anything/modeling/mask_decoder.py).

HF config의 `vision_config`에는 `intermediate_size=6144`와 `mlp_dim=3072`, `mlp_ratio=4`가 함께 있다. 키 하나만 읽어 FFN을 6,144로 단정하지 않았다. 위 표는 공식 생성 코드의 ratio 4와 [image encoder의 MLP 생성](https://github.com/facebookresearch/segment-anything/blob/dca509fe793f601edb92606367a655c15ac00fdf/segment_anything/modeling/image_encoder.py)을 기준으로 3,072를 설명한다. 실제 HF 로딩 버전별 필드 처리까지 실행 검증한 것은 아니다.

이미지 encoder가 전체 이미지를 한 번 읽어 특징 지도를 만든다. prompt encoder는 클릭한 위치가 전경인지 배경인지, 또는 상자가 어디인지를 벡터로 표현한다. 작은 decoder는 prompt가 이미지를 읽는 attention과 이미지가 prompt를 읽는 attention을 교대로 수행한다. 마스크별 벡터와 확대된 이미지 특징을 결합해 영역을 출력한다. 같은 이미지에서 클릭을 바꿀 때 이미지 특징을 재사용할 수 있는 구조다. [공식 decoder](https://github.com/facebookresearch/segment-anything/blob/dca509fe793f601edb92606367a655c15ac00fdf/segment_anything/modeling/mask_decoder.py), [two-way Transformer](https://github.com/facebookresearch/segment-anything/blob/dca509fe793f601edb92606367a655c15ac00fdf/segment_anything/modeling/transformer.py)

64×64는 4,096개 패치 위치라는 계산이다. window=14는 패치 단위이므로 14픽셀 창이 아니다. 12층·2층은 네트워크 깊이이며 학습 epoch·사용자 클릭 횟수와 다르다.

## 학습 설정: ViT-B 변형을 별도로 읽기

| 항목 | 공개 근거·범위 |
|---|---|
| 기반 사전학습 | MAE 기반 ViT 계열. 원 사전학습의 epoch·step·batch·장비는 이번 조사에서 확인 불가. |
| 분할 학습 데이터 | SA-1B, 약 11M 이미지·1.1B mask |
| ViT-B iteration / batch | 180,000 / 이미지 128개 |
| ViT-B 분산 장치 | GPU 128개. 이 문장의 GPU 종류는 확인 불가. |
| optimizer | 논문 기본 recipe AdamW, β1=0.9·β2=0.999 |
| ViT-B LR / 규제 | 8e-4, layer-wise LR decay 0.6, weight decay 0.1, drop path 0.6 |
| epoch | ViT-B 전용 명시값 확인 불가. `180,000×128=23,040,000` 이미지 입력은 계산값이며 정확한 고유 이미지 반복 횟수가 아니다. |
| scheduler | 기본 recipe는 warm-up 250 iteration·step decay. ViT-B의 정확한 decay 시점은 별도 확인 불가. |
| accumulation / precision | 해당 실행 설정 확인 불가 |
| SFT·DPO·RL·LoRA | 이 체크포인트의 후속 단계로 보고되지 않음 |

근거: [논문 부록 A, Training recipe](https://arxiv.org/html/2304.02643v1). ViT-H 기본 recipe의 90k iteration·batch 256을 ViT-B에 적용하지 않았다. 학습 중 prompt 수정 반복과 optimizer update도 구분해야 한다.

## 평가와 실용적인 해석

논문의 zero-shot 분할 실험은 다양한 데이터셋·prompt 조건을 사용한다. 클릭 위치, 다중 후보 중 oracle 선택 여부, 모델 크기에 따라 결과가 달라지므로 ViT-H 중심 대표 수치를 이 ViT-B 가중치의 점수로 옮기지 않았다. [논문 §7·부록 D](https://arxiv.org/html/2304.02643v1)

IoU head의 출력은 모델이 예상한 mask 품질이며 실제 정답과 비교한 IoU가 아니다. 여러 후보는 “사람 전체”와 “일부 영역”처럼 한 클릭이 모호할 때 선택지를 제공한다. 점수가 높다고 반드시 사용자가 원하는 대상이라는 보장은 없다. [공식 출력 구현](https://github.com/facebookresearch/segment-anything/blob/dca509fe793f601edb92606367a655c15ac00fdf/segment_anything/modeling/mask_decoder.py)

사진 편집의 선택 영역과 라벨링 보조 등에 활용할 수 있다는 것이 이번 분석의 해석이다. 원 공개 모델은 텍스트 입력으로 대상을 지정하는 기능을 제공하지 않으며, 자동 mask 생성도 점 grid와 필터링을 사용하는 별도 절차다. [공식 사용 안내](https://github.com/facebookresearch/segment-anything/blob/dca509fe793f601edb92606367a655c15ac00fdf/README.md)

[DETR](../10/2026-10-29-detr-resnet-50.md)이 클래스와 bounding box를 예측한다면 SAM은 prompt에 맞는 픽셀 영역을 만든다. [Depth Anything V2](2026-11-06-depth-anything-v2-small.md)의 픽셀 값은 상대 깊이인 반면 SAM의 값은 분할 mask다. 출력이 이미지 크기라는 사실만으로 역할을 같게 볼 수 없다.

출처·구조·중복을 검토했다. VRAM·응답 시간·분할 정확도를 실측하지 않았으며, 공개 모델의 용도와 구현 해석을 구분해 기록했다.
