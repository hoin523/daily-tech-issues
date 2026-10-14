# BLIP image-captioning base: 이미지를 읽는 12층과 설명을 쓰는 12층

조사일: 2026-10-15 KST. ID `Salesforce/blip-image-captioning-base`, revision `82a37760796d32b1411fe092ab5d4e227313294b`.

## 선정과 식별

Salesforce의 이미지 설명 생성 모델이다. Hub 라이선스는 BSD-3-Clause이며 카드의 사용·제약 설명도 함께 확인해야 한다. 정확한 최초 체크포인트 발표일은 확인 불가다. BLIP 논문 최초 공개일 2022-01-28을 배포일과 구분한다. 기반 계열은 ViT-base 이미지 backbone을 사용하는 BLIP이며, 원 논문은 이미지·텍스트 Transformer를 ImageNet ViT와 BERT-base에서 초기화한다고 설명한다. [고정 카드](https://huggingface.co/Salesforce/blip-image-captioning-base/blob/82a37760796d32b1411fe092ab5d4e227313294b/README.md), [논문](https://arxiv.org/html/2201.12086v1)

조사 당시 최근 한 달 다운로드 1,340,810회·좋아요 900개였다. 좋아요 집계 기간은 확인 불가이며 인기도는 품질의 증거가 아니다. [HF 페이지](https://huggingface.co/Salesforce/blip-image-captioning-base), [API](https://huggingface.co/api/models/Salesforce/blip-image-captioning-base)

전체 models·issues를 읽고 동일 모델 분석이 없음을 확인했다. 기존 CLIP·ViT와 시각 Transformer라는 공통점은 있지만 이미지에서 문장을 생성하는 별도 모델이다. 큰 가중치 다운로드·추론·학습은 수행하지 않았다.

## 구조

표는 [고정 config](https://huggingface.co/Salesforce/blip-image-captioning-base/blob/82a37760796d32b1411fe092ab5d4e227313294b/config.json)와 [전처리 설정](https://huggingface.co/Salesforce/blip-image-captioning-base/blob/82a37760796d32b1411fe092ab5d4e227313294b/preprocessor_config.json)에 근거한다.

| 항목 | 이미지 encoder | 텍스트 decoder |
|---|---|---|
| Transformer layers | 12 | 12 |
| hidden / FFN intermediate | 768 / 3,072 | 768 / 3,072 |
| heads | 12 | 12 |
| head dimension | 768÷12=64, 균등 분할 가정 계산 | 같은 계산 |
| 입력 크기·상한 | RGB 384×384, 패치 16×16 | 위치 임베딩 512 슬롯 |
| 어휘 | 해당 없음 | 30,524 슬롯 |
| 활성화 | GELU | GELU |
| attention | 이미지 self-attention | causal self-attention + 이미지 cross-attention |
| 위치 표현 | 학습 가능한 absolute 임베딩 | 학습 가능한 absolute 임베딩 |
| MoE·GQA | 전문가·별도 GQA 설정 없음 | 동일 |

전체 유형은 `BlipForConditionalGeneration`, 저장 dtype는 float32다. 정확한 총·구성요소별 파라미터 집계는 이번 조사에서 확인 불가이며, 논문의 전체 학습 프레임워크 파라미터를 이 캡션 전용 배포본 총수로 옮기지 않는다. MoE식 활성 파라미터 구분은 해당 없다.

텍스트 config에 `add_cross_attention=false`가 있어도 이를 이미지 참조가 없다는 뜻으로 읽으면 안 된다. BLIP 전용 구현이 encoder hidden states를 받아 cross-attention을 실행한다. [Transformers v4.26.0 텍스트 구현](https://github.com/huggingface/transformers/blob/v4.26.0/src/transformers/models/blip/modeling_blip_text.py)

**계산과 설명:** 384÷16=24, 패치는 24²=576개다. CLS 하나를 포함하면 577개 이미지 위치를 처리한다. 이미지 encoder가 만든 표현을 decoder가 참조하면서 다음 단어를 생성한다. 이 모델의 12+12층은 학습 24회를 의미하지 않는다.

카드는 텍스트 접두어가 있는 conditional caption과 접두어 없는 caption을 모두 예시로 제공한다. 여기서 접두어가 없어도 이미지는 입력된다. 생성 설정 `max_length=20`은 구조상 512 위치 상한이나 학습 epoch와 다르다. 최대 길이 설정만으로 긴 설명 품질도 보장하지 않는다. [사용 예제](https://huggingface.co/Salesforce/blip-image-captioning-base/blob/82a37760796d32b1411fe092ab5d4e227313294b/README.md)

전처리 크기는 384이며 RGB mean=[0.48145466,0.4578275,0.40821073], std=[0.26862954,0.26130258,0.27577711]이다. 이전 ViT 분류 모델의 0.5 정규화를 그대로 복사하지 않는다. [고정 전처리](https://huggingface.co/Salesforce/blip-image-captioning-base/blob/82a37760796d32b1411fe092ab5d4e227313294b/preprocessor_config.json)

## 학습: 계열 논문·공개 설정·실제 로그 구분

BLIP 계열은 이미지·텍스트 대조, 매칭, 언어 생성 목표와 CapFilt 데이터 개선을 설명한다. CapFilt는 생성한 캡션과 필터를 활용하는 데이터 구축 과정이며, 배포 모델이 추론마다 데이터 필터까지 실행한다는 뜻은 아니다. [논문 §3–4](https://arxiv.org/html/2201.12086v1#S3)

| 항목 | BLIP ViT-B 논문 사전학습 | 공개 COCO caption 설정 |
|---|---|---|
| epoch | 20 | max_epoch=5 |
| step | 확인 불가 | 실제 완료 step 확인 불가 |
| 데이터 | 기본 14M 이미지; 129M 확장 실험도 구분 | COCO, 정확한 해당 실행 샘플 수 확인 불가 |
| batch | 논문 2,880 | batch_size=32; global/per-device는 설정 파일만으로 확정 안 함 |
| learning rate | warmup peak 3e-4 | init_lr=1e-5, min_lr=0 |
| optimizer | AdamW | 설정 파일에는 optimizer 종류 미기재, 확인 불가 |
| weight decay | 0.05 | 0.05 |
| scheduler | 논문은 decay 설명 제공 | 파일만으로 확인 불가 |
| gradient accumulation | 확인 불가 | 확인 불가 |
| precision | 확인 불가 | 확인 불가 |
| hardware | 논문 16-GPU node 두 개 | 실제 해당 실행 구성 확인 불가 |
| SFT·DPO·RL·LoRA | 별도 적용 근거 확인 불가 | 캡션 지도학습과 LLM 후처리 용어를 혼용하지 않음 |

사전학습 표의 근거는 [논문 §4.1](https://arxiv.org/html/2201.12086v1#S4.SS1), 캡션 표는 [공식 caption_coco.yaml](https://github.com/salesforce/BLIP/blob/main/configs/caption_coco.yaml)이다. YAML은 움직이는 main의 레시피이며 HF revision의 실제 학습 로그가 아니다. 특히 pretrained 경로가 이미 caption 체크포인트를 가리키므로 5 epoch를 배포본의 처음부터 끝까지 전체 학습량으로 해석하지 않는다.

HF 카드만으로 정확한 14M/129M·CapFilt 변형과 실제 실행 로그의 대응을 확정하지 못했다. 확인 불가 값은 추정하지 않았다. Step은 업데이트, epoch는 데이터 반복, 추론 beam 수는 탐색 방식으로 서로 다르다.

## 평가와 비교

원 논문은 COCO 캡션 BLEU-4·CIDEr, NoCaps CIDEr·SPICE 등을 사용한다. 사전학습 데이터·backbone·CapFilt 변형에 따라 행이 다르다. 이번 배포본과 특정 실험 행의 정확한 대응 및 전체 디코딩 조건을 확인하지 못했으므로 계열 최고 점수를 이 모델의 성능으로 싣지 않았다. [원 논문 평가 표](https://arxiv.org/html/2201.12086v1)

실제 비교는 동일 이미지 split·참조 캡션·beam/길이 설정·평가 구현을 맞춰야 한다. 캡션 지표가 높아도 물체 수·관계·세부 속성이 항상 정확하다는 뜻은 아니다.

이전 [CLIP 분석](2026-10-09-clip-vit-base-patch32.md)은 이미지·문장 유사도 비교, [ViT 분석](2026-10-13-vit-base-patch16-224.md)은 고정 라벨 분류였다. BLIP 캡션 배포본은 문장 생성이라는 출력 목적이 달라 같은 점수로 우열을 비교할 수 없다.

## 활용과 제약

이미지 설명 초안·검색 메타데이터 생성에 검토할 수 있다. 존재하지 않는 대상이나 관계를 설명할 수 있으므로 이미지 사실 확인과 접근성 설명의 최종 검토가 필요하다. 이 영어 캡션 모델을 한국어 생성·OCR·범용 이미지 질의응답 모델로 간주하지 않는다. 카드도 모든 downstream 용도를 평가한 것은 아니라고 설명한다. [카드](https://huggingface.co/Salesforce/blip-image-captioning-base/blob/82a37760796d32b1411fe092ab5d4e227313294b/README.md)

VRAM·속도·캡션 품질을 실측하지 않았다. 이미지 토큰 수·생성 길이·batch·정밀도와 실행 환경에 따라 비용이 달라지므로 추론 예제의 FP16 선택을 실제 학습 precision이나 메모리 측정값으로 쓰지 않는다.
