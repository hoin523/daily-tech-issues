# Swin-Tiny: 4개 stage와 12개 블록으로 만드는 이미지 계층 구조

조사일: 2026-10-30 KST. 가중치 다운로드·추론·학습 없이 공개 자료를 분석했다.

## 선정·식별

모델 ID는 `microsoft/swin-tiny-patch4-window7-224`, revision은 `d00d478bfaf1f417d34a1186673ad4b4be264c51`다. 조사 시 [Hub API](https://huggingface.co/api/models/microsoft/swin-tiny-patch4-window7-224)는 다운로드 24,948·좋아요 58을 반환했고 [페이지](https://huggingface.co/microsoft/swin-tiny-patch4-window7-224)는 다운로드 기간을 last month로 표시했다. 최상위 대형 모델보다 지표는 작지만 지속적인 사용 지표와 공식 분석 자료를 갖춘 모델이다. 인기도는 성능 증거가 아니다.

기존 models·issues 전체 파일을 읽어 Swin 계열 분석이 없음을 확인했다. 이번에는 Microsoft의 계층형 이미지 분류 모델을 선정했다. 같은 비전 분야라도 전날 DETR의 집합 탐지 구조와 출력 목적이 다르다. ImageNet-1K에서 224×224 이미지로 학습한 모델이며 Apache-2.0 라이선스다. 다른 공개 기반 체크포인트에서 추가 fine-tuning됐다는 근거는 확인 불가이며, 논문은 이 ImageNet-1K 설정을 from-scratch 학습으로 설명한다. [고정 카드](https://huggingface.co/microsoft/swin-tiny-patch4-window7-224/blob/d00d478bfaf1f417d34a1186673ad4b4be264c51/README.md), [논문 §4.1](https://arxiv.org/html/2103.14030v2#S4.SS1).

논문 계열의 최초 발표는 [arXiv 기록](https://arxiv.org/abs/2103.14030) 기준 2021-03-25다. HF 변환 체크포인트의 정확한 최초 배포일은 이번 조사에서 확인 불가다. 카드가 Hugging Face 팀 작성본임을 명시하므로 학습 설명은 공식 코드·논문과 대조했다.

## 구조: stage 수와 block 수를 구분

[고정 config](https://huggingface.co/microsoft/swin-tiny-patch4-window7-224/blob/d00d478bfaf1f417d34a1186673ad4b4be264c51/config.json)의 depths=[2,2,6,2], embed_dim=96, num_heads=[3,6,12,24], patch_size=4, window_size=7을 확인했다. 아래 격자·차원 변화는 [공식 구현](https://github.com/microsoft/Swin-Transformer/blob/f82860bfb5225915aca09c3227159ee9e1df874d/models/swin_transformer.py)의 stage별 채널 배증과 patch merging을 적용한 계산이다.

| stage | 블록 수 | 224 입력의 특징 격자 | hidden dimension | heads | head dimension | MLP 폭, ratio 4 |
|---|---|---|---|---|---|---|
| 1 | 2 | 56×56 | 96 | 3 | 32 | 384 |
| 2 | 2 | 28×28 | 192 | 6 | 32 | 768 |
| 3 | 6 | 14×14 | 384 | 12 | 32 | 1,536 |
| 4 | 2 | 7×7 | 768 | 24 | 32 | 3,072 |

합계는 `2+2+6+2=12` Transformer 블록이다. 4개 stage 사이에는 3회의 patch merging이 있다. 이것이 4 epoch나 12번 추론한다는 뜻은 아니다. 처음에는 `224÷4=56`으로 4×4 픽셀 patch를 96차원에 투영한다. Merging은 인접 2×2 토큰을 합쳐 4C로 만든 후 선형 변환으로 2C에 맞추므로 공간 위치는 1/4, 채널은 2배가 된다. [구현](https://github.com/microsoft/Swin-Transformer/blob/f82860bfb5225915aca09c3227159ee9e1df874d/models/swin_transformer.py).

| 공통 항목 | 값·의미 |
|---|---|
| attention | 7×7 토큰 window 안의 MHA, W-MSA와 shifted W-MSA 교대 |
| 위치 표현 | 학습 가능한 상대 위치 bias; HF use_absolute_embeddings=false |
| 활성화·정규화 | GELU / LayerNorm, eps=1e-5 |
| QKV bias | true |
| 분류 head | 마지막 특징을 정규화·평균 pooling 후 1,000개 ImageNet 클래스 |
| GQA·MoE·텍스트 context | 해당 없음 |
| HF 저장 dtype | float32 |

7은 픽셀 수가 아니라 해당 stage 특징 격자에서 window 한 변의 토큰 수다. Head dimension은 각 stage의 `hidden÷heads=32`에서 계산했다. 별도 KV head 축소 설정이 없는 표준 MHA이며 LLM의 KV cache 절약 방식과 혼동하지 않는다.

Hub 메타데이터는 F32 28,288,354개와 I64 28,812개 항목, 합계 28,317,166개를 보고한다. 정수 항목은 학습 가능한 실수 가중치와 구분해야 한다. 공식 구현에는 상대 위치 인덱스 같은 buffer가 있어 API 전체 합계를 그대로 ‘학습 파라미터’로 세지 않았다. 논문은 약 29M으로 반올림한다. [API](https://huggingface.co/api/models/microsoft/swin-tiny-patch4-window7-224), [구현](https://github.com/microsoft/Swin-Transformer/blob/f82860bfb5225915aca09c3227159ee9e1df874d/models/swin_transformer.py).

## 창을 이동하는 이유

고정된 window만 쓰면 경계 양쪽의 토큰이 직접 정보를 주고받기 어렵다. 다음 블록에서는 창 경계를 약 절반만큼 이동해 이전의 서로 다른 window에 있던 부분이 만난다. 구현은 cyclic shift와 attention mask를 이용하므로 이미지 반대쪽 끝이 회전 때문에 실제 이웃인 것처럼 섞이지 않게 한다. 마지막 7×7 stage처럼 특징 격자가 window만큼 작아지면 구현은 shift를 끈다. 모든 블록에서 무조건 3칸 이동한다고 쓰지 않는다. [논문 §3.2](https://arxiv.org/html/2103.14030v2#S3.SS2), [구현](https://github.com/microsoft/Swin-Transformer/blob/f82860bfb5225915aca09c3227159ee9e1df874d/models/swin_transformer.py).

고정 window 크기에서는 토큰 수 증가에 따른 attention 계산의 증가가 전역 attention보다 완만하다. 이 복잡도 설명이 모든 이미지 크기·하드웨어에서 일정 배수 속도를 보장하지는 않는다. [ViT-B/16 분석](2026-10-13-vit-base-patch16-224.md)의 단일 해상도 토큰 처리와 달리 Swin은 단계별로 공간을 줄이고 채널을 키운다.

## 학습: 논문·코드·배포 설정의 차이

[논문 §4.1 및 부록 A2.1](https://arxiv.org/html/2103.14030v2#S4.SS1)은 ImageNet-1K from-scratch 분류 학습을 300 epochs, AdamW, batch 1,024, LR 0.001, weight decay 0.05, cosine decay와 20-epoch 선형 warmup으로 설명한다. 다음 표는 논문 설정과 조회한 공식 코드의 역할을 나눴다.

| 항목 | 공개 근거 |
|---|---|
| 데이터 | ImageNet-1K, 약 1.28M train images / 50K validation |
| epoch | 논문·코드 기본값 300 |
| optimizer steps | HF revision의 정확한 최종 step 확인 불가 |
| optimizer | AdamW, 공식 코드 betas=(0.9,0.999), eps=1e-8 |
| batch | 논문 global 1,024; 코드 DATA.BATCH_SIZE=128은 실행·분산 조건과 함께 해석 |
| learning rate | 논문 0.001; 코드 BASE_LR=5e-4는 스케일링 전 기본값 |
| scheduler | cosine, warmup 20 epochs |
| weight decay | 0.05 |
| gradient clipping | 논문 max norm 1, 조회 코드 기본 CLIP_GRAD=5.0 |
| gradient accumulation | 코드 기본 ACCUMULATION_STEPS=1; 원 HF 실행 인자 검증은 확인 불가 |
| precision·하드웨어 | 코드 AMP_ENABLE=true; HF 원 학습 dtype·GPU 구성은 이번 조사에서 확인 불가 |
| SFT·DPO·RL·LoRA | 이 이미지 분류 모델에 별도 적용했다는 근거 확인 불가 |

코드 근거는 [고정 config.py](https://github.com/microsoft/Swin-Transformer/blob/f82860bfb5225915aca09c3227159ee9e1df874d/config.py), [Tiny YAML](https://github.com/microsoft/Swin-Transformer/blob/f82860bfb5225915aca09c3227159ee9e1df874d/configs/swin/swin_tiny_patch4_window7_224.yaml)이다. 기본값과 논문 차이를 원 체크포인트의 오류로 판정하지 않고 버전·실행 인자 차이로 남겼다. 학습 명령과 런타임에서 달라질 수 있는 값을 추론 config만으로 채우지 않았다.

또한 Tiny YAML은 DROP_PATH_RATE=0.2인데 HF config는 drop_path_rate=0.1이다. Stochastic depth는 학습 regularization 설정이므로 배포 config의 0.1만 보고 원 stochastic-depth 비율이 0.1이었다고 확정할 수 없다. 반대로 코드를 읽었다고 이 Hub revision의 학습을 그대로 재현했다고 주장하지도 않는다. ‘확인 불가’는 이번 조사에서 실행별 근거를 확인하지 못했다는 뜻이다.

논문은 RandAugment·Mixup·CutMix·random erasing·stochastic depth를 사용하고 반복 증강·EMA는 사용하지 않았다고 설명한다. [고정 전처리](https://huggingface.co/microsoft/swin-tiny-patch4-window7-224/blob/d00d478bfaf1f417d34a1186673ad4b4be264c51/preprocessor_config.json)는 입력 크기 224와 ImageNet RGB mean·std를 저장하지만 이 파일 하나가 전체 학습 증강 절차를 표현하지는 않는다.

## 평가와 사용 판단

논문 표 1의 ImageNet-1K 학습, 224×224 조건에서 Swin-T는 top-1 81.3%, DeiT-S는 79.8%다. 논문이 보고한 결과이며 이번 분석의 실측값이 아니다. 이 비교의 데이터·학습 recipe를 벗어난 모든 분류 작업의 우위를 보장하지 않는다. 논문 초록의 87.3%나 COCO 검출 수치를 이 Tiny 분류 체크포인트에 붙이지 않았다. [논문 평가표](https://arxiv.org/html/2103.14030v2#S4.SS1).

Swin은 detection·segmentation의 backbone으로도 쓰이지만 이 체크포인트의 출력은 ImageNet 분류다. 전날 [DETR 분석](2026-10-29-detr-resnet-50.md)처럼 box를 바로 출력하지 않으며 별도 head와 작업별 학습이 필요하다. 원본 ImageNet 범주 밖의 실무 이미지에는 fine-tuning과 동일 검증셋 평가가 필요하다.

F32 항목만의 저장 용량 계산은 `28,288,354×4=113,153,416 bytes`, I64 항목은 `28,812×8=230,496 bytes`다. 총 약 113.38MB(십진 단위)이며 파일 헤더·추가 상태는 제외했다. VRAM 요구량·추론 속도 실측이 아니고 activation·배치·런타임 메모리는 별도다.
