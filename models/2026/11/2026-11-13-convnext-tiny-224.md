# ConvNeXt-Tiny 224: 4개 stage와 18개 잔차 블록의 현대적인 CNN

조사일: 2026-11-13 (한국 시간). 공개 구성·논문·코드를 분석했으며 가중치 다운로드·학습·추론은 실행하지 않았다.

## 선정과 식별

Hugging Face API 기준 최근 한 달 다운로드 **32,121회**, 누적 좋아요 **33개**를 확인했다. 텍스트 모델에 이어 이미지 분류 분야의 순수 합성곱 모델을 선택했다. 인기 지표는 정확도나 트렌딩 순위를 뜻하지 않는다. [API](https://huggingface.co/api/models/facebook/convnext-tiny-224), [모델 페이지](https://huggingface.co/facebook/convnext-tiny-224)

| 항목 | 확인 내용 |
|---|---|
| 모델 | `facebook/convnext-tiny-224` |
| 개발 | FAIR·UC Berkeley 연구진, 원 논문 *A ConvNet for the 2020s* |
| HF revision | `6166b7613034066690a621d8bf25ffdf181a34f0` |
| 공식 코드 revision | `048efcea897d999aed302f2639b6270aedf8d4c8`, 조사 시점 코드이며 원 학습 commit은 확인 불가 |
| 기반 / 용도 | ImageNet-1k에서 학습한 224×224 이미지 분류 모델. 22k 사전학습 변형과 구분. |
| 발표 | 원 논문 v1 2022-01-10. 이 HF 변환 가중치의 최초 공개일은 확인 불가. |
| 라이선스 | HF 카드 Apache-2.0, 공식 코드 저장소 MIT. 각각의 배포물을 구분한다. |

근거: [고정 카드](https://huggingface.co/facebook/convnext-tiny-224/blob/6166b7613034066690a621d8bf25ffdf181a34f0/README.md), [공식 저장소](https://github.com/facebookresearch/ConvNeXt/blob/048efcea897d999aed302f2639b6270aedf8d4c8/README.md), [논문](https://arxiv.org/html/2201.03545v1). HF 카드는 원 개발팀 직접 작성본이 아니라 Hugging Face 팀 작성본이다.

기존 `models`·`issues` 전체 기록을 읽었으며 같은 체크포인트·미러·양자화 변형 분석은 없었다. 기존 ResNet·Swin과 비교할 수 있지만 동일 모델의 재소개는 아니다.

## 구조

| 항목 | 확인값 |
|---|---|
| 전체 크기 | 논문 표 약 29M. 정밀한 trainable 계수 수는 직접 확인하지 않았다. Dense 모델로 MoE 없음. |
| stem | 4×4 convolution, stride 4, RGB 3채널 → 96채널 |
| stage 깊이 | `[3,3,9,3]`, 합계 18개 잔차 block |
| stage 채널 | `[96,192,384,768]` |
| stage 사이 | LayerNorm → 2×2 convolution, stride 2 |
| 각 block | 7×7 depthwise convolution → LayerNorm → 4배 채널 확장 → GELU → 원 채널로 축소 → Layer Scale → residual addition |
| 확장 채널 | `[384,768,1536,3072]`, 채널 수×4 계산 |
| 분류 head | 공간 평균 → 정규화 → 768차원에서 1,000클래스 선형 분류 |
| Layer Scale | 초기값 1e-6 |
| HF LayerNorm epsilon | config 1e-12. 공식 PyTorch 코드의 1e-6과 구분. |
| attention·KV heads·위치 embedding | 없음. Transformer의 attention을 쓰지 않는 CNN. |

근거: [HF config](https://huggingface.co/facebook/convnext-tiny-224/blob/6166b7613034066690a621d8bf25ffdf181a34f0/config.json), [공식 block·stem·head](https://github.com/facebookresearch/ConvNeXt/blob/048efcea897d999aed302f2639b6270aedf8d4c8/models/convnext.py), [논문 모델 표](https://arxiv.org/html/2201.03545v1).

224×224 입력이라면 공간 격자는 stem부터 `56×56 → 28×28 → 14×14 → 7×7`로 줄어든다. padding·stride 구성에서 유도한 값이며 실행으로 측정한 것은 아니다. stage당 채널은 커지지만 공간 위치 수는 줄어 더 추상적인 표현을 만든다. [공식 구현](https://github.com/facebookresearch/ConvNeXt/blob/048efcea897d999aed302f2639b6270aedf8d4c8/models/convnext.py)

Depthwise convolution은 채널별로 공간 정보를 처리하고, 뒤의 두 선형 변환은 채널 사이를 섞는다. 확장 후 축소하는 inverted bottleneck과 입력을 더하는 residual 경로를 함께 쓴다. 구현에서 선형층으로 채널 변환을 표현해도 attention 모델이 되는 것은 아니다. [block 구현](https://github.com/facebookresearch/ConvNeXt/blob/048efcea897d999aed302f2639b6270aedf8d4c8/models/convnext.py)

## 학습 조건과 체크포인트의 범위

| 항목 | ImageNet-1K ConvNeXt-T 공개 recipe |
|---|---|
| 데이터 | ImageNet-1K, 약 1.2M 이미지·1,000클래스 |
| epoch | 300 |
| 총 optimizer step | 해당 실행 총수 미공개. dataset·sampler·drop-last 조건 없이 정확값으로 계산하지 않음. |
| effective batch | 4,096 |
| 다중 node 예제 | 4 node×8 GPU×장치별128×accumulation1 |
| 단일 node 예제 | 8 GPU×장치별128×accumulation4 |
| optimizer | AdamW, β1=0.9·β2=0.999 |
| LR / scheduler | 4e-3, 20 epoch linear warm-up 후 cosine decay |
| weight decay | 0.05 |
| 정규화·증강 | Tiny drop path 0.1, label smoothing 0.1, Mixup·CutMix·RandAugment·Random Erasing, EMA |
| precision / GPU 종류 | 공식 명령은 AMP 선택 옵션 제공. 이 HF 가중치 실행의 precision·GPU 종류는 확인 불가. |
| 후속 SFT·DPO·RL·LoRA | 이 분류 체크포인트의 학습 단계로 보고되지 않음 |

근거: [논문 §3.1·부록 A](https://arxiv.org/html/2201.03545v1), [공식 Tiny 학습 명령](https://github.com/facebookresearch/ConvNeXt/blob/048efcea897d999aed302f2639b6270aedf8d4c8/TRAINING.md). 명령은 재현 recipe이며 이 HF 파일의 개별 실행 로그까지 검증한 것은 아니다.

두 명령은 accumulation을 바꿔 effective batch를 같게 맞춘 사례다. microbatch 하나와 optimizer update 하나가 항상 같지는 않다. HF config의 `drop_path_rate=0.0`도 원 학습 recipe의 drop path 0.1을 부정하는 증거가 아니다. 추론용 모델 설정과 학습 정규화 설정을 구분해야 한다. [HF config](https://huggingface.co/facebook/convnext-tiny-224/blob/6166b7613034066690a621d8bf25ffdf181a34f0/config.json), [공식 명령](https://github.com/facebookresearch/ConvNeXt/blob/048efcea897d999aed302f2639b6270aedf8d4c8/TRAINING.md)

300은 데이터 반복 학습 epoch이고, 18은 block 수다. config의 `torch_dtype=float32`를 학습 전체가 FP32였다는 근거로 쓰지 않았다. [config](https://huggingface.co/facebook/convnext-tiny-224/blob/6166b7613034066690a621d8bf25ffdf181a34f0/config.json)

## 공개 평가와 활용 범위

논문 ImageNet-1K 표에서 ConvNeXt-T는 224×224, 약 29M·4.5G FLOPs 조건의 top-1 **82.1%**를 보고한다. 같은 표의 Swin-T 81.3%와 비교할 수 있지만 과거 보고 조건이며, 이번 HF 변환 가중치로 직접 재현한 수치는 아니다. 논문 초록의 87.8%를 Tiny의 성능으로 옮기지 않았다. [논문 ImageNet 평가 표](https://arxiv.org/html/2201.03545v1)

이미지 분류나 별도 학습의 backbone 출발점으로 활용할 수 있다는 것이 이번 분석의 해석이다. 1,000개 범주를 예측하는 head 자체가 임의의 대상 위치·mask를 제공하지는 않는다. [공식 분류 용도](https://huggingface.co/facebook/convnext-tiny-224/blob/6166b7613034066690a621d8bf25ffdf181a34f0/README.md)

[ResNet-50](2026-11-07-resnet-50.md)과는 block 내부·정규화·stem이 다르다. [Swin-Tiny](../10/2026-10-30-swin-tiny-patch4-window7-224.md)처럼 여러 해상도 stage를 갖지만, 이 모델에는 window self-attention이 없다. 구조의 계층성이 비슷해도 연산 방식까지 같지는 않다.

출처·값·중복 여부를 검토했다. 처리 속도·VRAM·새 데이터셋 정확도를 실측하지 않았으므로 실측값으로 제시하지 않는다.
