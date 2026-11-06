# Depth Anything V2 Small: 12층 이미지 encoder와 4개 특징 지도로 복원하는 상대 깊이

조사일: 2026-11-06 (한국 시간)

## 선정과 모델 식별

Hugging Face에서 최근 한 달 다운로드 **33,484회**, 누적 좋아요 **85개**를 확인했다. 현재 사용되는 원본 모델 중 깊이 추정 분야를 골랐다. 이 지표는 사용·관심의 단서이며 정확도나 트렌딩 순위를 증명하지 않는다. 기존 `models`와 `issues` 전체 기록을 읽었으며 동일 체크포인트·미러·양자화 모델의 분석은 없었다. [모델 페이지](https://huggingface.co/depth-anything/Depth-Anything-V2-Small), [조회 API](https://huggingface.co/api/models/depth-anything/Depth-Anything-V2-Small)

| 항목 | 확인 내용 |
|---|---|
| 모델 ID | `depth-anything/Depth-Anything-V2-Small` |
| 개발 | Depth Anything V2 연구진, HKU·TikTok |
| HF revision | `03876f8651c73a60fe4c2c48294e09fcb6838fcf` |
| 공식 코드 revision | `a561b849ebae10a6f5ef49e26c83cbbcd36c71bf` — 조사 시점 코드이며 학습 실행 당시 commit이라는 의미는 아니다. |
| 발표 | 논문 v1 2024-06-13, 공식 저장소의 코드·모델 공개 공지 2024-06-14 |
| 기반 / 용도 | DINOv2 ViT-S/14 기반, 단안 RGB 이미지의 상대 깊이 추정 |
| 라이선스 | 이 Small 체크포인트는 Apache-2.0. 다른 크기에 자동 적용하지 않는다. |

근거: [고정 모델 카드](https://huggingface.co/depth-anything/Depth-Anything-V2-Small/blob/03876f8651c73a60fe4c2c48294e09fcb6838fcf/README.md), [공식 저장소 공지·모델 표](https://github.com/DepthAnything/Depth-Anything-V2/blob/a561b849ebae10a6f5ef49e26c83cbbcd36c71bf/README.md), [논문 v1](https://arxiv.org/html/2406.09414v1).

## 구조와 각 계층의 역할

이 HF 저장소에는 `config.json`이 없다. `.pth` 가중치와 카드가 제공되는 공식 원본이며, 구조는 카드의 생성 인자와 공식 Python 구현을 대조했다. 가중치를 내려받거나 실행하지 않았다.

| 구성 | 값 / 해석 |
|---|---|
| 전체 파라미터 | 공식 표 24.8M, 약 2,480만 개. 반올림된 수치이며 직접 집계하지 않았다. Dense 구조로 MoE 활성 전문가 구분은 없다. |
| encoder | DINOv2 `vits`, 12개 Transformer block |
| hidden dimension | 384 |
| attention / KV heads | Q·K·V 각각 6개인 full self-attention. 이미지 전체 패치를 서로 참고하며 causal mask를 쓰는 언어 모델 구조가 아니다. |
| head dimension | 64 = 384 ÷ 6, 코드에서 유도 |
| FFN | MLP 비율 4, 중간 차원 1,536 = 384 × 4, GELU |
| 패치 | 14 × 14 픽셀 |
| 위치 정보 | 학습 가능한 절대 위치 embedding, 다른 패치 격자에 bicubic 보간 |
| 토큰 | CLS 토큰 1개, register token 0개. 기본 depth head는 CLS를 결합하지 않는다. |
| 중간 출력 | 0부터 센 block `[2,5,8,11]`, 즉 3·6·9·12번째 출력 |
| DPT decoder | 네 특징 지도를 채널 `[48,96,192,384]`로 투영하고, 공통 features=64로 맞춰 4개 fusion block에서 결합 |
| 최종 출력 | 1채널, ReLU 후 원본 이미지 크기로 보간 |
| 기본 전처리 | `input_size=518`, 종횡비 유지 및 14의 배수 맞춤. 항상 518×518로 찌그러뜨리는 처리가 아니다. |
| 미사용 구성 | 텍스트 decoder, VAE, diffusion sampling loop 없음 |

근거: [공식 모델 크기·Small 생성 인자](https://github.com/DepthAnything/Depth-Anything-V2/blob/a561b849ebae10a6f5ef49e26c83cbbcd36c71bf/README.md), [DINOv2 생성·위치 보간 코드](https://github.com/DepthAnything/Depth-Anything-V2/blob/a561b849ebae10a6f5ef49e26c83cbbcd36c71bf/depth_anything_v2/dinov2.py), [QKV 구현](https://github.com/DepthAnything/Depth-Anything-V2/blob/a561b849ebae10a6f5ef49e26c83cbbcd36c71bf/depth_anything_v2/dinov2_layers/attention.py), [DPT·중간 출력·전처리](https://github.com/DepthAnything/Depth-Anything-V2/blob/a561b849ebae10a6f5ef49e26c83cbbcd36c71bf/depth_anything_v2/dpt.py).

이미지를 패치로 나눠 encoder가 전체 장면의 관계를 읽고, 서로 다른 깊이의 네 출력을 공간 격자로 되돌린다. DPT는 이를 서로 다른 해상도로 조절한 뒤 거친 정보부터 세부 정보까지 결합한다. 마지막 합성곱이 각 픽셀의 깊이 값을 만든다. 12층은 특징을 변환하는 네트워크 깊이이지, 이미지를 12번 학습했다는 뜻이 아니다. [encoder](https://github.com/DepthAnything/Depth-Anything-V2/blob/a561b849ebae10a6f5ef49e26c83cbbcd36c71bf/depth_anything_v2/dinov2.py), [decoder](https://github.com/DepthAnything/Depth-Anything-V2/blob/a561b849ebae10a6f5ef49e26c83cbbcd36c71bf/depth_anything_v2/dpt.py)

518×518 정사각형을 가정하면 패치 토큰은 `(518÷14)²=1,369`개, CLS 포함 1,370개다. 이는 계산 예시이며 직사각형 입력에서는 달라진다. 해상도가 높아지면 attention의 토큰 간 관계 계산도 늘어난다. [전처리·패치 구현](https://github.com/DepthAnything/Depth-Anything-V2/blob/a561b849ebae10a6f5ef49e26c83cbbcd36c71bf/depth_anything_v2/dpt.py)

## 단계별 학습: 교사의 합성 데이터와 학생의 실제 이미지

| 단계 | 데이터 / batch / iteration |
|---|---|
| 기반 encoder 사전학습 | DINOv2 초기화. 원본 사전학습 epoch·step·장비는 이번 조사에서 확인 불가. |
| 교사 학습 | DINOv2-G, 합성 라벨 이미지 595K, batch 64, 160K iteration |
| 의사 라벨 생성 | 실제 이미지 약 62M에 교사가 깊이 부여. optimizer 학습 단계와 구분. |
| Small 학생 학습 | 의사 라벨 실제 이미지, batch 192, 480K iteration. 최종 학생은 합성 이미지를 섞지 않음. |
| 두 깊이 학습 단계의 optimizer / LR | Adam, encoder 5e-6 / decoder 5e-5 |
| 학습 입력 | 518×518 crop |
| epoch / accumulation / scheduler | 검토한 학습 설명에서 미공개 |
| precision / 하드웨어 | 해당 실행의 구체 설정 확인 불가 |
| SFT·DPO·RL·LoRA | 이 체크포인트의 학습 단계로 보고되지 않음 |

근거: [논문 §5·§7.1·§7.4](https://arxiv.org/html/2406.09414v1). iteration은 논문 표현이며 accumulation 설정을 확인하지 못해 별도의 optimizer update 수로 재해석하지 않았다. 공개된 metric-depth 학습 코드를 이 상대 깊이 모델의 원본 학습 설정으로 대신 사용하지 않았다.

학생이 보는 것은 교사가 실제 사진에 붙인 깊이 지도다. 따라서 합성 장면의 정확한 라벨을 실제 사진으로 전달하는 학습으로 이해할 수 있다. batch와 iteration을 곱해도 sampling·반복 조건이 빠져 있어 정확한 epoch가 되지 않는다. [학습 파이프라인](https://arxiv.org/html/2406.09414v1)

## 평가, 비교와 활용 범위

공개 논문의 DA-2K에서 ViT-S 정확도는 **95.3%**다. 1,000장·2,000개 픽셀 쌍의 앞뒤 관계를 평가하며, 모든 픽셀의 미터 오차를 뜻하지 않는다. KITTI·NYU-D 등 기존 zero-shot 평가에서는 V1과 비슷하고 일부 지표는 낮다. 한 벤치마크를 근거로 전 영역 우위를 단정할 수 없다. [논문 Table 2·3, §6](https://arxiv.org/html/2406.09414v1)

출력은 scale·shift가 정해지지 않은 **inverse depth**다. 큰 값은 가까운 쪽으로 해석하지만, 값 2가 2미터라는 뜻은 아니다. 거리 계측에는 별도 metric 모델이나 보정이 필요하다. [논문 §5.2](https://arxiv.org/html/2406.09414v1)

이미지 편집용 깊이 조건, 전경·배경 분리의 보조 정보 등에 활용할 수 있다는 것이 이번 분석의 해석이다. 안전한 거리 측정이나 완전한 3D 복원을 보장하는 모델로 해석하지 않는다. 공식 저장소의 영상 처리 기능도 프레임 간 일관성을 자동 보장하지 않는다. [공식 용도·영상 안내](https://github.com/DepthAnything/Depth-Anything-V2/blob/a561b849ebae10a6f5ef49e26c83cbbcd36c71bf/README.md)

이전에 다룬 [DINOv2-base](../10/2026-10-20-dinov2-base.md)는 범용 특징 encoder다. 오늘 모델은 Small encoder에 깊이 decoder와 별도 학습을 더한 작업 모델이어서 단순 미러가 아니다. 같은 V2의 Base는 공식 표상 97.5M으로 더 크지만, 실제 속도·메모리와 정확도는 입력·환경·평가 조건을 맞춰 비교해야 한다. [공식 크기 표](https://github.com/DepthAnything/Depth-Anything-V2/blob/a561b849ebae10a6f5ef49e26c83cbbcd36c71bf/README.md)

이번에는 모델 로딩·학습·추론을 실행하지 않았다. 논문 보고값과 코드 해석만 다루며 VRAM·지연 시간의 실측값은 제시하지 않는다.
