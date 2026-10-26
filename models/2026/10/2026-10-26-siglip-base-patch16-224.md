# SigLIP-base-patch16-224: 두 12층 encoder와 이미지·문장 쌍의 sigmoid 학습

조사일: 2026-10-26 KST. 추론·학습과 가중치 다운로드 없이 공개 자료를 분석했다.

## 선정·식별

모델은 `google/siglip-base-patch16-224`, revision은 `7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed`다. 조사 시 [Hub API](https://huggingface.co/api/models/google/siglip-base-patch16-224)는 다운로드 1,084,847·좋아요 103을 반환했고 [모델 페이지](https://huggingface.co/google/siglip-base-patch16-224)는 다운로드 기간을 last month로 표시했다. 인기도는 성능 증거가 아니다. 기존 models·issues 전체 파일을 읽어 동일 모델 및 변형의 분석이 없음을 확인했다. 전날 음악 생성에 이어 이미지·텍스트 표현 분야를 선정했다.

Google 연구진의 SigLIP이며 목적은 zero-shot 이미지 분류와 이미지·문장 검색이다. 다른 공개 LLM을 기반으로 fine-tuning한 모델이라는 근거는 확인 불가다. 논문에는 무작위 초기화와 기존 시각 모델 초기화 등 여러 실험이 있어 이 Hub revision의 초기화 관계를 임의로 정하지 않았다. 라이선스는 Apache-2.0이다. [고정 카드](https://huggingface.co/google/siglip-base-patch16-224/blob/7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed/README.md).

[논문 최초 제출](https://arxiv.org/abs/2303.15343)은 2023-03-27, Base·Large·SO400M 결과가 추가된 v4는 2023-09-27이다. 이 날짜를 해당 Hub 체크포인트의 정확한 공개일로 대신하지 않았다. 체크포인트 배포일은 이번 조사에서 확인 불가다. Hub 카드는 개발팀 자체 작성본이 아니라 Hugging Face 팀 작성본이라고 명시되어 있다. 개발팀의 [버전 고정 공식 데모](https://github.com/google-research/big_vision/blob/0127fb6b337ee2a27bf4e54dea79cff176527356/big_vision/configs/proj/image_text/SigLIP_demo.ipynb)에는 B/16·224 모델이 별도로 등재되어 있다.

## 구조: 생략된 config 값은 구현 기본값까지 확인

[고정 config.json](https://huggingface.co/google/siglip-base-patch16-224/blob/7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed/config.json)은 일부 값만 저장한다. 아래의 12층·224 해상도·텍스트 64 위치 등 생략값은 [Transformers v4.37.0 설정 클래스](https://github.com/huggingface/transformers/blob/v4.37.0/src/transformers/models/siglip/configuration_siglip.py)의 기본값으로 해석했다. JSON에 직접 쓰여 있다고 주장하지 않는다.

| 항목 | 이미지 encoder | 텍스트 encoder |
|---|---|---|
| 종류 | ViT, 양방향 dense encoder | 양방향 dense Transformer encoder |
| 레이어 | 12 | 12 |
| hidden / FFN dimension | 768 / 3,072 | 768 / 3,072 |
| attention heads | 12, 표준 MHA | 12, 표준 MHA |
| head dimension | 768÷12=64, 계산 | 768÷12=64, 계산 |
| 입력 범위 | RGB 224×224, 16×16 패치 | 최대 64 토큰 위치 |
| 위치 표현 | 학습 가능한 패치 위치 임베딩 | 학습 가능한 토큰 위치 임베딩 |
| pooling | 학습된 probe를 이용한 multihead attention pooling | 마지막 토큰 표현 → 선형 head |
| 출력 표현 폭 | 768 | 768 |
| 활성화 | GELU의 tanh 근사 | GELU의 tanh 근사 |
| GQA·MoE | 해당 없음 | 해당 없음 |

[고정 전처리 설정](https://huggingface.co/google/siglip-base-patch16-224/blob/7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed/preprocessor_config.json)은 resize 224×224, RGB mean·std 각각 0.5를 명시한다. [토크나이저 설정](https://huggingface.co/google/siglip-base-patch16-224/blob/7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed/tokenizer_config.json)은 최대 길이 64와 EOS를 padding으로 쓰는 구성을 담고 있다. 어휘 슬롯은 config의 32,000이다. 64는 학습 epoch나 이미지 패치 수가 아니다.

총 파라미터는 Hub API safetensors 메타데이터 기준 **203,155,970**개이며 저장 dtype은 F32다. 전문가 선택형 활성 파라미터 구분은 적용하지 않는다. 구조 설명은 [고정 구현](https://github.com/huggingface/transformers/blob/v4.37.0/src/transformers/models/siglip/modeling_siglip.py)을 대조했다. 이미지 패치는 `(224÷16)²=196`개다. 이 모델은 이미지에 CLS 토큰을 추가해 그 하나만 쓰는 방식 대신, 학습된 probe가 패치들을 조회하는 pooling head를 둔다.

이미지와 문장은 별도 encoder에서 처리한 뒤 정규화된 벡터로 비교한다. 두 modality가 매 층 cross-attention으로 직접 대화하는 구조는 아니다. 텍스트 encoder는 원 CLIP의 causal mask와 달리 양방향으로 읽고, 구현은 마지막 위치를 sticky EOS라고 가정해 pooling한다. 전처리·padding을 바꾸면 이 가정도 주의해야 한다. [구현](https://github.com/huggingface/transformers/blob/v4.37.0/src/transformers/models/siglip/modeling_siglip.py).

## sigmoid loss가 바꾸는 것

기존 [CLIP 분석](2026-10-09-clip-vit-base-patch32.md)의 핵심은 이미지·텍스트 표현 정렬이다. SigLIP도 같은 종류의 목적을 갖지만, 배치 안에서 softmax로 상대 확률을 정규화하는 대신 각 이미지·문장 쌍이 맞는지 sigmoid로 학습한다. 정답 쌍은 양성, 서로 다른 예제의 조합은 음성으로 둔다. 관련 없는 쌍이라는 가정에는 노이즈가 있을 수 있다. [논문 v4 §3, Algorithm 1](https://arxiv.org/pdf/2303.15343v4).

쌍의 점수는 개념적으로 `s=exp(t′)×cos(image,text)+b`이며 학습 가능한 scale과 bias를 포함한다. `sigmoid(s)`는 후보 목록 전체에서 합이 1이 되도록 만들지 않는다. 이를 업무 데이터에서 검증된 정답 확률로 받아들이거나 임계값 0.5가 항상 적절하다고 보장해서는 안 된다. 후보·prompt·분포에 맞춘 검증이 필요하다. [논문](https://arxiv.org/pdf/2303.15343v4), [배포 구현](https://github.com/huggingface/transformers/blob/v4.37.0/src/transformers/models/siglip/modeling_siglip.py).

이 손실은 전체 배치의 정규화에 대한 의존을 줄이지만 다른 장치의 음성 쌍 정보나 분산 통신이 모두 사라진다는 뜻은 아니다. 논문의 chunked 계산도 쌍을 나눠 처리한다. 손실만 바꾼 효과와 패치 크기·데이터·학습량 차이도 구분해야 한다.

## 학습: 계열 실험을 배포 체크포인트의 이력으로 옮기지 않기

카드는 WebLI의 영어 이미지·문장 쌍으로 사전학습했다고 설명한다. 다만 개발팀의 상세 원 실행 설정과 이 Hub revision을 연결하는 기록은 이번 조사에서 확인하지 못했다. 공식 big_vision 데모와 구현은 확인했지만 대응되는 완전한 학습 recipe는 확인 불가다.

| 항목 | 이 Hub revision에 대해 확인한 범위 |
|---|---|
| 사전학습 목표 | 이미지·문장 쌍의 sigmoid 정렬 |
| 데이터 | WebLI 영어 쌍; 해당 revision의 고유 쌍 수·처리 쌍 수 확인 불가 |
| epoch / optimizer steps | 확인 불가 / 확인 불가 |
| batch / gradient accumulation | 체크포인트 전용 값 확인 불가 / 확인 불가 |
| learning rate / scheduler | 체크포인트 전용 값 확인 불가 |
| optimizer / precision | 체크포인트 전용 값 확인 불가; 저장 F32로 학습 precision을 추정하지 않음 |
| 하드웨어 | HF 작성 카드는 16 TPU-v4 chips·3일이라고 기재; 원 실행 로그로의 대응 확인 불가 |
| SFT·DPO·RL·LoRA | 적용 근거 확인 불가 |

출처는 [고정 카드](https://huggingface.co/google/siglip-base-patch16-224/blob/7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed/README.md)다. ‘확인 불가’는 이번 조사에서 체크포인트 단위 근거를 확인하지 못했다는 뜻이며 모든 자료에 미공개라고 단정하는 표현과 구분한다. 논문 v4 표 1에도 16 TPU-v4·3일·batch 16k의 B/16 실험이 있으나, 같은 크기라는 이유로 그 설정과 성능을 이 가중치의 이력으로 확정하지 않았다. [논문 표 1](https://arxiv.org/pdf/2303.15343v4).

레이어 12는 네트워크 깊이, epoch는 학습 데이터 반복 횟수, step은 optimizer 업데이트 횟수다. 논문 속 examples seen 역시 고유 예제 수와 다르며 batch·샘플링·초기화 관계 없이 epoch를 계산할 수 없다.

## 평가·활용 판단

논문 표 1의 ImageNet zero-shot 71.0%는 사전학습 시각 B/16을 초기화에 쓴 특정 실험이다. 무작위 초기화 B/16 실험과 잠근 이미지 tower의 SigLiT 실험도 별도로 등장한다. 따라서 초록의 84.5%를 이 SigLIP-base 체크포인트 성능으로 쓰지 않는다. 실제 배포 revision에 정확히 대응하는 평가 점수·prompt·프로토콜을 이번에 연결하지 못해 별도 성능 순위를 싣지 않았다. [논문 v4 표 1 및 §4](https://arxiv.org/pdf/2303.15343v4).

이미지·텍스트 검색이나 자연어 라벨 후보를 비교하는 분류에 사용할 수 있다. CLIP ViT-B/32와 달리 이 모델은 16픽셀 패치를 사용하지만, 더 촘촘한 패치만으로 품질·속도 우위를 보장하지 않는다. 문장 생성·이미지 캡션 작성 모델은 아니며 영어 학습 모델의 한국어 검색 품질도 따로 확인해야 한다. 같은 이미지·라벨 prompt와 같은 평가 데이터에서 recall@k나 분류 정확도를 비교하는 것이 실용적이다. [고정 카드](https://huggingface.co/google/siglip-base-patch16-224/blob/7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed/README.md).

F32 가중치만의 계산은 `203,155,970×4=812,623,880 bytes`, 약 812.6MB(십진 단위)다. 실제 VRAM 요구량이나 속도 측정이 아니며 배치·activation·런타임·전처리 메모리는 별도다.
