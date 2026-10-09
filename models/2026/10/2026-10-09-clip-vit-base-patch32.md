# CLIP ViT-B/32: 이미지와 문장을 같은 공간에서 비교하는 모델

- 조사일: 2026-10-09 (한국 시간)
- 모델 ID: `openai/clip-vit-base-patch32`
- revision: `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`
- 발표: 2021년 1월. 용도: 이미지·텍스트 임베딩, 텍스트 후보를 이용한 이미지 분류 및 검색
- 기반 모델: OpenAI가 학습한 CLIP ViT-B/32. HF 저장소는 Transformers용 배포본이다.
- 라이선스: 원본 OpenAI CLIP 저장소는 [MIT](https://github.com/openai/CLIP/blob/main/LICENSE). HF 모델 카드에는 별도 라이선스 메타데이터가 없어 배포본의 표기는 확인 불가.

[모델 카드](https://huggingface.co/openai/clip-vit-base-patch32/blob/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268/README.md)와 [공식 코드](https://github.com/openai/CLIP)를 기준으로 분석했다. 가중치를 내려받거나 추론 실험을 실행하지 않았다.

## 선정 근거와 중복 확인

조사 당시 HF 공개 지표는 최근 한 달 다운로드 **20,237,996회**, 좋아요 **1,581개**였다. 좋아요의 집계 기간은 별도로 표시되지 않는다. 이 지표는 사용 관심을 보여줄 뿐 정확도나 실제 이용자 수를 증명하지 않는다. [HF 모델 페이지](https://huggingface.co/openai/clip-vit-base-patch32), [조회 API](https://huggingface.co/api/models/openai/clip-vit-base-patch32)

기존 models 및 issues 기록을 확인했다. Qwen3, MiniLM, Whisper, SDXL에 이어 이번에는 멀티모달 인코더를 선정했다. SDXL의 텍스트 인코더 설명과 관련은 있지만 이 ViT-B/32 모델 자체를 구조 분석한 기록은 없다.

## 구조: 인코더 두 개와 공통 512차원 출력

아래 값은 [고정 revision의 config.json](https://huggingface.co/openai/clip-vit-base-patch32/blob/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268/config.json)에 근거한다.

| 항목 | 이미지 인코더 | 텍스트 인코더 |
|---|---|---|
| 유형 | Vision Transformer, ViT-B/32 | Transformer |
| 레이어 수 | 12 | 12 |
| hidden dimension | 768 | 512 |
| FFN intermediate dimension | 3,072 | 2,048 |
| attention heads | 12 | 8 |
| head dimension | 64 (768÷12, 계산) | 64 (512÷8, 계산) |
| 입력 크기/길이 | 224×224 픽셀, 패치 32×32 | 최대 77 토큰, 특수 토큰 포함 |
| 어휘 크기 | 해당 없음 | 49,408 |
| 공통 투영 차원 | 512 | 512 |
| 활성화 함수 | QuickGELU | QuickGELU |
| 위치 표현 | 학습 가능한 위치 임베딩 | 학습 가능한 위치 임베딩 |
| attention | 이미지 토큰 간 self-attention | causal mask를 사용하는 self-attention |
| KV heads / MoE | 별도 GQA·MoE 설정 없음 | 별도 GQA·MoE 설정 없음 |
| 총·활성 파라미터 수 | config에 총수 명시 없음. 정확한 체크포인트 집계는 확인 불가 | 같은 기준 |

위치 임베딩과 causal mask는 [원본 구현](https://github.com/openai/CLIP/blob/main/clip/model.py)에서 확인할 수 있다. 코드 링크는 움직이는 main이므로 재현 시점에 주의한다. 텍스트 인코더에 causal mask가 있어도 이 배포 모델의 용도는 다음 단어 생성이 아니라 문장 표현 추출이다.

**계산과 설명:** 224×224 입력을 32×32 패치로 나누면 (224÷32)² = 49개다. 여기에 클래스 토큰 하나를 더해 이미지 Transformer는 50개 위치를 처리한다. 이는 해당 입력 해상도와 표준 클래스 토큰 사용을 전제로 한 계산이다. 레이어 12개는 입력을 처리하는 깊이이며 학습 epoch 12회를 뜻하지 않는다.

사진과 문장을 각각 인코더에 넣고 512차원으로 투영한 뒤 정규화하여 유사도를 비교한다. 예를 들어 사진에 대해 “a photo of a cat”, “a photo of a dog”를 후보로 넣으면 각 문장과의 점수를 비교할 수 있다. 점수는 후보 문구와 후보 집합에 의존한다. 이를 현실 세계의 절대적인 확률로 해석하면 안 된다. [공식 사용 예제](https://github.com/openai/CLIP#usage)

## 학습: 32 epoch와 12 layer는 서로 다른 숫자

아래는 원본 CLIP 논문에서 확인한 학습 조건이다. [논문 §2.2–2.5](https://arxiv.org/html/2103.00020v1#S2)

| 항목 | 확인 결과 |
|---|---|
| 사전학습 목표 | 대응 이미지·텍스트 쌍을 가까이, 다른 쌍을 멀리 두는 대조 학습 |
| 데이터 규모 | 인터넷에서 수집한 WIT 이미지·텍스트 4억 쌍 |
| epoch | 32. 논문은 실험한 모든 모델에 이 조건을 적용했다고 설명 |
| batch size | 32,768 쌍 |
| optimizer | Adam, decoupled weight decay |
| learning-rate scheduler | cosine schedule |
| precision | mixed precision |
| 실제 optimizer step 수 | 확인 불가: 이 체크포인트의 학습 로그를 확보하지 못함 |
| gradient accumulation | 확인 불가 |
| ViT-B/32의 learning rate·warmup 상세 | 확인 불가: 이번에 확인한 본문에서 모델별 상세값을 검증하지 못함 |
| ViT-B/32의 하드웨어·학습 시간 | 확인 불가 |
| SFT·DPO·RL·LoRA | 해당 배포본에 적용했다는 근거 없음. 단계별 epoch·batch 등도 확인 불가 |

논문의 큰 ViT-L 모델 학습 시간이나 추가 고해상도 학습 조건을 B/32에 옮겨 적지 않았다. epoch는 데이터 반복 횟수, step은 파라미터 업데이트 횟수다. 이 모델에는 이미지 생성용 denoising inference steps가 적용되지 않는다.

## 평가와 실용적인 해석

공식 논문은 ImageNet 등 여러 이미지 데이터셋에서 zero-shot 분류와 linear probe를 평가한다. 전자는 클래스명을 문장으로 만들어 이미지와 비교하고, 후자는 고정 이미지 표현 위에 분류기를 학습한다. 두 방식은 학습 조건이 달라 점수를 섞어 비교할 수 없다. 논문에서 별도 표시가 없는 대표 CLIP 결과는 더 큰 ViT-L/14@336을 가리키므로 B/32 성능으로 인용하지 않았다. [공식 논문](https://arxiv.org/html/2103.00020v1)

실용적으로는 이미지 검색, 후보 라벨 기반 분류, 검색 시스템의 초기 후보 추출에 쓸 수 있다. 한국어 서비스를 만들 때는 영어 중심 평가·학습이라는 제약을 고려해 실제 한국어 질의로 검증해야 한다. 세밀한 구분과 개수 세기에도 한계가 있으며, 클래스 구성과 프롬프트에 따라 결과가 변한다. [공식 모델 카드의 제한 사항](https://github.com/openai/CLIP/blob/main/model-card.md)

이전 [MiniLM 분석](2026-10-06-all-minilm-l6-v2.md)은 문장끼리의 유사도에 초점을 두었다. CLIP은 이미지와 문장을 비교하는 공통 공간을 학습한다는 점이 다르다. 둘 다 벡터를 출력한다고 해서 서로의 벡터를 그대로 섞어 비교할 수는 없다. ViT-B/16처럼 패치를 더 작게 나누는 계열과 비교할 때도 토큰 수, 평가 조건과 비용을 함께 확인해야 한다.

이 글의 해석은 공개 구조와 학습 목적에 기반한다. 속도·VRAM·한국어 정확도는 실측하지 않았다.
