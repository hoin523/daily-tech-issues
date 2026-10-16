# FLUX.1-schnell: 19+38 블록의 빠른 이미지 생성 Transformer

조사일: 2026-10-16 KST. ID `black-forest-labs/FLUX.1-schnell`. Hub API revision `741f7c3ce8b383c54771c7003378a50191e9efe9`.

## 선정과 자료 범위

Black Forest Labs의 텍스트 조건 이미지 생성 모델이다. FLUX.1 계열 공개 발표는 2024-08-01이며 schnell은 latent adversarial diffusion distillation을 적용한 변형이다. 정확한 교사 체크포인트 ID는 확인 불가다. 공개 카드 라이선스는 Apache-2.0이다. [공식 발표](https://bfl.ai/blog/24-08-01-bfl), [모델 페이지](https://huggingface.co/black-forest-labs/FLUX.1-schnell)

조사 시점 최근 한 달 다운로드 571,276회·좋아요 6,286개다. 좋아요 집계 기간은 확인 불가이며 인기와 품질은 별개다. [API](https://huggingface.co/api/models/black-forest-labs/FLUX.1-schnell)

기존 models·issues 전체를 읽어 같은 모델 분석이 없음을 확인했다. SDXL 이후 다른 조직·다른 생성 구조를 선정했다. Hub의 README·transformer/config.json·model_index.json 원문 조회는 401로 실패했다. 제한된 파일을 확인했다고 주장하지 않으며, 공개 웹 카드와 공식 GitHub 모델별 설정을 사용했다. 가중치 다운로드·학습·추론은 수행하지 않았다.

## 구조

다음 수치는 [공식 코드의 flux-schnell ModelSpec](https://github.com/black-forest-labs/flux/blob/802fb4713906133fcbd0d8dc5351620ca4773036/src/flux/util.py)에 근거한다. 코드 commit은 `802fb4713906133fcbd0d8dc5351620ca4773036`이다. 이는 공개 참조 구현의 설정이며 접근하지 못한 HF revision 파일과 직접 대조한 값은 아니다.

| 항목 | 공개 근거에서 확인한 값 |
|---|---|
| 유형 | rectified flow Transformer |
| 파라미터 | 공개 카드 약 12B, 생성 Transformer 기준; 텍스트 encoder·AE 포함 전체 파이프라인 총량으로 해석하지 않음 |
| 활성 파라미터 | MoE식 전문가 선택 구분 없음 |
| hidden size | 3,072 |
| attention heads | 24 |
| head dimension | 3,072÷24=128, 균등 분할 계산 |
| double-stream depth | 19 |
| single-stream depth | 38 |
| MLP ratio | 4.0; 계산상 3,072×4=12,288 |
| 입력·출력 채널 | 64 / 64, packed latent 토큰 표현 |
| 텍스트 context 입력 차원 | 4,096 |
| 벡터 조건 입력 차원 | 768 |
| 위치 표현 설정 | axes_dim=[16,56,56], theta=10,000 |
| QKV bias / guidance embedding | true / false |
| 별도 GQA·MoE 설정 | 없음 |

19+38=57개 블록이라는 합산은 가능하지만 서로 다른 블록 유형이다. 이를 같은 종류의 LLM 레이어 57개로 단순 비교하지 않는다. 이미지와 텍스트를 별도 경로에서 다루다가 함께 처리하는 생성 backbone이며, 이미지 잠재 표현을 반복적으로 갱신한다. 일반 텍스트 decoder의 다음 단어 예측과 목적이 다르다.

공식 AE 설정은 latent 16채널, 기본 채널 128, 채널 배수 [1,2,4,4], stage당 residual block 설정 2, scale_factor=0.3611·shift_factor=0.1159다. AE 설정의 resolution=256을 유일한 출력 해상도라고 읽지 않는다. [공식 모델·AE 설정](https://github.com/black-forest-labs/flux/blob/802fb4713906133fcbd0d8dc5351620ca4773036/src/flux/util.py)

공개 카드의 Diffusers 예제는 max_sequence_length=256·guidance_scale=0·num_inference_steps=4를 사용한다. 256은 그 예제의 텍스트 입력 길이 설정이며 hidden 차원 4,096이나 이미지 해상도와 다르다. 정확한 고정 revision의 텍스트 encoder 구성 파일은 접근하지 못해 세부 층수는 확인 불가다. [공개 사용 예제](https://huggingface.co/black-forest-labs/FLUX.1-schnell)

## 학습과 증류: 4 steps는 epoch가 아니다

| 항목 | 확인 상태 |
|---|---|
| 기반 생성 방식 | rectified flow |
| schnell 후처리 방식 | latent adversarial diffusion distillation |
| 사전학습·증류 epoch | 확인 불가 |
| optimizer steps | 확인 불가 |
| 데이터·토큰·이미지 규모 | 확인 불가 |
| batch / gradient accumulation | 확인 불가 |
| learning rate / scheduler / optimizer | 확인 불가 |
| 실제 학습 precision | 확인 불가 |
| 학습 하드웨어·시간 | 확인 불가 |
| 별도 SFT·DPO·RL·LoRA | 이 원본에 적용했다는 근거 확인 불가 |

근거는 [공개 카드](https://huggingface.co/black-forest-labs/FLUX.1-schnell)다. 이번 조사에서 체크포인트에 연결되는 완전한 학습 설정·로그를 확인하지 못했다. 관련 증류 논문의 일반 실험 레시피를 그대로 이 모델의 실제 실행값으로 옮기지 않았다.

카드의 1~4 steps는 이미지 한 장을 생성할 때의 추론 반복이다. 학습 데이터 반복인 epoch나 optimizer update 수와 다르다. 19·38은 구조 블록 수다. 빠른 생성용 증류는 반복 횟수를 줄이는 목적이지만 같은 해상도·장치에서 실제 지연 시간을 측정한 것은 아니다.

## 평가·비교·활용

공식 발표와 카드는 품질·프롬프트 반영 성능을 설명하지만 이번 조사에서 평가별 프롬프트 집합·표본 수·동일 생성 예산을 충분히 대조하지 못했다. 따라서 홍보 문구를 확정적인 순위나 실측으로 재기록하지 않는다. 비교 시 prompt·seed·해상도·추론 단계·후처리와 평가 방식을 함께 맞춰야 한다. [공식 발표](https://bfl.ai/blog/24-08-01-bfl)

기존 [SDXL 분석](2026-10-08-sdxl-base-1-0.md)은 U-Net 중심 latent diffusion, 이번 모델은 flow Transformer 중심이다. 서로 다른 backbone과 증류 조건 때문에 단순 블록 수·step 수만으로 품질이나 계산량 우열을 정할 수 없다.

이미지 시안과 반복적인 프롬프트 탐색에 검토할 수 있다. 생성 그림은 사실을 증명하는 자료가 아니며 문구·세부 관계·프롬프트 일치 여부를 확인해야 한다. 카드도 프롬프트 스타일에 따른 결과 차이와 편향 가능성을 설명한다. [모델 한계](https://huggingface.co/black-forest-labs/FLUX.1-schnell)

12B Transformer 가중치만 BF16으로 둔다고 가정하면 `12×10⁹×2=24 GB`(십진)다. 반올림 파라미터에 기초한 계산이며 전체 파이프라인 최소 VRAM이 아니다. 텍스트 encoder·AE·activation·실행 버퍼·offload 및 양자화에 따라 메모리가 달라진다. 실측은 수행하지 않았다.
