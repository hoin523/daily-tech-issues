# LDM text2im large 256: 픽셀 대신 작은 잠재 공간에서 이미지를 그리기

조사일: 2026-11-24 (한국 시간). 가중치 다운로드·추론·학습 없이 공개 구성과 원본 설정을 분석했다.

## 선정과 식별

Hub API에서 최근 한 달 다운로드 **728회**, 누적 좋아요 **37개**를 확인했다. 오늘날 최상위 인기라고 주장할 수는 없으나 사용 기록이 있고 원본 구조를 추적할 수 있는 역사적 이미지 생성 모델로 선정했다. 이전 영어 임베딩과 다른 분야·개발 조직이다. 기존 models·issues 53개 기록 전체를 읽고 동일 모델·revision·미러·양자화 분석이 없음을 확인했다. [API](https://huggingface.co/api/models/CompVis/ldm-text2im-large-256)

| 항목 | 내용 |
|---|---|
| ID / 개발 | `CompVis/ldm-text2im-large-256` / CompVis 연구팀 |
| revision | `30de525ca11a880baea4962827fb6cb0bb268955` |
| 기반·용도 | 원본 text2img LDM의 Diffusers 변환본, 256×256 텍스트 조건 이미지 생성 |
| 발표 / 조사 | 관련 논문 v1 2021-12-20 / 조사 2026-11-24. 해당 HF 변환본 최초 공개일 확인 불가 |
| 라이선스 | HF 카드 Apache-2.0 |

[고정 카드](https://huggingface.co/CompVis/ldm-text2im-large-256/blob/30de525ca11a880baea4962827fb6cb0bb268955/README.md), [원 논문](https://arxiv.org/abs/2112.10752). Stable Diffusion이나 SDXL의 가중치와 동일 모델로 취급하지 않는다.

## 세 구성 요소의 구조

| 구성 | 확인값 |
|---|---|
| 전체 파라미터 | 원본 설정 파일명은 1p4B이나, 이 HF revision의 정확한 총·trainable 계수는 확인 불가. 파일명만으로 정확한 합계를 확정하지 않음 |
| U-Net 입력·출력 | latent 4채널, sample_size 32 |
| U-Net 채널 단계 | [320,640,1280,1280], down 4개·up 4개 |
| down 구조 | CrossAttnDown 3개 + Down 1개, layers_per_block 2 |
| up 구조 | Up 1개 + CrossAttnUp 3개. down과 같은 residual 개수라고 단순 합산하지 않음 |
| cross-attention | 원본 context_dim 1,280, num_heads 8, spatial transformer depth 1 |
| 텍스트 encoder | LDMBert, 32층, d_model 1,280, FFN 5,120, GELU |
| 텍스트 attention | encoder_attention_heads 8, 별도 head_dim 64. hidden÷heads=160을 실제 head dimension으로 덮어쓰지 않음 |
| 텍스트 입력 | vocab 30,522, 최대 위치 77 |
| autoencoder | AutoencoderKL, RGB↔4채널 latent, 채널 [128,256,512,512], 양쪽 4개 단계·layers_per_block 2 |
| scheduler | DDIM, num_train_timesteps 1,000, linear beta 0.00085→0.012 |
| MoE | 없음. LLM 활성 전문가 수는 해당하지 않음 |

[U-Net config](https://huggingface.co/CompVis/ldm-text2im-large-256/blob/30de525ca11a880baea4962827fb6cb0bb268955/unet/config.json), [텍스트 config](https://huggingface.co/CompVis/ldm-text2im-large-256/blob/30de525ca11a880baea4962827fb6cb0bb268955/bert/config.json), [autoencoder config](https://huggingface.co/CompVis/ldm-text2im-large-256/blob/30de525ca11a880baea4962827fb6cb0bb268955/vqvae/config.json), [scheduler config](https://huggingface.co/CompVis/ldm-text2im-large-256/blob/30de525ca11a880baea4962827fb6cb0bb268955/scheduler/scheduler_config.json), [원본 평가 YAML](https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/configs/latent-diffusion/txt2img-1p4B-eval.yaml). 원본 YAML과 오래된 Diffusers 필드의 의미가 완전히 같다고 가정하지 않았다. 특히 `attention_head_dim=8`이라는 필드명만 보고 원본 U-Net의 head dimension을 8이라고 단정하지 않는다.

256×256 이미지를 latent 32×32로 표현하면 공간 위치 수는 `(256÷32)²=64`배 감소한다. 이는 픽셀 위치 수 비교이며 전체 연산이나 실행 시간이 정확히 64배 줄어든다는 뜻은 아니다. autoencoder가 작은 표현과 이미지를 연결하고 U-Net이 그 표현의 노이즈를 여러 번 제거한다. 텍스트 encoder의 벡터는 cross-attention으로 이미지 생성 과정에 들어간다. [구성 파일](https://huggingface.co/CompVis/ldm-text2im-large-256/blob/30de525ca11a880baea4962827fb6cb0bb268955/vqvae/config.json), [원본 설정](https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/configs/latent-diffusion/txt2img-1p4B-eval.yaml), [논문](https://arxiv.org/abs/2112.10752)

폴더 이름은 `vqvae`지만 실제 class는 **AutoencoderKL**이다. 따라서 이 변환본을 이산 VQ 코드북 모델이라고 설명하지 않았다. 텍스트 77 position도 학습 epoch나 이미지 해상도가 아니다. [고정 autoencoder](https://huggingface.co/CompVis/ldm-text2im-large-256/blob/30de525ca11a880baea4962827fb6cb0bb268955/vqvae/config.json), [텍스트 설정](https://huggingface.co/CompVis/ldm-text2im-large-256/blob/30de525ca11a880baea4962827fb6cb0bb268955/bert/config.json)

## 학습과 추론 설정 구분

| 항목 | 공개 근거와 한계 |
|---|---|
| 학습 단계 | 이미지 압축 표현 학습과 latent diffusion 학습을 구분. 개별 변환본의 단계별 실행 로그 확인 불가 |
| epoch / optimizer updates | 명시값 확인 불가 |
| 데이터·이미지 수 | 이 revision과 일대일 대응하는 정확한 데이터 버전·소비 이미지 수 확인 불가 |
| batch / gradient accumulation | 실제 실행값 확인 불가 |
| learning rate | 원본 eval YAML에 base_learning_rate 5e-5. effective LR·학습 실행값으로 확정하지 않음 |
| scheduler / optimizer | 실제 학습 LR scheduler·optimizer 상세 확인 불가. DDIM은 여기서 생성 scheduler |
| precision / 장비 | 텍스트 config는 float32, 전체 실제 학습 precision·GPU 종류·수 확인 불가 |
| text encoder | 원본 YAML cond_stage_trainable true. CLIP을 고정해 쓰는 SDXL과 동일 recipe가 아님 |
| SFT·DPO·RL·LoRA | 이 원본 체크포인트에 적용된 해당 후속 단계 미공개 |

[원본 평가 YAML](https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/configs/latent-diffusion/txt2img-1p4B-eval.yaml), [고정 카드](https://huggingface.co/CompVis/ldm-text2im-large-256/blob/30de525ca11a880baea4962827fb6cb0bb268955/README.md), [관련 논문](https://arxiv.org/abs/2112.10752). 평가용 설정을 완전한 학습 재현 recipe나 학습 로그로 바꾸지 않았다.

`num_train_timesteps=1000`은 노이즈 스케줄의 시간 인덱스 개수다. 학습 optimizer update가 1,000회라는 뜻이 아니다. 카드 예제의 `num_inference_steps=50`, eta 0.3, guidance_scale 6은 **이미지 생성 예시 설정**이다. 50 epoch 학습이라고 쓰거나 각 설정을 실측 품질 최적값이라고 주장하지 않는다. [scheduler](https://huggingface.co/CompVis/ldm-text2im-large-256/blob/30de525ca11a880baea4962827fb6cb0bb268955/scheduler/scheduler_config.json), [사용 예제](https://huggingface.co/CompVis/ldm-text2im-large-256/blob/30de525ca11a880baea4962827fb6cb0bb268955/README.md)

## 평가·비교·활용의 한계

논문은 text-to-image 외에 unconditional generation·inpainting·super-resolution도 평가한다. 다른 task나 압축률·sampling 조건의 FID를 오늘 변환본의 성능으로 옮기면 안 된다. 카드에는 생성 예제와 샘플 이미지가 있지만 이 revision의 고정 평가 데이터·sampling·sample 수가 대응되는 정량 표는 확인하지 못했다. 따라서 FID·CLIP score·속도는 확인 불가로 남긴다. 샘플 몇 장은 정량 평가나 재현 증거를 대신하지 못한다. [논문](https://arxiv.org/abs/2112.10752), [카드](https://huggingface.co/CompVis/ldm-text2im-large-256/blob/30de525ca11a880baea4962827fb6cb0bb268955/README.md)

기존 [SDXL base](../10/2026-10-08-sdxl-base-1-0.md)는 두 텍스트 encoder를 사용하는 후대의 별도 모델이다. 오늘 모델은 LDMBert 텍스트 조건과 256 해상도 구성을 읽는 사례로 활용할 수 있다. 기존 [SD-Turbo](2026-11-16-sd-turbo.md)의 1회 생성 설정을 이 모델에 적용 가능한 학습 특성으로 옮기지 않는다.

분석상 latent diffusion·cross-attention·autoencoder 역할을 학습하는 기준 사례에 적합하다. 최신 이미지 생성 품질을 대표한다고 보기는 어렵고 문구 표현·세부 묘사·사용자 prompt 적합성은 별도 검증이 필요하다. VRAM·처리량·사용자 데이터 품질은 실측하지 않았다.
