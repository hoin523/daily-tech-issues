# CogVideoX-2B: 30층 영상 Transformer와 3D VAE

조사일: 2026-10-22 KST. 모델 ID `zai-org/CogVideoX-2b`(이전 `THUDM/CogVideoX-2b`), revision `1137dacfc2c9c012bed6a0793f4ecf2ca8e7ba01`.

## 선정과 식별

Zhipu AI·Tsinghua 연구진의 텍스트→영상 생성 모델이다. 이전 THUDM URL은 현재 zai-org로 연결되며 동일 모델로 중복 판단한다. 모델·VAE 라이선스는 Apache-2.0이다. 정확한 최초 체크포인트 공개일은 확인 불가이고 논문 최초 공개일 2024-08-12와 구분한다. 별도 기반 체크포인트에서 fine-tuning됐는지와 그 revision은 확인 불가다. [고정 카드](https://huggingface.co/zai-org/CogVideoX-2b/blob/1137dacfc2c9c012bed6a0793f4ecf2ca8e7ba01/README.md), [논문](https://arxiv.org/abs/2408.06072)

조사 시점 최근 한 달 다운로드 15,849회·좋아요 380개였다. 좋아요 집계 기간은 확인 불가이며 인기 지표는 성능 증거가 아니다. [모델 페이지](https://huggingface.co/zai-org/CogVideoX-2b), [API](https://huggingface.co/api/models/THUDM/CogVideoX-2b)

모든 기존 models·issues를 읽어 동일 모델 분석이 없음을 확인했다. 기존 이미지 생성과 달리 시간축을 처리하는 영상 생성 분야를 선정했다. 큰 가중치 다운로드·추론·학습·유료 연산은 수행하지 않았다.

## 구조

표는 고정 revision의 [Transformer config](https://huggingface.co/zai-org/CogVideoX-2b/blob/1137dacfc2c9c012bed6a0793f4ecf2ca8e7ba01/transformer/config.json), [VAE config](https://huggingface.co/zai-org/CogVideoX-2b/blob/1137dacfc2c9c012bed6a0793f4ecf2ca8e7ba01/vae/config.json)에 근거한다.

| 항목 | 확인값 |
|---|---|
| 생성 backbone | CogVideoXTransformer3DModel |
| 규모 | 약 2B, Hub 표시; 텍스트 encoder·VAE 포함 전체 파이프라인 총량 아님 |
| layers | 30 |
| attention heads / head dimension | 30 / 64 |
| hidden 폭 | 30×64=1,920, 설정 기반 계산 |
| input / output latent channels | 16 / 16 |
| spatial patch size | 2 |
| text embedding dimension | 4,096 |
| max text sequence length | 226 |
| time embedding dimension | 512 |
| sample height / width | 60 / 90, latent 공간 설정 |
| sample frames | 49 |
| 위치 표현 관련 | use_rotary_positional_embeddings=false, spatial scale=1.875 |
| 활성화 | gelu-approximate, timestep에는 SiLU |
| MoE 전문가·활성량 | 토큰 top-k MoE 설정 없음; 별도 활성 파라미터 수 해당 없음 |
| VAE latent / RGB channels | 16 / 3 |
| VAE stage channels | [128,256,256,512] |
| VAE layers_per_block | 3 |
| VAE 시간 압축 비율 | 4 |
| VAE scaling factor | 1.15258426 |

논문 제목의 ‘Expert Transformer’는 모달별 adaptive LayerNorm을 다루며, LLM의 여러 FFN 전문가 중 일부를 라우팅하는 MoE와 동일시하면 안 된다. 영상 토큰은 공간과 시간의 관계를 함께 모델링한다. [논문 §2](https://arxiv.org/html/2408.06072v1#S2)

논문에는 3D-RoPE 설명이 있지만 이 **2B 고정 배포 설정은 rotary=false**다. 계열 설명을 모든 배포 변형에 그대로 적용하지 않는다. 정확한 위치 임베딩 구현의 세부는 config만으로 전부 확정하지 않았다.

3D VAE는 영상의 공간·시간을 압축하고 생성 Transformer는 텍스트 조건에 따라 잠재 표현의 노이즈를 줄인다. 이후 VAE decoder가 프레임으로 복원한다. Config의 latent 높이·폭 60×90과 VAE 출력 설정 480×720을 구분해야 한다. 시간 압축도 첫 프레임 처리 때문에 단순히 모든 길이를 4로 나눠 같은 길이라고 볼 수 없다.

카드는 영어 prompt, 약 6초·8fps·720×480 사용을 설명한다. `sample_frames=49`는 설정값이며 학습 49 epoch나 추론 반복 49회를 의미하지 않는다. 실제 내보낸 프레임 수·fps 처리도 파이프라인 설정을 함께 확인해야 한다. [고정 카드](https://huggingface.co/zai-org/CogVideoX-2b/blob/1137dacfc2c9c012bed6a0793f4ecf2ca8e7ba01/README.md)

## 학습과 공개 한계

논문은 progressive resolution 학습, 고품질 데이터 fine-tuning, 영상 필터링과 상세 캡션 생성 과정을 설명한다. [논문 §3](https://arxiv.org/html/2408.06072v1#S3)

| 항목 | 2B 체크포인트 확인 상태 |
|---|---|
| 사전학습·고품질 fine-tuning 단계 | 계열 방식은 확인, 배포본 실제 실행 로그 확인 불가 |
| epoch / optimizer steps | 확인 불가 |
| 실제 데이터·이미지·영상 소비량 | 확인 불가 |
| batch / gradient accumulation | 확인 불가 |
| learning rate / scheduler / optimizer | 확인 불가 |
| 학습 precision | 원본 학습 실행값 확인 불가 |
| 학습 하드웨어·시간 | 2B 전용 실행값 확인 불가 |
| DPO·RL·LoRA | 원본에 적용했다는 근거 확인 불가 |

카드의 FP16 권장 추론과 후속 LoRA·SFT 메모리 예시는 원본 사전학습 precision·batch를 뜻하지 않는다. 공식 [CogVideo 코드 저장소](https://github.com/zai-org/CogVideo)의 사용·개발 경로는 확인했지만 원본의 완전한 학습 레시피·실행 로그를 검증한 것은 아니다.

Epoch는 데이터 반복, optimizer step은 학습 업데이트다. 카드의 50 sampling steps는 생성 중 노이즈를 줄이는 반복으로 또 다르다. 30 layers도 구조 깊이이므로 학습량을 대신하지 않는다. 확인 못한 값은 임의로 보충하지 않았다.

## 평가·비교·활용

보고서는 자동 지표와 사람 평가로 프롬프트 일치·영상 품질·동작 등을 다룬다. 이번 조사에서는 2B 배포본에 대응하는 전체 평가 prompt·seed·sampler 조건을 대조하지 못해 계열 최고 점수나 순위를 이 모델의 성능으로 쓰지 않는다. [논문 평가](https://arxiv.org/html/2408.06072v1#S4)

실제 비교는 같은 prompt·해상도·길이·fps·추론 반복·생성 수를 맞추고 프레임 품질과 시간 일관성을 나눠 평가해야 한다. 카드의 A100/H100 속도와 최소 VRAM 문구는 offload·라이브러리·정밀도 조건이 있어 모든 환경의 보장값이 아니다.

기존 [FLUX.1-schnell 분석](2026-10-16-flux-1-schnell.md)은 빠른 정지 이미지 생성이다. CogVideoX는 시간축과 3D VAE를 포함하므로 모델 크기·sampling steps만으로 생성 비용이나 품질을 비교할 수 없다.

영상 시안과 동작 아이디어 실험에 검토할 수 있다. 객체 정체성·물리 동작·세부 관계의 일관성은 별도 확인이 필요하다. 이 텍스트→영상 배포본을 이미지→영상 변형과 혼동하지 않는다.

2B backbone 가중치만 FP16으로 둔다고 가정하면 `2×10⁹×2≈4 GB`(십진)다. 반올림 규모에 기반한 계산이며 전체 파이프라인 최소 VRAM·실측이 아니다. 텍스트 encoder·VAE·activation·프레임 수·batch·실행 버퍼가 추가된다.
