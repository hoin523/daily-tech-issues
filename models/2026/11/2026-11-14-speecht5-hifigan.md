# SpeechT5 HiFi-GAN: Mel 프레임을 256배 확대해 만드는 16kHz 파형

조사일: 2026-11-14 (한국 시간). 공개 구성·논문·코드를 분석했으며 가중치 다운로드·학습·추론은 수행하지 않았다.

## 선정과 식별

Hugging Face API에서 최근 한 달 다운로드 **209,584회**, 누적 좋아요 **29개**를 확인했다. 이미지 분류에 이어 오디오 파형 합성 구성 요소를 선택했다. 사용 관심도이지 음질이나 트렌딩 순위의 증거는 아니다. [API](https://huggingface.co/api/models/microsoft/speecht5_hifigan), [모델 페이지](https://huggingface.co/microsoft/speecht5_hifigan)

| 항목 | 확인 내용 |
|---|---|
| 모델 / 배포 조직 | `microsoft/speecht5_hifigan` / Microsoft |
| revision | `bb6f429406e86a9992357a972c0698b22043307d` |
| 용도 / 기반 | SpeechT5 TTS·voice conversion과 함께 쓰는 HiFi-GAN vocoder. Mel 스펙트로그램에서 파형 생성. |
| 발표 | HiFi-GAN 원 논문 2020-10-12, SpeechT5 논문 v1 2021-10-14. 이 변환 가중치 최초 배포일은 확인 불가. |
| 라이선스 | 고정 카드 MIT |
| 카드 작성 | Hugging Face 팀 작성본이라고 명시 |

근거: [고정 카드](https://huggingface.co/microsoft/speecht5_hifigan/blob/bb6f429406e86a9992357a972c0698b22043307d/README.md), [HiFi-GAN 논문](https://arxiv.org/abs/2010.05646), [SpeechT5 논문](https://arxiv.org/html/2110.07205v1).

기존 `models`·`issues` 전체를 읽었다. [SpeechT5 TTS 분석](../10/2026-10-19-speecht5-tts.md)은 이 vocoder를 연결한다고 언급했지만 내부 구조를 분석하지 않았다. Mel을 만드는 TTS 본체와 파형을 만드는 별도 모델이며, 양자화·미러를 새 모델로 취급한 것이 아니다.

## 구조와 파형 길이

| 항목 | 확인값 |
|---|---|
| 입력 | log-Mel 80차원, frame별 벡터 |
| 출력 | 16,000Hz mono 파형 |
| 입력 정규화 | `normalize_before=true`, 저장된 mean·scale로 정규화 |
| 첫 convolution | 80→512채널, kernel 7 |
| upsampling | ConvTranspose1d 4단계, stride `[4,4,4,4]`, kernel `[8,8,8,8]` |
| 단계별 채널 | 512→256→128→64→32, 코드에서 유도 |
| multi-receptive-field 경로 | 단계마다 kernel `[3,7,11]`의 residual block 3개를 병렬 적용 후 평균 |
| residual dilation | 각 kernel 경로에 `[1,3,5]` |
| residual block 총수 | `4단계×3경로=12`, 계산값 |
| 활성화 | LeakyReLU, 설정 slope 0.1. 마지막은 tanh |
| 마지막 convolution | 32→1채널, kernel 7 |
| attention·KV·position encoding·MoE | 해당 없음. Transformer·VAE가 아닌 convolutional vocoder. |
| 총·활성 파라미터 | 이번 조사에서 직접 집계하지 못해 확인 불가. MoE 활성 계수 구분은 없음. |

근거: [고정 config](https://huggingface.co/microsoft/speecht5_hifigan/blob/bb6f429406e86a9992357a972c0698b22043307d/config.json), [공식 Transformers 구현](https://github.com/huggingface/transformers/blob/v4.27.0/src/transformers/models/speecht5/modeling_speecht5.py). 라이브러리 v4.27.0 태그는 구조 확인용이며 원 학습 commit이 아니다.

Mel 스펙트로그램은 시간별 주파수 대역의 크기를 요약한다. vocoder는 이를 촘촘한 파형으로 복원한다. 각 단계는 시간 해상도를 4배로 올리고, 서로 다른 kernel 경로가 다른 범위의 패턴을 본다. 세 경로를 직렬로 이어붙이는 것이 아니라 같은 입력에 적용한 출력을 평균 낸다. [forward 구현](https://github.com/huggingface/transformers/blob/v4.27.0/src/transformers/models/speecht5/modeling_speecht5.py)

총 확대율은 `4⁴=256`이다. 구현의 kernel·padding·stride를 그대로 적용한 T개 입력 프레임은 256T개 sample로 출력된다. 따라서 예를 들어 100프레임은 25,600sample, 16kHz에서 **1.6초**라는 계산이 된다. 이는 생성 시간이나 속도 실측이 아니라 파형 길이 계산이다. 모델 입력의 frame 간격도 이 대응과 맞아야 한다. [구현](https://github.com/huggingface/transformers/blob/v4.27.0/src/transformers/models/speecht5/modeling_speecht5.py)

16kHz를 16,000개 Mel bin이나 모델 layer 수로 해석하면 안 된다. 또한 4개 upsampling stage는 diffusion inference step이나 학습 epoch와 다르다.

## 학습 이력의 공개 범위

| 항목 | 해당 vocoder에 대해 확인한 내용 |
|---|---|
| 데이터 | SpeechT5 논문은 LibriTTS 학습 데이터로 speaker-independent HiFi-GAN을 학습했다고 설명 |
| 데이터 시간·정확한 split | 이 HF revision 대응 집계 확인 불가 |
| epoch / optimizer step | 고정 카드에서 미공개, 해당 실행 로그 확인 불가 |
| batch / accumulation | 확인 불가 |
| optimizer / LR / scheduler | 확인 불가 |
| precision / 학습 장비 | 확인 불가 |
| SFT·DPO·RL·LoRA | 이 vocoder에 별도 적용한 단계로 보고되지 않음 |

근거: [고정 카드](https://huggingface.co/microsoft/speecht5_hifigan/blob/bb6f429406e86a9992357a972c0698b22043307d/README.md), [SpeechT5 논문 §3.2](https://arxiv.org/html/2110.07205v1). 공식 [SpeechT5 저장소](https://github.com/microsoft/SpeechT5)도 확인했으나 이 변환본의 vocoder 실행 설정을 특정하지 못했다.

SpeechT5 논문의 **TTS 300k step·LR 0.0004**는 Mel을 예측하는 TTS 본체 설명이다. 같은 문단에 HiFi-GAN이 등장한다고 그 수치를 vocoder에 복사하지 않았다. HiFi-GAN 원 논문의 다른 sample rate·dataset·변형 설정도 이 16kHz 모델의 학습값으로 사용하지 않았다. `torch_dtype=float32`는 저장·로드 구성이지 원 학습 precision의 증거가 아니다. [논문 §3.2](https://arxiv.org/html/2110.07205v1), [config](https://huggingface.co/microsoft/speecht5_hifigan/blob/bb6f429406e86a9992357a972c0698b22043307d/config.json)

HiFi-GAN 연구는 생성기와 판별기의 적대적 학습으로 파형 품질을 개선한다. 배포된 vocoder의 forward에 판별기가 없다고 원 학습에 판별기가 없었다고 결론내릴 수는 없다. [원 논문](https://arxiv.org/abs/2010.05646), [배포 구현](https://github.com/huggingface/transformers/blob/v4.27.0/src/transformers/models/speecht5/modeling_speecht5.py)

## 평가와 활용 범위

HiFi-GAN 원 논문의 MOS·속도는 원 실험의 화자·22.05kHz·장치 조건에 해당하며 이 16kHz HF 가중치의 성능이 아니다. SpeechT5의 TTS MOS·WER도 텍스트 모델·화자 조건·vocoder를 포함한 파이프라인 평가다. vocoder 단독 점수로 옮기지 않았다. [HiFi-GAN 원 논문](https://arxiv.org/abs/2010.05646), [SpeechT5 Table 3](https://arxiv.org/html/2110.07205v1)

호환되는 SpeechT5 Mel 출력을 청취 가능한 파형으로 바꾸는 데 쓸 수 있다. 임의의 80차원 행렬이면 모두 호환되는 것은 아니며 sample rate·정규화·Mel 추출 조건이 맞아야 한다는 것이 구조에서 도출한 해석이다. 이 모델 자체는 텍스트를 읽거나 음성을 전사하지 않는다. [고정 카드](https://huggingface.co/microsoft/speecht5_hifigan/blob/bb6f429406e86a9992357a972c0698b22043307d/README.md), [입력 정규화 구현](https://github.com/huggingface/transformers/blob/v4.27.0/src/transformers/models/speecht5/modeling_speecht5.py)

[EnCodec 24kHz](2026-11-08-encodec-24khz.md)는 파형을 코드북으로 압축하고 복원한다. 오늘 vocoder는 연속적인 Mel 표현을 입력으로 받으므로 RVQ 코드북을 대신 입력할 수 없다. [SpeechT5 TTS](../10/2026-10-19-speecht5-tts.md)와의 연결에서는 본체와 vocoder의 비용을 따로 구분해야 한다.

출처·값·기존 글과의 차이를 검토했다. VRAM·속도·음질은 실측하지 않았으며 공개되지 않은 학습값은 구조에서 추정하지 않았다.
