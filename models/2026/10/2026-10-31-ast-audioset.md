# AST AudioSet: 12층 Transformer로 소리의 종류를 분류하기

조사일: 2026-10-31 KST. 가중치 다운로드·추론·학습 없이 공개 설정·논문·코드를 분석했다.

## 선정과 식별

모델 ID는 `MIT/ast-finetuned-audioset-10-10-0.4593`, revision은 `f826b80d28226b62986cc218e5cec390b1096902`다. [Hub API](https://huggingface.co/api/models/MIT/ast-finetuned-audioset-10-10-0.4593)는 다운로드 1,494,685·좋아요 388을 반환했고 [모델 페이지](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593)는 다운로드 기간을 last month로 표시했다. 인기도는 성능이나 고유 사용자 수가 아니다. 기존 models·issues 전체 파일을 읽어 AST 계열 분석이 없음을 확인했다. 최근 이미지 모델에 이어 새로운 개발 조직 MIT의 오디오 이벤트 분류 모델을 선정했다.

연구진은 MIT의 Yuan Gong·Yu-An Chung·James Glass다. [논문](https://arxiv.org/abs/2104.01778)은 2021-04-05 최초 제출됐으며 HF 변환 체크포인트의 정확한 배포일은 이번 조사에서 확인 불가다. Hub 라이선스는 BSD-3-Clause다. 카드는 Hugging Face 팀 작성본임을 명시해 개발팀 논문·원 코드와 구분했다. [고정 카드](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593/blob/f826b80d28226b62986cc218e5cec390b1096902/README.md).

기반은 ImageNet에서 학습한 distilled DeiT이며, 원 구현은 `vit_deit_base_distilled_patch16_384`를 오디오로 전이한다. RGB patch projection을 단일 채널에 맞추고 위치 임베딩을 crop·보간하며 오디오 클래스 head를 새로 둔다. 이미지로 분류하던 표현을 소리의 시간·주파수 표현에 맞춰 재사용하는 방식이다. [논문 §2.2](https://arxiv.org/html/2104.01778v2#S2.SS2), [공식 모델 코드](https://github.com/YuanGongND/ast/blob/31088be8a3f6ef96416145c4b8d43c81f99eba7a/src/models/ast_models.py).

## 구조: 음성 decoder가 없는 encoder

아래는 [고정 config](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593/blob/f826b80d28226b62986cc218e5cec390b1096902/config.json), [고정 전처리](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593/blob/f826b80d28226b62986cc218e5cec390b1096902/preprocessor_config.json)와 공식 모델 코드를 대조한 값이다.

| 항목 | 값·의미 |
|---|---|
| 구조 | ASTForAudioClassification, dense Transformer encoder |
| 총 파라미터 | 86,594,063, Hub safetensors 메타데이터 |
| 레이어 / hidden dimension | 12 / 768 |
| FFN intermediate / activation | 3,072 / GELU |
| attention heads | 12, 표준 multi-head self-attention |
| head dimension | 768÷12=64, 균등 분할 계산 |
| 입력 | 16,000Hz 파형 → 128 Mel bins, 최대 1,024 frames |
| patch / stride | 16×16 / 주파수·시간 각각 10 |
| 겹침 | 한 축에서 16−10=6, 설정 기반 계산 |
| 위치 표현 | 학습 가능한 위치 임베딩 |
| special tokens·pooling | CLS와 distillation token, 두 출력의 평균 |
| head | 정규화 후 527 AudioSet 라벨 점수 |
| 정규화 / QKV bias | LayerNorm eps=1e-12 / true |
| MoE·KV head 축소 | 해당 없음 |
| 저장 dtype | float32, 학습 정밀도와 구분 |

1,024 frames를 기준으로 유효 patch 수는 `floor((128−16)/10)+1=12`와 `floor((1024−16)/10)+1=101`의 곱인 **1,212**개다. 두 special token을 더하면 **1,214** 토큰이다. 이는 padding·crop된 고정 특징 입력을 가정한 계산이며 학습 step 수나 1,214초 오디오가 아니다. [config](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593/blob/f826b80d28226b62986cc218e5cec390b1096902/config.json), [원 구현](https://github.com/YuanGongND/ast/blob/31088be8a3f6ef96416145c4b8d43c81f99eba7a/src/models/ast_models.py).

논문 초기 설명은 단일 CLS 표현을 쓰는 형태로 서술하지만 조회한 공개 구현은 두 토큰을 유지하고 마지막에 평균한다. 코드의 실제 경로를 기준으로 구분했다. 또한 논문의 convolution-free라는 말은 깊은 CNN backbone 없이 attention encoder를 사용한다는 뜻이다. Patch의 선형 투영을 Conv2d로 구현할 수 있다는 사실과 모순되지 않는다. [논문 §2](https://arxiv.org/html/2104.01778v2#S2), [코드](https://github.com/YuanGongND/ast/blob/31088be8a3f6ef96416145c4b8d43c81f99eba7a/src/models/ast_models.py).

## 소리가 입력에서 라벨까지 가는 과정

파형은 시간 간격별 log-Mel filterbank로 바뀐다. 이를 작은 사각형 patch로 나누면 각 토큰은 특정 시점·주파수 구간의 특징을 담는다. Encoder의 전역 attention은 멀리 떨어진 구간도 참조한다. 마지막 표현으로 말소리·개 짖음·음악 같은 라벨의 점수를 계산한다. 이 모델은 말을 문자로 전사하거나 음악을 생성하지 않는다.

AudioSet은 여러 소리가 한 clip에 함께 있을 수 있는 **multi-label** 문제다. 학습은 BCE를 사용하며 sigmoid 점수는 클래스 간 합이 1이 되는 softmax와 다르다. 가장 큰 라벨 하나만 반환하는 UI를 만들 수는 있지만 원 과제를 단일 정답 분류로 바꾸어 해석하면 안 된다. Threshold는 실제 데이터에서 정해야 한다. [논문 §3.1](https://arxiv.org/html/2104.01778v2#S3.SS1).

[전처리 설정](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593/blob/f826b80d28226b62986cc218e5cec390b1096902/preprocessor_config.json)의 mean=-4.2677393, std=4.5689974는 데이터 통계다. 논문은 입력을 평균 0·표준편차 0.5가 되도록 정규화한다고 설명한다. 설정의 std만 보고 목표 표준편차가 1이라고 단정하지 않았다. 긴 녹음은 frame 제한과 truncation·분할 전략을 확인해야 하며, 10초 단위 평가가 긴 파일의 모든 사건 위치를 찾아준다는 뜻은 아니다.

## 학습: ImageNet 사전학습과 AudioSet fine-tuning

[논문 §3.1](https://arxiv.org/html/2104.01778v2#S3.SS1)과 [고정 AudioSet recipe](https://github.com/YuanGongND/ast/blob/31088be8a3f6ef96416145c4b8d43c81f99eba7a/egs/audioset/run.sh)를 대조했다. 공개 recipe에는 full과 balanced 분기가 있어 두 설정을 섞으면 안 된다.

| 항목 | full AudioSet | balanced AudioSet |
|---|---|---|
| 데이터 규모, 논문 | 약 2M clips | 약 22K clips |
| clip 길이 | 약 10초 | 약 10초 |
| epoch | 5 | 25 |
| batch size | 12 | 12 |
| optimizer / loss | Adam / BCE | Adam / BCE |
| 초기 learning rate | 1e-5 | 5e-5 |
| scheduler | 2번째 epoch 이후 매 epoch 0.5배 감소 | 10번째 epoch 이후 5 epoch마다 0.5배 감소 |
| augmentation | mixup 0.5, frequency mask 최대48·time mask 최대192 | 동일 |
| sampling | class-balanced sampling | 원 recipe bal=none |
| weight averaging | epoch 1~5 | epoch 6~25 |

Full recipe는 warmup=True도 지정하지만 그 파일만으로 warmup 업데이트 수를 확정하지 않았다. 표의 batch가 실제 분산 구성에서 어떤 단위인지와 해당 HF revision의 정확한 step·gradient accumulation·optimizer 세부 beta·precision·GPU 구성은 이번 조사에서 **확인 불가**다. 스크립트의 Slurm GPU 4개 요청은 공개 실행 예시이지 이 HF 체크포인트가 실제로 그 자원에서 만들어졌다는 증거가 아니다. 별도 SFT·DPO·RL·LoRA 적용 근거도 확인 불가다.

ImageNet 기반 DeiT의 사전학습 epoch·step·배치·LR·scheduler·optimizer·정밀도·하드웨어 전체 이력은 이번에 checkpoint 단위로 검증하지 못해 확인 불가로 남긴다. AudioSet의 5 epochs를 처음부터 모델 전체를 학습한 총량으로 표현하지 않았다. Balanced sampling은 예제의 노출 횟수를 바꿀 수 있으므로 epoch를 모든 clip이 정확히 한 번씩 나왔다는 보장으로 취급하지 않는다.

모델 이름의 `10-10`은 시간·주파수 stride이고 `0.4593`은 공개 모델의 성능 표기이며 학습 epoch나 step이 아니다. Weight averaging은 여러 checkpoint의 **가중치 평균**으로 한 모델을 만드는 것, ensemble은 여러 모델의 **출력 평균**이므로 실행 비용이 다르다. [논문 §3.1.2](https://arxiv.org/html/2104.01778v2#S3.SS1.SSS2).

## 평가와 사용 판단

논문은 AudioSet evaluation 약 20K clips에서 mean average precision(mAP)을 사용한다. Full-set single weight-averaged 모델은 0.459±0.000, 마지막 epoch 모델은 0.448±0.001, 다양한 설정 6개 모델 ensemble은 0.485다. 초록의 0.485를 이 단일 HF 모델의 실측으로 옮기지 않았다. 이름의 0.4593과 논문의 반올림 수치를 구분하며 HF revision의 평가를 이번에 재실행하지 않았다. [논문 §3.1.2](https://arxiv.org/html/2104.01778v2#S3.SS1.SSS2).

mAP 0.459는 모든 clip의 정답률 45.9%가 아니며 클래스별 precision-recall 기반 평가를 평균한 지표다. AudioSet은 clip에 소리가 존재한다는 약한 라벨을 쓰므로 이 checkpoint의 점수만으로 사건의 정확한 시작·끝 시간이나 신뢰도 보정을 보장하지 않는다. ESC-50·Speech Commands 논문 결과도 각 데이터로 추가 학습한 실험으로 이 체크포인트에 그대로 적용하지 않았다.

[Whisper 분석](2026-10-07-whisper-large-v3.md)은 음성 전사, [MusicGen 분석](2026-10-25-musicgen-small.md)은 음악 생성이다. AST는 환경음·음성·음악 범주의 clip tagging에 적합한 기준선이다. 실제 녹음에서 잡음·장치·클래스 불균형에 따른 recall과 false positive를 검증해야 한다.

F32 가중치만 계산하면 `86,594,063×4=346,376,252 bytes`, 약 346.4MB(십진 단위)다. VRAM 실측이나 최소 요구량이 아니다. 1,214토큰의 attention·activation, 입력 batch와 런타임이 별도 메모리를 요구한다.
