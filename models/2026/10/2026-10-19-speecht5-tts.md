# SpeechT5 TTS: 텍스트를 Mel로, Mel을 파형으로 바꾸는 음성 합성

조사일: 2026-10-19 KST. ID `microsoft/speecht5_tts`, revision `30fcde30f19b87502b8435427b5f5068e401d5f6`.

## 선정과 식별

Microsoft SpeechT5 계열을 LibriTTS로 음성 합성 fine-tuning한 체크포인트다. 라이선스는 MIT다. 정확한 최초 체크포인트 배포일은 확인 불가이고 논문 최초 공개일 2021-10-14와 구분한다. HF 카드는 Hugging Face 팀 작성임을 명시한다. 기반 사전학습 SpeechT5의 정확한 revision은 확인 불가다. [고정 카드](https://huggingface.co/microsoft/speecht5_tts/blob/30fcde30f19b87502b8435427b5f5068e401d5f6/README.md), [논문 기록](https://arxiv.org/abs/2110.07205)

조사 시점 최근 한 달 다운로드 48,506회·좋아요 847개다. 좋아요 집계 기간은 확인 불가이며 인기 지표는 품질 증거가 아니다. [HF 페이지](https://huggingface.co/microsoft/speecht5_tts), [API](https://huggingface.co/api/models/microsoft/speecht5_tts)

모든 기존 models·issues를 읽어 중복이 없음을 확인했다. 기존 Whisper·Wav2Vec2의 음성 인식과 반대 방향인 음성 생성 분야를 선정했다. 큰 가중치 다운로드·추론·학습·유료 연산은 수행하지 않았다.

## 구조

아래 값은 [고정 config.json](https://huggingface.co/microsoft/speecht5_tts/blob/30fcde30f19b87502b8435427b5f5068e401d5f6/config.json)에 근거한다.

| 항목 | 값 |
|---|---|
| 유형 | SpeechT5ForTextToSpeech, encoder-decoder |
| encoder / decoder layers | 12 / 6 |
| hidden dimension | 768 |
| encoder / decoder FFN | 3,072 / 3,072 |
| attention heads | 양쪽 각각 12; 별도 GQA KV heads 설정 없음 |
| head dimension | 768÷12=64, 균등 분할 가정 계산 |
| activation | GELU |
| text / speech 위치 설정 | max_text_positions=600 / max_speech_positions=1,876 |
| encoder 상대 위치 범위 설정 | encoder_max_relative_position=160 |
| Mel bins | 80 |
| reduction factor | 2 |
| speaker embedding dimension | 512 |
| speech decoder prenet | 2층, 256 units |
| speech decoder postnet | 5층, 256 units, kernel 5 |
| vocab slots | 81 |
| 저장 dtype | float32; 학습 precision과 별개 |
| 총·활성 파라미터 | 정확한 배포본 총수 확인 불가; MoE식 활성량 구분 해당 없음 |
| MoE·이미지 입력 | 없음 |

텍스트 encoder가 문맥을 읽고, causal decoder가 encoder를 참조해 음성 특징을 차례로 출력한다. Speaker embedding은 발화 음색의 조건이다. Reduction factor 2는 decoder step당 Mel frame 묶음과 관련된 구조값이며 학습 epoch 2회를 뜻하지 않는다. [모델 카드의 generate_speech 예제](https://huggingface.co/microsoft/speecht5_tts/blob/30fcde30f19b87502b8435427b5f5068e401d5f6/README.md)

이 체크포인트의 주 출력은 Mel 스펙트로그램이다. 카드 예제는 별도 `microsoft/speecht5_hifigan`을 연결해 16kHz 파형으로 만든다. 따라서 TTS 모델만의 크기·계산량과 vocoder 포함 비용을 구분해야 한다. 음성 위치 설정 1,876을 1,876초나 텍스트 1,876토큰으로 읽지 않는다.

config에는 speech encoder용 합성곱 설정도 있으나 이를 텍스트→음성 경로에서 모두 실행되는 층으로 합산하지 않는다. 전체 SpeechT5 프레임워크가 여러 모달 pre/post-net을 갖는 것과 해당 TTS 경로를 구분한다. 위치 인코딩의 전체 구현은 고정 config만으로 확인 불가이며 LLM의 RoPE를 임의로 적용하지 않는다.

## 학습: 논문 초판과 배포 모델을 구분

| 항목 | SpeechT5 논문 v1 레시피 | 이 HF TTS 체크포인트 |
|---|---|---|
| 사전학습 | 음성·텍스트 공동 표현 학습 | 기반 사전학습 상세 revision 확인 불가 |
| 사전학습 step | 40K | 실제 실행 로그 확인 불가 |
| 사전학습 optimizer | Adam, 초기 10% warmup | 확인 불가 |
| 사전학습 accumulation·GPU | update frequency 4, 32GPU 각 32GB | 확인 불가 |
| TTS fine-tuning 데이터 | LibriTTS | 카드에서 LibriTTS 확인 |
| TTS step / learning rate | 300K / 4e-4 | 실제 완료 step·LR 확인 불가 |
| TTS batch | GPU당 최대 45K tokens | 실제 실행값 확인 불가 |
| TTS GPU | 논문 fine-tuning 8GPU | 실제 실행값 확인 불가 |
| epoch | 해당 값 확인 불가 | 미공개 |
| scheduler·optimizer 세부·precision | TTS 전용 전체값 확인 불가 | 확인 불가 |
| 별도 DPO·RL·LoRA | 적용 근거 확인 불가 | 적용 근거 확인 불가 |

논문 수치 근거는 [v1 §3.2](https://arxiv.org/html/2110.07205v1#S3.SS2)다. 초판은 speech 입력 pre-net·텍스트 어휘 등에서 현재 HF config와 차이가 있으므로 그 레시피를 배포본의 확정 학습값으로 옮겨 적지 않는다. 카드의 빈 hyperparameter 항목도 확인한 값처럼 채우지 않았다.

공식 [Microsoft SpeechT5 코드 경로](https://github.com/microsoft/SpeechT5/tree/main/SpeechT5)는 이번 웹 조회에 실패했다. 따라서 실제 학습 스크립트·실행 인자까지 검증했다고 주장하지 않는다. ‘미공개’는 검토한 배포 카드에 값이 없다는 뜻이며, ‘확인 불가’는 이번 조사에서 검증하지 못한 값이다.

Epoch는 데이터 순회 횟수, step은 optimizer 업데이트, 12·6은 모델 깊이다. 추론에서 Mel을 생성하는 반복 횟수 역시 학습량과 다르다. 예제의 speaker dataset이나 추론 dtype를 학습 데이터·precision으로 취급하지 않는다.

## 평가와 비교

논문 v1 TTS 표는 SpeechT5의 MOS 3.65±0.04와 WER 1.49를 보고한다. MOS는 고정된 문장 200개에서 평가했고 WER는 전체 test 데이터 대상이다. 이는 당시 논문 실험 결과이며, 현재 HF revision·vocoder·speaker 조건이 같다고 검증하지 못했으므로 배포본의 실측 성능으로 쓰지 않는다. [논문 표 3](https://arxiv.org/html/2110.07205v1#S3.SS3)

실용 비교는 같은 문장·화자 조건·vocoder·샘플링 속도에서 자연스러움과 발음 오류를 나누어 평가해야 한다. MOS와 인식기로 계산한 WER는 다른 지표다. HF 카드 자체의 평가 항목은 미완성이므로 그 빈칸이 높은 품질의 증거가 아니다.

기존 [Whisper 분석](2026-10-07-whisper-large-v3.md)은 음성→텍스트, 이번 모델은 텍스트→음성이다. 둘 다 encoder-decoder이지만 입출력과 학습 목표가 달라 같은 점수로 비교할 수 없다.

## 활용과 제약

영어 안내 음성·읽기 음성 합성 실험에 검토할 수 있다. 한국어 입력과 장문 처리 품질은 별도 검증이 필요하며 단순 speaker embedding 제공을 임의 화자의 정확한 재현 보장으로 해석하지 않는다. TTS 파이프라인 구성에는 텍스트 처리·화자 조건·vocoder가 함께 필요하다.

속도·VRAM·자연스러움은 실측하지 않았다. 총 파라미터를 확정하지 못해 가중치 메모리 수치도 임의로 계산하지 않았다. 입력 길이·생성 길이·batch·vocoder에 따라 실제 비용이 달라진다.
