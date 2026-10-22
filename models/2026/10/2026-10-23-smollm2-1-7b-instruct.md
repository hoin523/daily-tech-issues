# SmolLM2-1.7B-Instruct: 24층 소형 모델의 사전학습·SFT·DPO

조사일: 2026-10-23 KST. ID `HuggingFaceTB/SmolLM2-1.7B-Instruct`, revision `31b70e2e869a7173562077fd711b654946d38674`.

## 선정과 식별

Hugging Face Smol Models 연구팀의 영어 중심 지시 수행 모델이다. 기반은 `HuggingFaceTB/SmolLM2-1.7B`이며 라이선스는 Apache-2.0이다. 정확한 최초 체크포인트 발표일은 이번 조사에서 확인 불가다. 2025년 기술 보고서 공개일을 이전 모델 출시일로 치환하지 않는다. [고정 카드](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/blob/31b70e2e869a7173562077fd711b654946d38674/README.md), [보고서](https://arxiv.org/abs/2502.02737)

조사 당시 최근 한 달 다운로드 240,372회·좋아요 765개였다. 좋아요 집계 기간은 확인 불가이며 인기도는 성능 증거가 아니다. [HF 페이지](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct), [API](https://huggingface.co/api/models/HuggingFaceTB/SmolLM2-1.7B-Instruct)

전체 models·issues를 읽어 동일 모델 분석이 없음을 확인했다. 최근 영상·비전·음성 분석 이후 다른 개발 조직의 소형 LLM을 선정했다. 가중치 다운로드·추론·학습은 수행하지 않았다.

## 구조

표는 [고정 config.json](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/blob/31b70e2e869a7173562077fd711b654946d38674/config.json)과 조사 당시 API 집계에 근거한다.

| 항목 | 값 |
|---|---|
| architecture | LlamaForCausalLM, dense decoder |
| 총 파라미터 | 1,711,376,384 BF16 원소, 약 1.71B |
| layers | 24 |
| hidden / FFN intermediate | 2,048 / 8,192 |
| query / KV heads | 32 / 32 |
| head dimension | 2,048÷32=64, 균등 분할 가정 계산 |
| 위치 상한 | 8,192 |
| RoPE theta / scaling | 130,000 / null |
| 활성화 | SiLU |
| RMSNorm epsilon | 1e-5 |
| 어휘 슬롯 | 49,152 |
| 임베딩 공유 | true |
| attention / MLP bias | 모두 false |
| 저장 dtype | bfloat16 |
| MoE·멀티모달 | 없음; MoE식 활성 파라미터 구분 해당 없음 |

같은 Llama 구현을 사용한다는 사실이 Meta의 Llama 가중치를 fine-tuning했다는 뜻은 아니다. 모델 카드는 독자적인 SmolLM2 base 학습을 설명한다. Query·KV heads가 같아 GQA로 KV heads를 줄이는 구성과 다르다.

토큰은 이전 토큰을 참조하며 24개 블록을 지나 다음 토큰 점수로 변환된다. 대화 때는 카드의 chat template을 사용해 역할을 표현한다. Function calling은 모델이 구조화된 호출 텍스트를 출력하는 기능이며 외부 함수를 자체 실행한다는 뜻은 아니다. [사용 예제](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/blob/31b70e2e869a7173562077fd711b654946d38674/README.md)

config의 Transformers.js q4 옵션은 배포용 추론 설정이며 원본을 4bit로 학습했다는 증거가 아니다. 위치 상한 8K도 모든 길이에서 동일한 검색·추론 품질을 보장하지 않는다.

## 단계별 학습

다음 표는 [보고서 §4–5](https://arxiv.org/html/2502.02737v1)와 [카드 Training](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/blob/31b70e2e869a7173562077fd711b654946d38674/README.md)의 공개값을 구분했다.

| 항목 | 사전학습 | SFT | DPO |
|---|---|---|---|
| 데이터·처리량 | 11T tokens | SmolTalk | UltraFeedback |
| epoch | 보고서 약 2회, 혼합 데이터 전체 기준 | 2 | 2 |
| step | 실제 총 로그 확인 불가 | 확인 불가 | 확인 불가 |
| global batch | 이번 조사에서 검증하지 못함 | 128 | 128 |
| sequence length | 단계별 변경 | 8,192 | 1,024 |
| learning rate | peak 5e-4 | 3e-4 | 1e-6 |
| scheduler | WSD, warmup 2,000 steps | 확인 불가 | 확인 불가 |
| optimizer | AdamW | 이번 조사에서 검증하지 못함 | 이번 조사에서 검증하지 못함 |
| gradient accumulation | 확인 불가 | 확인 불가 | 확인 불가 |
| DPO beta | 해당 없음 | 해당 없음 | 0.5 |
| precision·hardware | 카드 BF16·256 H100 | 해당 단계 전용 확인 불가 | 해당 단계 전용 확인 불가 |
| 별도 RL·LoRA | 적용 근거 확인 불가 | 적용 근거 확인 불가 | 적용 근거 확인 불가 |

사전학습은 단계마다 웹·수학·코드 혼합 비율을 바꾼다. 약 2 epoch를 각 데이터셋이 정확히 두 번 순회됐다고 해석하지 않는다. SFT와 DPO의 2회도 서로 다른 데이터와 목적에 대한 반복이다. DPO는 선호 쌍 최적화이며 온라인 환경에서 reward를 받아 학습한 RL과 구분한다.

공식 [smollm 코드 저장소](https://github.com/huggingface/smollm)는 카드가 공개한 사전·후처리·평가 경로다. 이번 분석에서 모든 실제 실행 인자와 로그를 대조한 것은 아니며 미검증 값은 확인 불가로 표시했다. Epoch는 데이터 반복, step은 업데이트, 24 layers는 구조 깊이다.

## 평가와 관련 모델 비교

고정 카드의 Instruct 평가 표는 IFEval 평균 56.7, GSM8K 5-shot 48.2를 보고한다. 저자 측 lighteval 결과이며 직접 재측정하지 않았다. Base 표의 HellaSwag 68.7을 Instruct 점수로 옮기면 안 된다. 같은 카드의 Instruct HellaSwag는 66.1이다. [고정 평가 표](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/blob/31b70e2e869a7173562077fd711b654946d38674/README.md)

벤치마크의 shot·출력 예산·판정 기준을 맞춘 비교가 필요하다. 영어 중심 평가를 한국어 업무 정확도로 일반화하지 않는다. Function calling 예제 존재만으로 임의 API 호출의 정확성도 보장되지 않는다.

기존 [Phi-2 분석](2026-10-14-phi-2.md)은 32층·hidden 2,560·2K 위치 상한의 base 모델이며 instruction SFT가 없다. 이번 모델은 24층·hidden 2,048·8K 위치 상한과 SFT·DPO를 갖춘다. 파라미터 수나 층수만으로 품질·지연 시간의 우열을 판단할 수 없다.

## 활용과 제약

영어 요약·문장 재작성·제한된 지시 수행과 로컬 실행 실험에 검토할 수 있다. 한국어, 사실 정확도, 장문 안정성과 도구 호출 인자는 실제 입력에서 평가해야 한다. [카드 제한 사항](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/blob/31b70e2e869a7173562077fd711b654946d38674/README.md)

BF16 가중치만 가정하면 `1,711,376,384×2≈3.42 GB`(십진)다. 계산값이며 최소 VRAM·실측이 아니다. KV cache·activation·batch·런타임 버퍼가 추가된다. 속도·한국어 품질을 직접 측정하지 않았다.
