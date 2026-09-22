시계열 화재/비화재/정상 분류 모델 개발

현재 프로젝트의 LDT_processed 폴더에 있는 시계열 데이터를 이용하여 **정상(0), 화재(1), 비화재(2)**를 분류하는 머신러닝 모델을 개발해줘.

1. 데이터 구조

LDT_processed 폴더 아래에는 다음과 같은 3개의 폴더가 있다.

LDT_processed/
├── 0_normal/
├── 1_fire/
└── 2_nofire/


각 폴더에는 여러 개의 CSV 파일이 존재한다.

중요:
0_normal, 1_fire, 2_nofire 폴더는 각 CSV 파일의 모든 시점이 해당 클래스를 의미한다는 뜻이 아니다.

이 폴더 구분은 데이터셋을 Training / Validation / Test로 분할할 때 각 폴더의 시나리오 비율을 8:1:1로 유지하기 위한 그룹 구분이다.

실제 각 시점의 정답 클래스는 CSV 내부의 LABEL 컬럼을 사용해야 한다.

즉,

폴더명 → 데이터셋 분할 시 stratification/group 기준

LABEL → 실제 시점별 classification target

으로 처리해야 한다.

2. CSV 구조

각 CSV 파일은 하나의 시나리오(scenario)에 해당한다.

컬럼은 총 9개이며 다음과 같다.

MAC
INST_DT
TEMPO
FLAME
SMOKE
CODE
STATUS
LABEL
CATEGORY


각 컬럼의 의미는 다음과 같다.

MAC: 화재감지기의 MAC 주소

INST_DT: timestamp

TEMPO: 온도 센서 값

FLAME: 불꽃 센서 값

SMOKE: 연기 센서 값

CODE: 기타 정보

STATUS: 기타 상태 정보

LABEL: 시점별 ground-truth label

0: 정상

1: 화재

2: 비화재

CATEGORY: 기타 분류 정보

이번 모델에서는 오직 TEMPO, FLAME, SMOKE만 입력 feature로 사용한다.

MAC, CODE, STATUS, CATEGORY는 모델 입력으로 사용하지 않는다.

3. 목표

시계열 데이터를 이용하여 특정 시점의 LABEL을 예측하는 분류 모델을 개발한다.

특정 시점 t의 label을 예측할 때,

t-9
t-8
t-7
...
t-1
t


총 10개의 시점을 입력으로 사용한다.

각 시점에는 다음 3개의 센서 값이 있다.

TEMPO
FLAME
SMOKE


따라서 하나의 입력 sample은 다음과 같은 10 × 3 행렬이 된다.

[
    [TEMPO(t-9), FLAME(t-9), SMOKE(t-9)],
    [TEMPO(t-8), FLAME(t-8), SMOKE(t-8)],
    ...
    [TEMPO(t),   FLAME(t),   SMOKE(t)]
]


그리고 해당 sample의 target은 현재 시점 t의

LABEL(t)


이다.

즉,

X.shape = (10, 3)
y = LABEL(t)


가 되도록 한다.

4. 데이터 전처리 및 정렬

각 CSV 파일을 읽을 때 반드시 INST_DT를 기준으로 시간 순서대로 정렬한다.

시간 순서가 보장되지 않은 CSV가 있을 가능성을 고려하여 다음을 확인한다.

INST_DT를 datetime 형식으로 변환

timestamp 기준 오름차순 정렬

timestamp 중복 여부 확인

결측값 여부 확인

TEMPO, FLAME, SMOKE, LABEL의 결측값 여부 확인

전처리 과정에서 문제가 발견되면 조용히 데이터를 삭제하지 말고 로그에 기록한다.

5. 매우 중요한 데이터 분할 규칙

Training / Validation / Test는 CSV 파일(시나리오) 단위로 분할해야 한다.

하나의 CSV 파일에서 생성된 여러 개의 sliding-window sample이 서로 다른 dataset에 들어가면 안 된다.

즉, 하나의 scenario CSV는 반드시 다음 중 하나에만 속해야 한다.

Train
Validation
Test


절대로 하나의 CSV에서 일부 window는 train, 일부 window는 validation/test에 들어가는 방식으로 분할하지 않는다.

이는 동일한 시나리오의 매우 유사한 연속 데이터가 train과 test에 동시에 존재하는 data leakage를 방지하기 위한 것이다.

6. 8:1:1 분할

각 상위 폴더의 CSV 파일들을 기준으로 다음 비율을 최대한 유지하여 분할한다.

Training   : 80%
Validation : 10%
Test       : 10%


그리고 반드시 각 폴더별로 8:1:1 비율을 유지한다.

예를 들어,

0_normal
1_fire
2_nofire


각각의 CSV 파일들을 독립적으로

80% → train
10% → validation
10% → test


로 나눈다.

중요한 점은 분할 기준이 CSV 파일의 개수이며, window/sample의 개수가 아니다.

CSV 개수가 8:1:1로 정확하게 나누어지지 않는 경우에는 정수 개수 문제를 합리적으로 처리하고, 실제 분할 결과를 로그로 출력한다.

분할 결과를 재현할 수 있도록 random seed를 고정한다.

예:

SEED = 42


가능하다면 train/validation/test에 사용된 CSV 파일 목록을 별도의 파일로 저장한다.

예:

splits/
├── train_files.csv
├── val_files.csv
└── test_files.csv

7. Sliding Window 생성

각 CSV 파일 내부에서만 sliding window를 생성한다.

window size는 10이다.

예를 들어 한 CSV가 다음과 같이 구성되어 있다면,

t0
t1
t2
...
t99


첫 번째 sample은 충분한 과거 데이터가 있는 시점부터 생성한다.

예:

X[0] = [t0, t1, ..., t9]
y[0] = LABEL(t9)

X[1] = [t1, t2, ..., t10]
y[1] = LABEL(t10)

...


즉, 일반적으로

X[i] = data[i:i+10, [TEMPO, FLAME, SMOKE]]
y[i] = LABEL[i+9]


형태가 되도록 한다.

CSV의 경계를 넘어 sliding window를 만들면 안 된다.

8. Label 처리

target은 반드시 window의 마지막 시점의 LABEL을 사용한다.

즉,

window = [t-9, ..., t]
target = LABEL(t)


이다.

LABEL은 다음과 같이 처리한다.

0 → normal
1 → fire
2 → nofire


모델의 output은 3-class classification으로 구성한다.

9. Feature scaling

TEMPO, FLAME, SMOKE에 대해 feature scaling이 필요한지 검토하고 적절한 방법을 사용한다.

Scaling을 사용하는 경우 반드시 training dataset으로만 scaler를 fit해야 한다.

예를 들어 StandardScaler를 사용하는 경우:

fit   → train
transform → train
transform → validation
transform → test


와 같이 처리한다.

Validation이나 test 데이터를 이용하여 scaler를 fit하면 안 된다.

사용한 scaler도 저장한다.

예:

artifacts/scaler.pkl

10. 개발할 모델

먼저 다음 두 모델을 구현한다.

Model 1: MLP

입력은 10×3 시계열 데이터를 flatten하여 사용한다.

10 × 3 → 30


예를 들어 다음과 같은 구조를 사용할 수 있다.

Input: 30
↓
Dense
↓
ReLU
↓
Dropout
↓
Dense
↓
ReLU
↓
Dropout
↓
Dense(3)
↓
Softmax


단, hidden layer의 크기와 dropout 등 hyperparameter는 합리적인 기본값을 사용하고 코드에서 쉽게 변경할 수 있도록 한다.

Model 2: LSTM

LSTM은 10개의 timestep과 3개의 feature를 그대로 사용한다.

입력 shape:

(batch_size, 10, 3)


예를 들어 다음과 같은 구조를 사용한다.

Input: (10, 3)
↓
LSTM
↓
Dropout
↓
Dense
↓
ReLU
↓
Dense(3)
↓
Softmax


LSTM의 hidden size, dropout 등의 hyperparameter 역시 코드에서 쉽게 변경할 수 있도록 한다.

11. Training

두 모델 모두 동일한 train/validation/test split을 사용한다.

가능하면 다음을 적용한다.

fixed random seed

Early stopping

validation loss 또는 validation macro-F1을 기준으로 best model 저장

batch size를 configurable하게 설정

epoch 수를 configurable하게 설정

optimizer 및 learning rate를 configurable하게 설정

분류 문제이므로 적절한 loss function을 사용한다.

기본적으로 cross entropy 기반 loss를 사용한다.

12. Class imbalance 확인

Training 데이터에서 다음을 계산한다.

class 0 sample count
class 1 sample count
class 2 sample count


그리고 class imbalance가 심한지 확인한다.

필요한 경우 class weight를 적용할 수 있도록 구현한다.

단, class weight를 사용할 경우 train set의 label distribution만 이용해서 계산한다.

Validation/test distribution을 이용하여 class weight를 결정하지 않는다.

13. 평가

MLP와 LSTM을 동일한 test set에서 평가한다.

단순 accuracy뿐만 아니라 다음 지표를 모두 계산한다.

Accuracy
Precision
Recall
F1-score


특히 3-class classification이므로 다음을 모두 출력한다.

Macro F1
Weighted F1
Per-class Precision
Per-class Recall
Per-class F1


Class별 이름은 다음과 같이 표시한다.

0 = normal
1 = fire
2 = nofire


그리고 confusion matrix를 생성한다.

Confusion matrix의 축에는 반드시 class name을 표시한다.

14. 시계열 분류 특성을 고려한 추가 평가

이 데이터는 일반적인 독립적인 tabular sample이 아니라 시계열 데이터이므로, 가능하다면 test set에 대해 다음도 확인한다.

각 CSV scenario별로 prediction을 수행하고,

timestamp
true LABEL
predicted LABEL


을 확인할 수 있도록 한다.

특히 실제 화재 구간에서 다음을 확인한다.

실제 fire(1)를 얼마나 잘 탐지하는지

fire를 normal(0) 또는 nofire(2)로 잘못 판단하는 경우

normal/nofire를 fire로 잘못 판단하는 경우

이를 확인할 수 있는 scenario별 prediction 결과를 저장한다.

15. 결과물

프로젝트를 다음과 같이 구성해줘.

예:

project/
├── LDT_processed/
│   ├── 0_normal/
│   ├── 1_fire/
│   └── 2_nofire/
│
├── src/
│   ├── data.py
│   ├── preprocessing.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
│
├── splits/
│   ├── train_files.csv
│   ├── val_files.csv
│   └── test_files.csv
│
├── artifacts/
│   ├── scaler.pkl
│   ├── mlp_best.*
│   └── lstm_best.*
│
├── results/
│   ├── metrics.csv
│   ├── confusion_matrix_mlp.png
│   ├── confusion_matrix_lstm.png
│   ├── training_curve_mlp.png
│   ├── training_curve_lstm.png
│   └── scenario_predictions/
│
└── README.md


실제 프로젝트 구조에 맞게 조정해도 된다.

16. 최종 비교

최종적으로 MLP와 LSTM의 성능을 비교할 수 있도록 표를 만들어라.

예:

Model	Accuracy	Macro F1	Weighted F1
MLP	...	...	...
LSTM	...	...	...

추가적으로 class별 성능도 비교한다.

Model	Class	Precision	Recall	F1
MLP	Normal	...	...	...
MLP	Fire	...	...	...
MLP	Nofire	...	...	...
LSTM	Normal	...	...	...
LSTM	Fire	...	...	...
LSTM	Nofire	...	...	...
17. 매우 중요한 검증 사항

코드를 작성한 후 실제 데이터를 이용하여 다음을 반드시 검증해라.

데이터 누수 확인

동일 CSV 파일이 train/validation/test에 중복으로 들어가지 않았는지 확인

하나의 CSV에서 생성된 window가 여러 dataset에 분산되지 않았는지 확인

scaler가 train 데이터에만 fit되었는지 확인

Window 확인

임의의 CSV 하나를 선택하여 실제 생성된 sample을 출력하고 다음을 검증한다.

X shape = (10, 3)
y = 마지막 timestep의 LABEL

Label 확인

실제 CSV의 timestamp와 생성된 window의 마지막 timestamp 및 LABEL을 출력하여 target이 정확히 마지막 timestep의 LABEL인지 검증한다.

Split 확인

각 폴더별 CSV 개수에 대해

total
train
validation
test


를 출력하고 8:1:1 분할이 의도대로 이루어졌는지 확인한다.

18. 실행 방법

코드 작성이 끝난 후 실제 데이터에 대해 학습을 실행한다.

MLP와 LSTM 모두 학습하고 test set에서 평가한다.

실행에 필요한 명령어를 README에 작성한다.

예:

python train.py
python evaluate.py


실제 프로젝트 환경에 맞게 수정한다.

19. 최종 보고

개발이 끝나면 단순히 코드만 작성하지 말고 다음 내용을 요약해서 보고해라.

전체 CSV 파일 개수

각 폴더별 CSV 파일 개수

Train / Validation / Test CSV 개수

각 dataset의 window/sample 개수

각 dataset의 LABEL 분포

사용한 preprocessing

MLP 구조

LSTM 구조

training 설정

MLP test 성능

LSTM test 성능

confusion matrix 결과

두 모델의 class별 성능 차이

발생한 문제 및 해결 방법

특히 folder label과 CSV 내부 LABEL을 혼동하지 않았는지 최종 보고에서 명시적으로 확인해라.

20. 구현 원칙

가장 중요한 원칙은 다음과 같다.

Folder
  ↓
Scenario-level split (8:1:1)
  ↓
CSV별 preprocessing
  ↓
CSV 내부에서 sliding window 생성
  ↓
10 × 3 input
  ↓
MLP / LSTM
  ↓
LABEL(t) 예측


폴더명은 실제 timestep의 label이 아니다.

실제 classification target은 반드시 CSV 내부의 LABEL 컬럼이다.

또한 동일 scenario의 데이터가 train과 test에 동시에 들어가 데이터 leakage가 발생하지 않도록 해야 한다.
