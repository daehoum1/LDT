추가 요구사항: 5개 머신러닝 모델 학습 및 성능 비교
앞서 설명한 프로젝트의 데이터 구조, 전처리 방법, scenario-level split 규칙, sliding window 생성 방식, LABEL 처리 규칙을 그대로 유지한다.

기존에 설명한 MLP와 LSTM뿐만 아니라 다음 총 5개 모델을 구현하고 동일한 조건에서 학습 및 평가하라.

1. 비교할 모델
반드시 다음 5개 모델을 구현한다.

MLP

LSTM

GRU

1D-CNN

CNN-LSTM

모델 이름은 코드 전체에서 다음과 같이 통일한다.

MLP
LSTM
GRU
CNN1D
CNN_LSTM

2. 가장 중요한 비교 원칙
5개 모델의 성능을 공정하게 비교하기 위해 다음 조건을 반드시 동일하게 유지한다.

동일한 CSV scenario split

동일한 random seed

동일한 train / validation / test CSV

동일한 sliding window

동일한 window size = 10

동일한 input feature = TEMPO, FLAME, SMOKE

동일한 target = window 마지막 timestep의 LABEL

동일한 train-fitted scaler

동일한 test set

동일한 평가 metric

동일한 class name

가능한 한 동일한 training 조건

특히 모델별로 별도의 random split을 새로 수행하지 않는다.

한 번 생성한 split을 모든 모델이 공유해야 한다.

즉,

CSV files
   ↓
Scenario-level split
   ↓
train_files.csv
val_files.csv
test_files.csv
   ↓
동일한 split을
모든 5개 모델이 사용

이어야 한다.

3. 모델별 입력
MLP
각 sample:

(10, 3)

을 flatten한다.

10 × 3 = 30

따라서 입력은:

(batch_size, 30)

이다.

권장 기본 구조:

Input(30)
    ↓
Dense(64)
    ↓
ReLU
    ↓
Dropout(0.3)
    ↓
Dense(32)
    ↓
ReLU
    ↓
Dropout(0.3)
    ↓
Dense(3)

최종 출력은 3-class logits 또는 softmax probability 중 하나로 구현한다.

PyTorch를 사용한다면 마지막 layer에는 softmax를 직접 넣지 않고 CrossEntropyLoss를 사용하는 방식을 권장한다.

4. LSTM
입력:

(batch_size, 10, 3)

권장 기본 구조:

Input(10, 3)
    ↓
LSTM(hidden_size=64)
    ↓
last timestep 또는 final hidden state
    ↓
Dropout(0.3)
    ↓
Dense(32)
    ↓
ReLU
    ↓
Dense(3)

기본 hidden size:

64

bidirectional은 기본적으로 사용하지 않는다.

필요하면 configuration에서 쉽게 변경할 수 있도록 한다.

5. GRU
입력:

(batch_size, 10, 3)

권장 기본 구조:

Input(10, 3)
    ↓
GRU(hidden_size=64)
    ↓
last timestep 또는 final hidden state
    ↓
Dropout(0.3)
    ↓
Dense(32)
    ↓
ReLU
    ↓
Dense(3)

기본 hidden size:

64

LSTM과 최대한 비슷한 parameter 규모와 training 조건을 사용하여 비교가 가능하도록 한다.

6. 1D-CNN
입력은 기본적으로:

(batch_size, 10, 3)

이다.

PyTorch Conv1d를 사용할 경우 필요한 shape:

(batch_size, channels=3, sequence_length=10)

으로 transpose하여 사용한다.

권장 기본 구조:

Input(10, 3)
    ↓
Transpose
    ↓
Conv1D(in_channels=3, out_channels=32, kernel_size=3, padding=1)
    ↓
ReLU
    ↓
Conv1D(32, 64, kernel_size=3, padding=1)
    ↓
ReLU
    ↓
AdaptiveAvgPool1d(1)
    ↓
Flatten
    ↓
Dropout(0.3)
    ↓
Dense(32)
    ↓
ReLU
    ↓
Dense(3)

kernel size, channel 수, dropout 등은 configuration에서 쉽게 변경 가능하도록 한다.

7. CNN-LSTM
5개 모델 중 hybrid model이다.

CNN을 이용하여 센서의 local temporal pattern을 추출하고, 그 결과를 LSTM에 입력한다.

권장 구조:

Input(10, 3)
    ↓
Conv1D
    ↓
ReLU
    ↓
Conv1D
    ↓
ReLU
    ↓
Transpose back to
(batch_size, sequence_length, channels)
    ↓
LSTM
    ↓
last hidden state
    ↓
Dropout
    ↓
Dense(32)
    ↓
ReLU
    ↓
Dense(3)

예를 들어:

Conv1D:
in_channels = 3
out_channels = 32
kernel_size = 3
padding = 1

Conv1D:
in_channels = 32
out_channels = 64
kernel_size = 3
padding = 1

LSTM:
input_size = 64
hidden_size = 64

를 기본값으로 사용한다.

8. Framework
가능하면 전체 프로젝트는 PyTorch 기반으로 통일한다.

권장 라이브러리:

Python
PyTorch
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib

필요한 경우 tqdm 등을 추가한다.

TensorFlow/Keras 대신 PyTorch를 사용한다.

9. 프로젝트 구조 확장
기존 구조를 다음과 같이 확장한다.

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
│   ├── utils.py
│   └── config.py
│
├── splits/
│   ├── train_files.csv
│   ├── val_files.csv
│   └── test_files.csv
│
├── artifacts/
│   ├── scaler.pkl
│   ├── mlp_best.pt
│   ├── lstm_best.pt
│   ├── gru_best.pt
│   ├── cnn1d_best.pt
│   └── cnn_lstm_best.pt
│
├── results/
│   ├── metrics.csv
│   ├── class_metrics.csv
│   ├── confusion_matrix_mlp.png
│   ├── confusion_matrix_lstm.png
│   ├── confusion_matrix_gru.png
│   ├── confusion_matrix_cnn1d.png
│   ├── confusion_matrix_cnn_lstm.png
│   ├── training_curve_mlp.png
│   ├── training_curve_lstm.png
│   ├── training_curve_gru.png
│   ├── training_curve_cnn1d.png
│   ├── training_curve_cnn_lstm.png
│   └── scenario_predictions/
│       ├── mlp/
│       ├── lstm/
│       ├── gru/
│       ├── cnn1d/
│       └── cnn_lstm/
│
├── logs/
│   ├── data_validation.log
│   ├── split.log
│   └── training.log
│
└── README.md

실제 프로젝트 상황에 맞게 합리적으로 조정해도 된다.

10. Configuration
hyperparameter를 코드 곳곳에 하드코딩하지 말고 configuration으로 관리한다.

예를 들어:

SEED = 42

WINDOW_SIZE = 10

BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

DROPOUT = 0.3

HIDDEN_SIZE = 64
DENSE_SIZE = 32

PATIENCE = 10

USE_CLASS_WEIGHT = True

모델별 설정도 쉽게 변경할 수 있도록 한다.

예:

MODEL_CONFIG = {
    "MLP": {...},
    "LSTM": {...},
    "GRU": {...},
    "CNN1D": {...},
    "CNN_LSTM": {...},
}

11. Random Seed
재현성을 위해 다음 random seed를 고정한다.

Python random
NumPy
PyTorch
CUDA

가능하면 deterministic behavior를 설정한다.

단, CUDA deterministic 설정으로 인해 성능이 감소하거나 특정 operation에서 문제가 발생한다면 README와 log에 기록한다.

12. Class imbalance
Train set의 LABEL distribution을 먼저 계산한다.

class 0 = normal
class 1 = fire
class 2 = nofire

예:

Train:
normal = ...
fire = ...
nofire = ...

class imbalance를 확인한다.

USE_CLASS_WEIGHT=True인 경우 train label distribution만 이용해서 class weight를 계산한다.

예:

sklearn.utils.class_weight.compute_class_weight

등을 사용할 수 있다.

중요:

Validation/Test label distribution은 class weight 계산에 절대 사용하지 않는다.

모든 모델에 동일한 class weight를 사용한다.

13. Loss
기본 loss는:

CrossEntropyLoss

를 사용한다.

class weight를 사용하는 경우:

CrossEntropyLoss(weight=class_weights)

를 사용한다.

모델의 마지막 layer에는 softmax를 넣지 않는 PyTorch 방식을 권장한다.

Prediction 시:

softmax(logits)
argmax

를 사용한다.

14. Early stopping
각 모델에 Early Stopping을 적용한다.

기본 기준:

Validation Macro F1

을 사용한다.

Validation Macro F1이 개선될 경우 best model을 저장한다.

예:

artifacts/mlp_best.pt
artifacts/lstm_best.pt
artifacts/gru_best.pt
artifacts/cnn1d_best.pt
artifacts/cnn_lstm_best.pt

Early stopping patience는 기본:

10 epochs

로 설정한다.

필요하면 configuration에서 변경 가능하게 한다.

15. Training history
각 epoch마다 최소한 다음을 기록한다.

epoch
train_loss
train_accuracy
train_macro_f1
val_loss
val_accuracy
val_macro_f1
learning_rate

CSV로 저장한다.

예:

results/history_mlp.csv
results/history_lstm.csv
results/history_gru.csv
results/history_cnn1d.csv
results/history_cnn_lstm.csv

16. Training curve
각 모델에 대해 다음 그래프를 생성한다.

Loss
Epoch
  vs
Train Loss
Validation Loss

Macro F1
Epoch
  vs
Train Macro F1
Validation Macro F1

가능하면 하나의 figure에 loss와 F1을 별도의 subplot으로 표시한다.

저장:

results/training_curve_mlp.png
results/training_curve_lstm.png
results/training_curve_gru.png
results/training_curve_cnn1d.png
results/training_curve_cnn_lstm.png

17. Test evaluation
Best validation checkpoint를 불러온 뒤 test set에서 평가한다.

반드시 다음 metric을 계산한다.

Accuracy
Macro Precision
Macro Recall
Macro F1
Weighted Precision
Weighted Recall
Weighted F1

그리고 class별:

Normal
Fire
Nofire

에 대해:

Precision
Recall
F1
Support

를 계산한다.

18. Fire detection 성능을 별도로 확인
이 프로젝트에서는 Fire class가 중요하므로 다음을 별도로 출력한다.

Fire Precision
Fire Recall
Fire F1

특히:

Fire Recall

을 반드시 별도 표시한다.

또한 실제 fire가 다음으로 오분류되는 횟수를 확인한다.

True Fire → Predicted Normal
True Fire → Predicted Nofire

반대로:

True Normal → Predicted Fire
True Nofire → Predicted Fire

도 확인한다.

19. Confusion Matrix
각 모델마다 confusion matrix를 생성한다.

Class 순서는 반드시:

Normal
Fire
Nofire

로 고정한다.

축 이름:

True Label
Predicted Label

을 표시한다.

파일:

results/confusion_matrix_mlp.png
results/confusion_matrix_lstm.png
results/confusion_matrix_gru.png
results/confusion_matrix_cnn1d.png
results/confusion_matrix_cnn_lstm.png

가능하면 raw count confusion matrix와 normalized confusion matrix 중 최소 하나를 생성한다.

20. Scenario-level prediction
매우 중요하다.

Test set의 각 CSV scenario에 대해서 각 모델의 prediction을 수행한다.

각 scenario마다 다음 정보를 저장한다.

scenario_file
timestamp
true_label
predicted_label
prediction_probability_normal
prediction_probability_fire
prediction_probability_nofire

예:

results/scenario_predictions/mlp/scenario_001.csv
results/scenario_predictions/lstm/scenario_001.csv
results/scenario_predictions/gru/scenario_001.csv
results/scenario_predictions/cnn1d/scenario_001.csv
results/scenario_predictions/cnn_lstm/scenario_001.csv

각 row는 해당 window의 마지막 timestep에 대응해야 한다.

즉:

window:
[t-9, ..., t]

prediction timestamp:
t

true label:
LABEL(t)

predicted label:
model prediction at t

가 되도록 한다.

21. Scenario별 Fire Detection 분석
각 test scenario마다 다음을 계산한다.

scenario
total_samples
true_fire_samples
predicted_fire_samples
correct_fire_predictions
fire_recall
false_fire_predictions

특히 실제 fire 구간이 존재하는 scenario에서:

True Fire
→ Predicted Fire
→ Predicted Normal
→ Predicted Nofire

개수를 확인한다.

결과를:

results/scenario_fire_detection.csv

에 저장한다.

22. Transition 분석
시계열 모델이므로 단순 sample metric 외에 prediction transition을 분석한다.

각 scenario에 대해:

True Label sequence
Predicted Label sequence

을 비교한다.

예를 들어:

Normal → Fire
Normal → Nofire
Nofire → Fire
Fire → Normal
Fire → Nofire

등의 transition을 확인할 수 있도록 한다.

가능하면 scenario prediction plot도 생성한다.

예:

X-axis = timestamp
Y-axis = class

True LABEL
Predicted LABEL

두 sequence를 같은 plot 또는 subplot으로 비교한다.

특히 Fire 구간이 잘 탐지되는지 시각적으로 확인할 수 있도록 한다.

23. 데이터 leakage 자동 검증
학습 전에 반드시 자동 검증 코드를 실행한다.

다음을 검사한다.

CSV 중복
train ∩ validation = empty
train ∩ test = empty
validation ∩ test = empty

이어야 한다.

Window leakage
각 window에 반드시 scenario_id를 부여한다.

동일 scenario_id가 여러 dataset에 존재하면 오류를 발생시킨다.

Scaler leakage
Scaler가 train data에만 fit되었는지 확인한다.

가능하면 scaler metadata에:

fit_dataset = train

등의 정보를 저장한다.

Target leakage
입력 feature에 다음 컬럼이 들어가지 않았는지 확인한다.

LABEL
CATEGORY
MAC
CODE
STATUS
INST_DT

모델 입력은 정확히:

TEMPO
FLAME
SMOKE

뿐이어야 한다.

24. 매우 중요한 Folder vs LABEL 검증
이 부분을 반드시 코드와 최종 보고서에서 명시적으로 확인한다.

폴더:

0_normal
1_fire
2_nofire

는 timestep target으로 사용하지 않는다.

폴더는 오직:

scenario-level split stratification/group 기준

으로 사용한다.

실제 classification target은:

CSV 내부 LABEL

이다.

예를 들어 0_normal 폴더의 CSV에:

LABEL = 0
LABEL = 0
LABEL = 2
LABEL = 1

등이 존재할 수 있으며 이를 그대로 target으로 사용해야 한다.

절대로:

0_normal → 모든 target = 0
1_fire → 모든 target = 1
2_nofire → 모든 target = 2

로 처리하지 않는다.

이 부분을 코드에 명확한 주석으로 남겨라.

25. Split stratification/group 규칙
기존 요구사항에 따라 각 상위 폴더별로 CSV scenario를:

80% Train
10% Validation
10% Test

로 분할한다.

즉:

0_normal CSVs
    → 80/10/10

1_fire CSVs
    → 80/10/10

2_nofire CSVs
    → 80/10/10

으로 처리한다.

분할은 반드시 CSV 파일 단위이다.

Window 단위로 split하지 않는다.

분할 결과를:

splits/train_files.csv
splits/val_files.csv
splits/test_files.csv

에 저장한다.

각 파일에는 최소한:

scenario_id
filepath
folder_class
split

정보를 포함한다.

26. Split count 출력
실행 시 반드시 다음을 출력한다.

예:

========== SCENARIO SPLIT ==========

0_normal
total      : XX
train      : XX
validation : XX
test       : XX

1_fire
total      : XX
train      : XX
validation : XX
test       : XX

2_nofire
total      : XX
train      : XX
validation : XX
test       : XX

실제 파일 개수와 비율을 함께 출력한다.

CSV 개수가 적어서 정확하게 8:1:1이 불가능한 경우 실제 정수 분할 결과를 명확히 출력한다.

27. Dataset statistics
학습 시작 전에 다음을 출력한다.

CSV statistics
Total CSV
0_normal CSV
1_fire CSV
2_nofire CSV

Scenario split
Train CSV
Validation CSV
Test CSV

Window count
Train samples
Validation samples
Test samples

LABEL distribution
각 dataset에 대해:

Train
  Normal: XX
  Fire: XX
  Nofire: XX

Validation
  Normal: XX
  Fire: XX
  Nofire: XX

Test
  Normal: XX
  Fire: XX
  Nofire: XX

을 출력한다.

28. Window sanity check
학습 전에 임의의 scenario CSV 하나를 선택하여 다음을 출력한다.

예:

========== WINDOW SANITY CHECK ==========

Scenario: xxx.csv

Window shape:
(10, 3)

Window:
[
 ...
]

Last timestamp:
...

LABEL at last timestamp:
...

Generated target:
...

Target match:
True

반드시:

Generated target == LABEL(last timestep)

인지 확인한다.

29. Timestamp 검증
각 CSV를 읽을 때:

INST_DT

를 datetime으로 변환한다.

다음 검사를 수행한다.

invalid timestamp count
duplicate timestamp count
missing timestamp count

시간 순서대로 정렬한다.

중복 timestamp가 존재할 경우 임의로 조용히 제거하지 않는다.

문제 내용을 log에 기록한다.

결측값도 동일하다.

특히:

TEMPO
FLAME
SMOKE
LABEL

의 결측값을 별도로 검사한다.

30. 결측값 처리
결측값을 발견했을 경우:

무조건 조용히 drop

하지 않는다.

어떤 scenario에서 몇 개의 row가 문제가 있는지 log로 기록한다.

기본적으로 가장 안전한 preprocessing 방법을 선택하고 그 이유를 README에 기록한다.

예를 들어 window 생성에 필요한 feature 또는 LABEL이 결측이면 해당 window를 생성할 수 없으므로 제외할 수 있다.

단, 제외된 row/window/scenario 수를 반드시 기록한다.

31. Scaling
기존 요구사항과 동일하게:

Scaler fit → Train samples only

으로 한다.

Validation/Test에는 transform만 수행한다.

기본적으로 StandardScaler를 사용한다.

TEMPO
FLAME
SMOKE

각 feature에 적용한다.

Scaler 저장:

artifacts/scaler.pkl

중요:

각 window를 만든 후 train sample 전체에서 fit할지, raw train timestep에서 fit할지 구현 방식을 명확히 정하고 README에 기록한다.

가장 중요한 것은 validation/test 정보가 scaler fitting에 들어가지 않는 것이다.

32. 동일한 preprocessing
5개 모델은 서로 다른 preprocessing을 사용하면 안 된다.

모두 동일한:

CSV
→ timestamp sort
→ missing-value handling
→ sliding window
→ scaler
→ tensor

pipeline을 사용해야 한다.

MLP만 flatten하고 나머지 모델은 (10, 3)을 유지한다.

33. Batch size 및 DataLoader
기본:

batch_size = 64

로 한다.

Train:

shuffle=True

Validation:

shuffle=False

Test:

shuffle=False

를 사용한다.

34. Optimizer
기본 optimizer:

Adam

learning rate:

1e-3

weight decay:

1e-4

로 시작한다.

configuration에서 변경 가능하도록 한다.

35. Learning rate scheduler
가능하면 다음 scheduler 중 하나를 사용한다.

ReduceLROnPlateau

를 권장한다.

validation macro-F1 또는 validation loss를 기준으로 scheduler를 적용한다.

사용한 scheduler와 parameter를 log에 기록한다.

36. 모델별 parameter 수
각 모델의 trainable parameter 수를 출력한다.

예:

MLP       parameters: XXXXX
LSTM      parameters: XXXXX
GRU       parameters: XXXXX
CNN1D     parameters: XXXXX
CNN_LSTM  parameters: XXXXX

모델 비교 결과에 parameter count도 기록한다.

37. 최종 모델 비교표
최종적으로 다음 형태의 표를 생성한다.

Model       Accuracy    Macro F1    Weighted F1    Params
MLP         ...         ...         ...            ...
LSTM        ...         ...         ...            ...
GRU         ...         ...         ...            ...
CNN1D       ...         ...         ...            ...
CNN_LSTM    ...         ...         ...            ...

이를:

results/model_comparison.csv

로 저장한다.

38. Class별 비교표
다음 형식으로 생성한다.

Model       Class     Precision    Recall    F1
MLP         Normal    ...          ...       ...
MLP         Fire      ...          ...       ...
MLP         Nofire    ...          ...       ...

LSTM        Normal    ...          ...       ...
LSTM        Fire      ...          ...       ...
LSTM        Nofire    ...          ...       ...

GRU         Normal    ...          ...       ...
GRU         Fire      ...          ...       ...
GRU         Nofire    ...          ...       ...

CNN1D       Normal    ...          ...       ...
CNN1D       Fire      ...          ...       ...
CNN1D       Nofire    ...          ...       ...

CNN_LSTM    Normal    ...          ...       ...
CNN_LSTM    Fire      ...          ...       ...
CNN_LSTM    Nofire    ...          ...       ...

저장:

results/class_comparison.csv

39. 모델별 Fire 성능 비교
별도의 표를 생성한다.

Model       Fire Precision    Fire Recall    Fire F1
MLP         ...               ...            ...
LSTM        ...               ...            ...
GRU         ...               ...            ...
CNN1D       ...               ...            ...
CNN_LSTM    ...               ...            ...

저장:

results/fire_detection_comparison.csv

40. 모델 성능을 임의로 해석하지 말 것
최종 결과를 보고할 때 단순히:

CNN-LSTM이 가장 좋다

라고 결론 내리지 말고 실제 metric을 기반으로 비교한다.

특히 다음을 함께 고려한다.

Macro F1
Fire Recall
Fire F1
Weighted F1
Accuracy
Parameter count
Training time

어떤 모델이 특정 metric에서 높은지 객관적으로 기록한다.

41. Training time
각 모델의 학습 시간을 측정한다.

예:

MLP       training time: ...
LSTM      training time: ...
GRU       training time: ...
CNN1D     training time: ...
CNN_LSTM  training time: ...

가능하면:

training_time_sec
inference_time_sec

도 기록한다.

42. 최종 결과 파일
최소한 다음 파일들이 생성되어야 한다.

results/
├── model_comparison.csv
├── class_comparison.csv
├── fire_detection_comparison.csv
├── metrics.csv
├── confusion_matrix_mlp.png
├── confusion_matrix_lstm.png
├── confusion_matrix_gru.png
├── confusion_matrix_cnn1d.png
├── confusion_matrix_cnn_lstm.png
├── training_curve_mlp.png
├── training_curve_lstm.png
├── training_curve_gru.png
├── training_curve_cnn1d.png
├── training_curve_cnn_lstm.png
└── scenario_predictions/
