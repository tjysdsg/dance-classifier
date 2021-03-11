# Temporal CNN

- Modified ResNet 101
- Color channel is the number of consecutive frames
- Added one fully-connected layer after the final layer, with output size equal to the number of dance categories and
  SoftMax activation
- Loss function: cross entropy
- Optimizer: SGD with 0.9 momentum
- Learning rate 0.001, using ReduceOnPlateau scheduler

# PoseCNN

- Layers:
    - Conv2d, size 3x3, stride 1, input channel 2, output channel 16
    - batch normalization
    - ReLU activation
    - Max Pooling, size 3x3, stride 1
    - Conv2d, size 3x3, stride 1, input channel 16, output channel 16
    - batch normalization
    - ReLU activation
    - Max Pooling, size 3x3, stride 1
    - Fully-connected, output size equal to number of dance categories, SoftMax activation
- Loss function: cross entropy
- Optimizer: SGD with 0.9 momentum
- Learning rate 0.0001, using ReduceOnPlateau scheduler

# LSTMRNN

- Layers:
    - LSTM, hidden size 256, dropout 0.2
    - LSTM, hidden size 256, dropout 0.2
    - Fully-connected, output size 1024
    - Fully-connected, output size equal to number of dance categories, SoftMax activation
- Loss function: cross entropy
- Optimizer: SGD with 0.9 momentum
- Learning rate 0.0001, using ReduceOnPlateau scheduler
