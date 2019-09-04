# Simple ResNet for CIFAR10 with TensorFlow

## 0. CIFAR10 Dataset Layout
### Train data: 50000 samples
* Input shape: (32, 32, 3)
* Label shape: (10)
### Train data: 10000 samples
* Input shape: (32, 32, 3)
* Label shape: (10)

## 1. Model Structure

### Input
| Layer                 | Output Shape    | Connected to                        |
|-----------------------|-----------------|-------------------------------------|
| input_1               | (?, 32, 32, 3)  |                                     |

### CNN Block 1
| Layer                 | Output Shape    | Connected to                        |
|-----------------------|-----------------|-------------------------------------|
| conv2d_1              | (?, 32, 32, 16) | input_1                             |
| batch_normalization_1 | (?, 32, 32, 16) | conv2d_1                            |
| activation_1          | (?, 32, 32, 16) | batch_normalization_1               |
| conv2d_2              | (?, 32, 32, 16) | activation_1                        |
| batch_normalization_2 | (?, 32, 32, 16) | conv2d_2                            |
| activation_2          | (?, 32, 32, 16) | batch_normalization_2               |
| conv2d_3              | (?, 32, 32, 16) | activation_2                        |
| batch_normalization_3 | (?, 32, 32, 16) | conv2d_3                            |
| add_1                 | (?, 32, 32, 16) | activation_1, batch_normalization_3 |
| activation_3          | (?, 32, 32, 16) | add_1                               |
| max_pool_1            | (?, 16, 16, 16) | activation_3                        |

### CNN Block 2
| Layer                 | Output Shape    | Connected to                        |
|-----------------------|-----------------|-------------------------------------|
| conv2d_4              | (?, 16, 16, 32) | max_pool_1                          |
| batch_normalization_4 | (?, 16, 16, 32) | conv2d_4                            |
| activation_4          | (?, 16, 16, 32) | batch_normalization_4               |
| conv2d_5              | (?, 16, 16, 32) | activation_4                        |
| batch_normalization_5 | (?, 16, 16, 32) | conv2d_5                            |
| activation_5          | (?, 16, 16, 32) | batch_normalization_5               |
| conv2d_6              | (?, 16, 16, 32) | activation_5                        |
| batch_normalization_6 | (?, 16, 16, 32) | conv2d_6                            |
| add_2                 | (?, 16, 16, 32) | activation_4, batch_normalization_6 |
| activation_6          | (?, 16, 16, 32) | add_2                               |
| max_pool_2            | (?, 8, 8, 32)   | activation_6                        |

### CNN Block 3
| Layer                 | Output Shape    | Connected to                        |
|-----------------------|-----------------|-------------------------------------|
| conv2d_7              | (?, 8, 8, 64)   | max_pool_2                          |
| batch_normalization_7 | (?, 8, 8, 64)   | conv2d_7                            |
| activation_7          | (?, 8, 8, 64)   | batch_normalization_7               |
| conv2d_8              | (?, 8, 8, 64)   | activation_7                        |
| batch_normalization_8 | (?, 8, 8, 64)   | conv2d_8                            |
| activation_8          | (?, 8, 8, 64)   | batch_normalization_8               |
| conv2d_9              | (?, 8, 8, 64)   | activation_8                        |
| batch_normalization_9 | (?, 8, 8, 64)   | conv2d_9                            |
| add_3                 | (?, 8, 8, 64)   | activation_7, batch_normalization_9 |
| activation_9          | (?, 8, 8, 64)   | add_3                               |
| avg_pool_1            | (?, 1, 1, 64)   | activation_9                        |

### Output
| Layer                 | Output Shape    | Connected to                        |
|-----------------------|-----------------|-------------------------------------|
| flatten_1             | (?, 64)         | avg_pool_1                          |
| dense_1               | (?, 10)         | flatten_1                           |
| softmax_1             | (?, 10)         | dense_1                             |

## 2. Model Train Result
![Train Plot](https://user-images.githubusercontent.com/2123763/64223743-01e93d80-cf10-11e9-9982-d96c75977ef5.png)
* Average train cost: **0.073** (at 30 epoch)
* Train accuracy: **0.9800** (at 30 epoch)
* Test accuracy: **0.8037** (at 30 epoch)
