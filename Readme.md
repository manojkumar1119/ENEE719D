# TinyML Speech Command Classification on Arduino Nano 33 BLE Sense

A highly optimized deep learning model for real-time speech command recognition ("zero" vs "one") deployed on resource-constrained embedded hardware. Achieves 99%+ accuracy in just 7.5 KB.

## Project Overview

This project demonstrates end-to-end machine learning deployment on microcontrollers, from model design to hardware validation. The system classifies spoken commands using a custom CNN architecture optimized for both accuracy and efficiency.

**Key Achievements:**
-  **99.25% hardware accuracy** (validated on Arduino)
-  **7.5 KB model size** (INT8 quantized)
-  **Real-time inference** on Cortex-M4F MCU

## Architecture

### Model Design

Completely redesigned from baseline with focus on:
- Temporal feature extraction via strided 1D convolutions
- Quantization-friendly operations (ReLU6, BatchNorm)
- Parameter efficiency through Global Max Pooling
- Regularization to prevent overfitting

### Network Architecture

```
Input (1, 16000, 1)
    ↓
[Conv2D(1×K) → BatchNorm → ReLU6] × N blocks (strided)
    ↓
Global Max Pooling
    ↓
Dropout(0.5)
    ↓
Dense(2) + Softmax
    ↓
Output: [P(zero), P(one)]
```

## Results

### Accuracy Comparison

| Platform | Accuracy | Model Size |
|----------|----------|------------|
| TensorFlow (Float32) | 99.00% | - |
| TFLite (Float32) | 99.00% | 10,108 bytes |
| TFLite (INT8 Quantized) | 99.12% | **7,592 bytes** |
| Arduino Hardware | **99.25%** | 7,592 bytes |

### Performance Gains

- **Accuracy improvement:** 50% → 99.25% (+49.25%)
- **Model compression:** ~80% reduction from baseline
- **Hardware validation:** Matches quantized accuracy

## Technical Implementation

### 1. Data Preprocessing

```python
# Audio signals converted to temporal arrays
# Shape: (batch, 16000, 1) for Conv2D with [1, K] kernels
# Normalized for quantization calibration
```

### 2. Model Training

- **Optimizer:** Adam
- **Loss:** Categorical Cross-Entropy
- **Regularization:** Dropout (0.5) + BatchNormalization
- **Early Stopping:** Monitoring validation accuracy
- **Final Training Accuracy:** 99%

### 3. Quantization Strategy

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()
```

**Result:** 7,592 bytes with improved accuracy (quantization benefited from ReLU6/BatchNorm)

### 4. Arduino Deployment

**Hardware:** Arduino Nano 33 BLE Sense (Cortex-M4F @ 64MHz, 256KB RAM)

**Conversion to C Array:**
```bash
xxd -i model.tflite > model.cpp
```

**Files Structure:**
```
├── model.cpp          # Quantized model as C array
├── model.h            # Header exposing g_model[] and g_model_len
├── Command_word.ino   # Main inference loop
└── output_handler.h   # Serial output interface
```

**Memory Allocation:**
- Model stored in flash memory (program space)
- Tensor arena sized for 256KB RAM constraint
- No dynamic allocation during inference

### 5. Hardware Validation

Testing via `Python_input_output.py`:
1. Python sends audio features via serial
2. Arduino runs TFLite Micro inference
3. Arduino returns predicted label
4. Python calculates overall accuracy

**Validated accuracy: 99.254%**

##  Quick Start

### Prerequisites

```bash
pip install tensorflow numpy
```

## Repository Structure

```
.
├── model.tflite                # Quantized model
├── model.cpp                   # Model C array
├── model.h                     # Model header
├── Command_word.ino            # Arduino inference code
├── output_handler.h            # Output utilities
├── report.pdf                  # Detailed technical report

```

## Learning Outcomes

- Deep understanding of TinyML constraints and optimizations
- Practical quantization-aware architecture design
- End-to-end embedded ML deployment workflow
- Hardware validation and debugging techniques
- Trade-offs between accuracy, size, and latency

