# TinyML Speech Command Classification on Arduino Nano 33 BLE Sense

A highly optimized deep learning model for real-time speech command recognition ("zero" vs "one") deployed on resource-constrained embedded hardware. Achieves 99%+ accuracy in just 7.5 KB.

## Project Overview

This project demonstrates end-to-end machine learning deployment on microcontrollers, from model design to hardware validation. The system classifies spoken commands using a custom CNN architecture optimized for both accuracy and efficiency.

**Key Achievements:**
-  **99.25% hardware accuracy** (validated on Arduino)
- 📦 **7.5 KB model size** (INT8 quantized)
- ⚡ **Real-time inference** on Cortex-M4F MCU
- 🚀 **49% improvement** over baseline (50% → 99%)

## 🏗️ Architecture

### Model Design Philosophy

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

**Key Design Choices:**

| Component | Baseline | Optimized | Rationale |
|-----------|----------|-----------|-----------|
| Convolution Layers | 2 | 6+ | Deeper feature extraction |
| Activation | ReLU | ReLU6 | Better quantization |
| Normalization | None | BatchNorm | Training stability |
| Pooling Strategy | Early MaxPool | Strided Conv | Preserves temporal patterns |
| Dense Layer | Large | Tiny (after GlobalMaxPool) | 90% parameter reduction |
| Regularization | None | Dropout | Generalization |

## 📊 Results

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

## 🔧 Technical Implementation

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

## 🚀 Quick Start

### Prerequisites

```bash
pip install tensorflow numpy
```

### Training

```python
# Train model
python train_model.py

# Convert to TFLite
python convert_to_tflite.py
```

### Deployment

```bash
# Generate C array
xxd -i model.tflite > model.cpp

# Flash to Arduino (using Arduino IDE)
# Upload Command_word.ino

# Test hardware
python Python_input_output.py
```

## 📁 Repository Structure

```
.
├── train_model.py              # Model training script
├── convert_to_tflite.py        # TFLite conversion
├── model.tflite                # Quantized model
├── model.cpp                   # Model C array
├── model.h                     # Model header
├── Command_word.ino            # Arduino inference code
├── output_handler.h            # Output utilities
├── Python_input_output.py      # Hardware testing script
├── report.pdf                  # Detailed technical report
└── README.md                   # This file
```

## 🔬 Key Innovations

1. **Strided Convolutions for Compression**
   - Reduces 16,000 samples to hundreds without information loss
   - Better than pooling for temporal patterns

2. **Global Max Pooling Replacement**
   - Eliminates expensive dense layers
   - 90% reduction in parameters

3. **Quantization-Optimized Architecture**
   - ReLU6 bounds activations → better INT8 mapping
   - BatchNorm → stable quantization calibration
   - Post-quantization accuracy *increases*

4. **Real-Time Validated**
   - Not just simulation — actual hardware testing
   - 99.25% accuracy proves correct deployment

## 📈 Comparison with Baseline

| Metric | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| Accuracy | 50% | 99.25% | +49.25% |
| Conv Layers | 2 | 6+ | +4 layers |
| Normalization | ❌ | ✅ BatchNorm | Stability |
| Activation | ReLU | ReLU6 | Quantization |
| Pooling | MaxPool | Strided Conv | Feature preservation |
| Dense Layer | Large | Minimal | -90% params |
| Quantization | Poor | Excellent | +0.12% post-quant |

## 🎓 Learning Outcomes

- Deep understanding of TinyML constraints and optimizations
- Practical quantization-aware architecture design
- End-to-end embedded ML deployment workflow
- Hardware validation and debugging techniques
- Trade-offs between accuracy, size, and latency

## 📝 License

This project was completed as part of an embedded machine learning course assignment.

## 🙏 Acknowledgments

- Speech Commands dataset
- TensorFlow Lite for Microcontrollers
- Arduino community

---

**Author:** [Your Name]  
**Course:** [Course Name]  
**Date:** [Submission Date]
