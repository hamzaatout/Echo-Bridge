# Echo-Bridge: Bridging Voices and Signs Through AI for Enhanced Accessibility

## Table of Contents
- [Introduction](#introduction)
- [Motivation](#motivation)
- [Features](#features)
- [Methodology](#methodology)
  - [Speech-to-Text](#speech-to-text)
  - [Sign-to-Text](#sign-to-text)
- [Datasets](#datasets)
- [Results](#results)
- [Ethical Considerations](#ethical-considerations)
- [Future Work](#future-work)
- [Installation](#installation)
- [Usage](#usage)
- [Contributors](#contributors)
- [References](#references)

---

## Introduction
Echo-Bridge is an AI-driven system designed to bridge the communication gap between individuals with speech, hearing, or vision impairments and the broader community. The project incorporates:
1. A Speech-to-Text model.
2. A Sign-to-Text model using computer vision.

Future work includes extending functionality to support Text-to-Speech and Text-to-Sign Language translation.

## Motivation
Accessibility tools for individuals with disabilities are often scarce and expensive. Echo-Bridge aims to provide an affordable and effective solution to facilitate communication, leveraging AI models for transforming speech and sign language into text.

## Features
- **Speech-to-Text**: Converts audio input into text using a Connectionist Temporal Classification (CTC) loss-based model.
- **Sign-to-Text**: Recognizes sign language from videos and converts it to text using CNN+LSTM and 3D-CNN architectures.

## Methodology

### Speech-to-Text
#### Methodology:
- Utilizes the CTC algorithm for sequence-to-sequence learning.
- Transforms audio into mel-spectrograms for feature extraction.

#### Model Architecture:
- A Long Short-Term Memory (LSTM) network with bidirectional layers and softmax activation.

#### Challenges:
- The Mozilla Common Voice dataset was suboptimal due to longer audio sequences.
- Vanilla RNN models failed due to poor performance on sequential data.

---

### Sign-to-Text
#### Methodology:
- Two approaches:
  1. CNN+LSTM: Combines convolutional layers for spatial feature extraction and LSTMs for temporal dependencies.
  2. 3D-CNN: Accounts for spatio-temporal dependencies simultaneously by stacking video frames.

#### Dataset:
- Modified National Institute of Standards and Technology (MNIST) dataset for American Sign Language (ASL).
- Word-Level American Sign Language (WLASL-100) dataset for word-level recognition.

#### Challenges:
- Computational resource limitations led to overfitting and inconsistent results on validation and test datasets.

## Datasets
1. **Speech-to-Text**:
   - TIMIT dataset (initially attempted but not feasible due to API limitations).
   - Mozilla Common Voice dataset (used despite longer sequences).
2. **Sign-to-Text**:
   - MNIST (alphabet-level recognition).
   - WLASL-100 (word-level recognition).

## Results
### Speech-to-Text:
- LSTM: Loss remained high due to suboptimal dataset.
- Vanilla RNN: Poor performance with higher average loss.

### Sign-to-Text:
- CNN+LSTM:
  - Training Accuracy: 37.5%
  - Validation Accuracy: 14.29%
  - Test Accuracy: 25%
- 3D-CNN:
  - Training Accuracy: 78.23%
  - Validation Accuracy: 17.14%
  - Test Accuracy: 16%

## Ethical Considerations
- Prohibits the use of profanity to foster positive communication.
- Mitigates risks of catastrophic recognition errors through rigorous testing and ethical oversight.

## Future Work
- Improve model accuracy using transformer-based architectures.
- Integrate Text-to-Speech and Text-to-Sign Language functionalities.
- Enhance dataset preprocessing and expand hardware capabilities.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/KarimHabbal1/EECE490_FinalProject.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Follow the setup instructions for the respective models in the repository.

## Usage
1. **Speech-to-Text**:
   - Provide an audio file for transcription.
   - Run the `speech_to_text.py` script.

2. **Sign-to-Text**:
   - Input a video file containing sign language.
   - Run the `sign_to_text.py` script.

## Contributors
- **Karim Habbal**
- **Hamza Atout**
- **Khaled Nasser**
- **Khaled Ammoura**

## References
1. Graves, A., et al. "Connectionist temporal classification: Labelling unsegmented sequence data with recurrent neural networks." Proceedings of the 23rd International Conference on Machine Learning. 2006.
2. Li, D., et al. "Word-level deep sign language recognition from video: A new large-scale dataset and methods comparison." Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. 2020.
3. Rakhmatov, D., and Breccia, A. "Real-time sign language recognition with convolutional neural networks." 2024.
