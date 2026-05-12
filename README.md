Here is a **professional, polished `README.md`** with proper structure, clear technical wording, and **complete commands section** (interview/placement-ready).

You can directly use this in your GitHub repo.

---

#  ASR Benchmarking System

### Whisper vs Deepgram – Multilingual Speech Evaluation Pipeline

---

##  Project Overview

This project is a **speech recognition benchmarking system** designed to evaluate and compare two state-of-the-art ASR (Automatic Speech Recognition) models:

 **OpenAI Whisper (large-v3)**
 **Deepgram Nova-3 API**

The system processes real-world audio containing **Hindi, Kannada, English, and Hinglish speech**, and evaluates model performance using standard NLP metrics and analytical visualizations.

---

##  Objectives

* Benchmark ASR models using **Word Error Rate (WER)**
* Evaluate multilingual transcription performance
* Analyze robustness under noisy and clean audio conditions
* Measure **locality (Bangalore location) detection accuracy**
* Perform structured **error classification and failure analysis**
* Generate **visual insights for model comparison**

---

##  System Architecture

```text
Audio Dataset
     ↓
Audio Preprocessing (Normalization + Resampling)
     ↓
ASR Inference Layer
   ├── Whisper (local model)
   └── Deepgram (cloud API)
     ↓
Transcript Generation
     ↓
Ground Truth Comparison
     ↓
Evaluation Engine
   ├── WER Calculation (JIWER)
   ├── Language Analysis
   ├── Noise Analysis
   ├── Locality Detection
   └── Error Categorization
     ↓
Visualization Layer
   ├── WER Comparison Chart
   └── Locality Accuracy Pie Chart
```

---

##  Project Structure

```text
asr-shootout/
│
├── audio/                         # Input audio files
├── clean_audio/                  # Preprocessed audio files
├── outputs/                      # Generated results
│   ├── whisper_results.csv
│   ├── deepgram_nova3_results.csv
│   ├── final_evaluation.csv
│   ├── error_analysis.csv
│   ├── wer_comparison.png
│   └── locality_pie.png
│
├── whisper_test.py              # Whisper inference pipeline
├── deepgram_test.py             # Deepgram API pipeline
├── evaluate.py                  # Evaluation + analytics engine
├── ground_truth.csv            # Reference transcripts
└── README.md
```

---

##  Tech Stack

* Python 3.12
* Whisper (faster-whisper)
* Deepgram API (Nova-3)
* Pandas
* JiWER (WER computation)
* Matplotlib (visualization)
* Pydub (audio preprocessing)

---

##  Evaluation Metrics

### 1. Word Error Rate (WER)

Measures transcription accuracy at word level.

### 2. Language-wise Performance

Evaluates performance across:

* English
* Hindi
* Kannada
* Mixed (Hinglish)

### 3. Locality Detection Accuracy

Evaluates correct recognition of Bangalore locality names such as:
Koramangala, Indiranagar, Whitefield, HSR Layout, etc.

### 4. Noise Robustness Analysis

Compares performance under:

* Clean audio
* Medium noise
* High noise conditions

---




If requirements file is not available:

```
pip install faster-whisper pandas jiwer matplotlib pydub deepgram-sdk langdetect
```

---

##  How to Run the Project

### 1. Run Whisper Transcription

```bash
python whisper_test.py
```

Generates:

* `outputs/whisper_results.csv`

---

### 2. Run Deepgram Transcription

```
python deepgram_test.py
```

Generates:

* `outputs/deepgram_nova3_results.csv`

---

### 3. Run Evaluation & Analysis

```
python evaluate.py
```

Generates:

* WER comparison report
* Language analysis
* Error analysis table
* Visualization graphs

---

## Output Visualizations

Generated automatically:

 **WER Comparison Chart**

  * Whisper vs Deepgram performance

 **Locality Detection Accuracy Pie Chart**

  * Correct vs Incorrect location detection

---

## 🔍 Key Results

| Model    | Average WER |
| -------- | ----------- |
| Whisper  | ~0.98       |
| Deepgram | ~0.80       |

---

##  Key Insights

* Whisper performs better for **Indian multilingual speech (Hindi + Kannada + Hinglish)**
* Deepgram performs better for **clean English audio**
* Noise significantly impacts ASR accuracy
* Location entity recognition is a major failure area

---

##  Error Analysis Summary

Observed issues:

* Phonetic mismatch (e.g., "Koramangala" → "Kormangala")
* Language blending errors in Hinglish speech
* Partial transcription in noisy environments
* Missing or truncated words in low-quality audio

---

##  Commands Summary

```
# Install dependencies
pip install -r requirements.txt

# Run Whisper ASR
python whisper_test.py

# Run Deepgram ASR
python deepgram_test.py

# Run evaluation pipeline
python evaluate.py
``

##  Conclusion

This project demonstrates a complete **real-world ASR benchmarking pipeline**, including:

* Multi-model comparison
* Multilingual speech handling
* Error analysis framework
* Noise robustness evaluation
* Production-style ML evaluation system

 Recommended production approach:
**Hybrid ASR System (Whisper + Deepgram + Post-processing layer)**

---

## Future Enhancements

* Real-time speech-to-text web app (Streamlit)
* Advanced noise suppression (RNNoise / spectral filtering)
* Named Entity Recognition for location correction
* Ensemble-based ASR voting system
* Live microphone transcription system


Note:
FFmpeg must be installed separately for audio processing.


Good — this is exactly the kind of small detail interviewers like because it shows you understand real-world setup issues.

Here’s a **professional README-ready section** you can directly paste:

---

##  FFmpeg Installation (Required)

This project uses **pydub** for audio preprocessing, which depends on **FFmpeg** for decoding and converting audio files.

FFmpeg is **not installed via pip**, so it must be installed separately.

---

###  Windows Installation

#### Option 1: Using Chocolatey (Recommended)

```bash
choco install ffmpeg
```

---

#### Option 2: Manual Installation

1. Download FFmpeg from official site:
    [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)

2. Extract the zip file

3. Add `bin` folder to system PATH:

   ```
   C:\ffmpeg\bin
   ```

4. Verify installation:

```
ffmpeg -version
```

---

### Linux Installation

```
sudo apt update
sudo apt install ffmpeg
```

---

###  Mac Installation

```
brew install ffmpeg
```

---

##  Verification

Run this command to confirm installation:

```
ffmpeg -version
```

If installed correctly, you will see version details.

---

##  Important Note

```text
FFmpeg is required for audio loading and conversion in pydub.
Without FFmpeg, audio preprocessing and Whisper transcription will fail.
```



