<!--
SPDX-FileCopyrightText: VIA Technologies, Inc. VISVIA Software Team
SPDX-License-Identifier: MIT
-->

# Scream Detection System — VIA VAB-5000 (MTK MDLA 3.0 NPU)

> **This project is a summer internship project developed under the [VIA Technologies](https://www.viatech.com) internship program, VISVIA Software Team. All rights reserved © VIA Technologies, Inc.**

A real-time scream and distress detection system for public safety monitoring in transit environments (subways, buses). Runs on the **VIA VAB-5000** edge AI board powered by the **MediaTek Genio 700 SoC**, with inference accelerated by the **MDLA 3.0 NPU**.

![Live Demo](assets/demo_screenshot.png)

---

## Overview

This project detects screams and cries for help in public spaces using a two-stage AI pipeline:

```
USB Microphone → CPU: YAMNet embedding → CPU/NPU: Custom classifier → Web Dashboard
```

The live dashboard runs in any browser and updates in real time using Chart.js and HTML5 Canvas — no image generation overhead, sub-second latency from audio input to display.

| Component | Details |
|-----------|---------|
| **Board** | VIA VAB-5000 (MediaTek Genio 700) |
| **NPU** | MDLA 3.0 (compiled via MediaTek NeuroPilot `ncc-tflite`) |
| **Feature extractor** | YAMNet (frozen, CPU) → 1024D embedding |
| **Classifier** | Custom Dense network (CPU TFLite or NPU DLA) |
| **Teacher model** | CLAP (laion/clap-htsat-fused, 154M params) for knowledge distillation |
| **Training data** | 96,418 samples (Kaggle + AudioSet + augmented) |
| **Detection threshold** | 0.85 (configurable) |
| **Web dashboard** | Flask + Chart.js + HTML5 Canvas (native browser rendering) |

---

## Architecture

### Model Pipeline

```
Raw Audio (48kHz stereo, USB mic)
        ↓
  Left channel → resample to 16kHz mono
        ↓
  YAMNet backbone (frozen, CPU)
  → 1024-dimensional embedding
        ↓
  Custom classifier (CPU TFLite or NPU DLA):
    Dense(512) → Dropout(0.4)
    Dense(256) → Dropout(0.3)
    Dense(64)  → Dropout(0.2)
    Dense(1)   → Sigmoid
        ↓
  Scream probability (0–1)
        ↓
  Flask API → Browser (Chart.js + Canvas)
```

### Web Dashboard Architecture

Previous versions served matplotlib PNG images from the server, causing 2–3 second rendering delays. The current architecture eliminates this bottleneck entirely:

```
Flask server                    Browser
───────────                     ───────
record_thread  ─→  audio_queue
process_thread ─→  /data JSON   ─→  Chart.js line chart  (score over time)
                   (every 500ms) ─→  HTML5 Canvas         (spectrogram)
                                ─→  Status cards          (detection state)
```

- **Server** runs two threads: one records audio, one runs inference
- **`/data` endpoint** returns a tiny JSON payload (~2KB) with score, time, history, and one spectrogram column
- **Browser** renders everything natively — Chart.js for the line chart, Canvas for the spectrogram
- **Update rate**: ~500ms poll interval, rendering is instant on client side
- **Spectrogram** is built incrementally — one frequency column per update, masked below the score line for the merged visualization

### Knowledge Distillation

The classifier was trained using knowledge distillation from **CLAP** (Contrastive Language-Audio Pretraining, 154M parameters):

- **Teacher**: `laion/clap-htsat-fused` — generates soft probability labels using the prompt *"woman or man screaming loudly in fear"*
- **Student**: Our small custom classifier (deployed on CPU/NPU)
- **Loss**: Combined hard label loss (BCE) + soft label loss (KL divergence, temperature=2.0)
- **Soft label separation**: 0.88 (scream) vs 0.34 (non-scream) — 0.54 separation

### Why a Split Pipeline?

YAMNet's TFLite model uses `kTfLiteComplex64` operations for STFT preprocessing which the MTK MDLA 3.0 NPU does not support. The solution is to split:

- **CPU**: Raw audio → STFT → mel spectrogram → 1024D embedding (YAMNet)
- **CPU/NPU**: 1024D embedding → scream probability (custom classifier)

---

## Model Performance

### Benchmark Metrics (96,418 sample test set)

| Metric | Value |
|--------|-------|
| Precision | 0.603 |
| Recall | 0.781 |
| F1-Score | 0.681 |
| ROC-AUC | 0.892 |
| Test samples | 14,463 |

### Threshold Analysis

| Threshold | Precision | Recall | F1 |
|-----------|-----------|--------|-----|
| 0.3 | 0.523 | 0.867 | 0.653 |
| 0.5 | 0.603 | 0.781 | 0.681 |
| 0.7 | 0.712 | 0.643 | 0.676 |
| **0.85** | **0.740** | **0.610** | **0.669** |

### Real-World Test Results

| Audio scenario | Score | Detected |
|----------------|-------|----------|
| Pure scream (no background) | 0.97 | ✅ |
| Scream over café music | 0.99 | ✅ |
| Scream over singing/music | 0.98 | ✅ |
| YAMNet baseline — café scream | 0.082 | ❌ |

The custom model dramatically outperforms baseline YAMNet on noisy real-world scenarios.

---

## Dataset

| Source | Samples | Label type |
|--------|---------|-----------|
| Kaggle Human Screaming Detection Dataset | 3,493 | CLAP soft labels |
| Google AudioSet scream clips (384 files) | ~5,177 windows | CLAP soft labels |
| Google AudioSet background/noise clips (800 files) | ~8,000 windows | CLAP soft labels |
| Google AudioSet vocal/speech clips (800 files) | ~8,000 windows | CLAP soft labels |
| Augmented noisy screams (scream × background mix, SNR 0.3/0.5/0.7) | ~2,586 | CLAP soft labels |
| **Total** | **96,418** | |

All samples have CLAP-generated soft probability labels for knowledge distillation training.

---

## Hardware Requirements

- **VIA VAB-5000** (MediaTek Genio 700 SoC)
- **OS**: Debian 12 EVK image
- **USB Microphone** — 48kHz stereo (tested with Alcor Micro USB Audio Device)
- **Network** — WiFi or Ethernet for web dashboard access

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/jordanl8-cyber/BATHNet.git
cd BATHNet
```

### 2. Install dependencies on the board

```bash
pip3 install tensorflow flask scipy matplotlib --break-system-packages
```

### 3. Place model files

Download and place these files in `/home/jordanlee/`:

| File | Description |
|------|-------------|
| `yamnet_embedding.tflite` | YAMNet embedding extractor (CPU) |
| `scream_detector_v13.tflite` | Custom scream classifier (CPU) |
| `scream_detector_v13.dla` | NPU-compiled classifier (MDLA 3.0) |

Compile the DLA file if needed:
```bash
sudo ncc-tflite --arch mdla3.0 --relax-fp32 \
  -d scream_detector_v13.dla scream_detector_v13.tflite
```

### 4. Set up the launcher

```bash
chmod +x scripts/scream_live
sudo cp scripts/scream_live /usr/local/bin/scream_live
```

### 5. Configure microphone

Ensure your USB microphone is recognized:
```bash
sudo modprobe snd-usb-audio
arecord -l
```

The mic must be in the `audio` group:
```bash
sudo usermod -aG audio $USER
```

---

## Usage

### Live Web Dashboard

```bash
scream_live
```

Open **http://jordan-vab.local:5000** (or the board's IP:5000) in any browser.

The dashboard shows:
- **Detection status** — clean pill indicator (no detection / scream detected)
- **Confidence score** — live numeric value, updates every ~1 second
- **Session time** — elapsed time since launch
- **Probability graph** — white line showing score over 30-second rolling window
- **Frequency spectrogram** — inferno colormap fills area below the score line, showing real-time frequency content
- Score labels on each data point, threshold line in amber

### 5-Second Analysis Demo

```bash
python3 scripts/scream_analysis_demo.py
```

Records 5 seconds (with 3-second countdown) and saves 5 analysis graphs to a timestamped folder:
1. Raw audio waveform
2. Merged scream probability + spectrogram
3. Spectrogram
4. RMS energy over time
5. Zero crossing rate

### Test on a WAV File

```bash
python3 scripts/test_audio_file.py path/to/audio.wav
```

---

## Configuration

Edit `scripts/live_scream_web.py` to change:

```python
THRESHOLD = 0.85       # Detection threshold (0–1)
WINDOW_SECONDS = 30    # Rolling window width on chart
RECORD_SECONDS = 1     # Audio chunk duration
SPEC_BINS = 64         # Frequency bins for spectrogram
```

---

## Repository Structure

```
BATHNet/
├── README.md
├── requirements.txt
├── LICENSE
├── LICENSES/
│   └── MIT.txt                      ← REUSE-compliant license file
├── .reuse/
│   └── dep5                         ← REUSE dependency info
├── scripts/
│   ├── scream_live                  ← Shell launcher
│   ├── live_scream_web.py           ← Live web dashboard (Flask + Chart.js)
│   ├── scream_analysis_demo.py      ← 5-second analysis with graphs
│   └── test_audio_file.py           ← Test on any WAV file
├── training/
│   ├── generate_clap_labels.py      ← CLAP teacher soft label generation
│   ├── train_v13.py                 ← Final model training (96K samples)
│   └── download_audioset.py         ← AudioSet data download
├── models/
│   └── .gitkeep                     ← Model files go here (not committed)
└── assets/
    └── demo_screenshot.png
```

---

## Training Your Own Model

### 1. Download data

```bash
# Kaggle dataset
pip install kagglehub
python3 -c "import kagglehub; kagglehub.dataset_download('whats2000/human-screaming-detection-dataset')"

# AudioSet clips
python3 training/download_audioset.py
```

### 2. Generate CLAP soft labels (requires GPU)

```bash
python3 training/generate_clap_labels.py
```

Runs the CLAP teacher model (154M parameters) on all training audio. Takes ~10 minutes on a GPU.

### 3. Train the classifier

```bash
python3 training/train_v13.py
```

~5 minutes on NVIDIA GPU.

### 4. Compile for NPU

```bash
sudo ncc-tflite --arch mdla3.0 --relax-fp32 \
  -d models/scream_detector_v13.dla models/scream_detector_v13.tflite
```

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

REUSE compliance: all files carry proper `SPDX-FileCopyrightText` and `SPDX-License-Identifier` headers per [REUSE Specification v3.3](https://reuse.software/).

Copyright © VIA Technologies, Inc. VISVIA Software Team

---

## Acknowledgements

- [YAMNet](https://tfhub.dev/google/yamnet/1) — Google's audio event classifier
- [CLAP](https://huggingface.co/laion/clap-htsat-fused) — LAION's contrastive language-audio model
- [Human Screaming Detection Dataset](https://www.kaggle.com/datasets/whats2000/human-screaming-detection-dataset) — Kaggle
- [AudioSet](https://research.google.com/audioset/) — Google Research
- [Chart.js](https://www.chartjs.org/) — JavaScript charting library
- [VIA Technologies](https://www.viatech.com) — VAB-5000 hardware platform
- MediaTek NeuroPilot SDK — NPU compilation toolchain
