# SPDX-FileCopyrightText: 2026 VIA Technologies, Inc. VISVIA Software Team
#
# SPDX-License-Identifier: MIT

"""
CLAP Soft Label Generation
Runs the CLAP teacher model (154M params) on all training audio
to generate soft probability labels for knowledge distillation.

Requires GPU. Run on DGX Spark or similar.
Usage: python3 generate_clap_labels.py
"""

import torch
import torchaudio
import numpy as np
import wave
import os
import time
from transformers import ClapModel, ClapProcessor
import kagglehub
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device}")

print("\nLoading CLAP teacher model (154M parameters)...")
processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused")
clap_model = ClapModel.from_pretrained("laion/clap-htsat-fused").to(device)
clap_model.eval()
print("CLAP loaded!")

# Best prompts found via systematic evaluation
SCREAM_PROMPT = "woman or man screaming loudly in fear"
NOT_SCREAM_PROMPT = "calm music or normal conversation"
PROMPTS = [SCREAM_PROMPT, NOT_SCREAM_PROMPT]

print("\nDownloading Kaggle dataset...")
path = kagglehub.dataset_download("whats2000/human-screaming-detection-dataset")
print(f"Dataset at: {path}")


def load_wav(p):
    try:
        with wave.open(p, 'rb') as f:
            n = f.getnchannels()
            sr = f.getframerate()
            audio = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
        if n == 2:
            audio = audio[::2]
        return audio.astype(np.float32) / 32768.0, sr
    except Exception:
        return None, None


def get_clap_score(audio, sr):
    """Get scream probability from CLAP teacher model."""
    try:
        audio_tensor = torch.tensor(audio).unsqueeze(0)
        if sr != 48000:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, 48000)
        audio_48k = audio_tensor.squeeze().numpy()
        inputs = processor(
            audio=audio_48k, text=PROMPTS,
            return_tensors="pt", padding=True, sampling_rate=48000)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = clap_model(**inputs)
            probs = torch.softmax(outputs.logits_per_audio, dim=-1)
        return float(probs[0][0].cpu())
    except Exception:
        return None


# Time estimate
audio, sr = load_wav(os.path.join(path, 'Screaming',
                      os.listdir(os.path.join(path, 'Screaming'))[0]))
start = time.time()
_ = get_clap_score(audio, sr)
elapsed = time.time() - start
print(f"\nTime per file: {elapsed:.3f}s")
print(f"Estimated total for 3,493 files: {3493 * elapsed / 60:.1f} minutes")

# Generate soft labels
soft_embeddings, y_hard, y_soft, skipped = [], [], [], 0

for label, folder in [(1, 'Screaming'), (0, 'NotScreaming')]:
    folder_path = os.path.join(path, folder)
    files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
    print(f"\nProcessing {folder}: {len(files)} files")
    for fname in tqdm(files):
        audio, sr = load_wav(os.path.join(folder_path, fname))
        if audio is None or len(audio) < 1000:
            skipped += 1
            continue
        soft = get_clap_score(audio, sr)
        if soft is None:
            skipped += 1
            continue
        y_hard.append(label)
        y_soft.append(soft)

y_hard = np.array(y_hard)
y_soft = np.array(y_soft)

print(f"\nGenerated {len(y_hard)} samples (skipped {skipped})")
print(f"Scream mean:    {y_soft[y_hard==1].mean():.4f}")
print(f"NonScream mean: {y_soft[y_hard==0].mean():.4f}")
print(f"Separation:     {y_soft[y_hard==1].mean() - y_soft[y_hard==0].mean():.4f}")

np.save('y_hard_clap.npy', y_hard)
np.save('y_soft_clap.npy', y_soft)
print("\nSaved y_hard_clap.npy and y_soft_clap.npy")
