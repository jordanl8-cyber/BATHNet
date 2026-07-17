# SPDX-FileCopyrightText: 2026 VIA Technologies, Inc. VISVIA Software Team
#
# SPDX-License-Identifier: MIT

"""
5-Second Scream Analysis Demo
Records 5 seconds of audio and generates analysis graphs.

Usage: python3 scream_analysis_demo.py
Output: scream_analysis_YYYYMMDD_HHMMSS/ folder with 5 PNG graphs
"""

import numpy as np
import subprocess
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal
import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

THRESHOLD = 0.8
RECORD_SECONDS = 5
SAMPLE_RATE_HW = 48000
SAMPLE_RATE_MODEL = 16000
CHUNK_SAMPLES = 15600
HOP_SAMPLES = 8000

YAMNET_PATH = '/home/jordanlee/yamnet_embedding.tflite'
DLA_PATH = '/home/jordanlee/scream_detector_v13.dla'
MIC_DEVICE = 'plughw:Device,0'

print("="*55)
print("  Scream Analysis Demo - VIA VAB-5000")
print("  MTK Genio 700 MDLA 3.0 NPU")
print("="*55)

print("\nLoading models...")
yamnet = tf.lite.Interpreter(YAMNET_PATH)
yamnet.allocate_tensors()
yamnet_inp = yamnet.get_input_details()[0]
yamnet_out = yamnet.get_output_details()
print("Models loaded!")

def get_embedding(chunk):
    yamnet.resize_tensor_input(yamnet_inp['index'], [len(chunk)])
    yamnet.allocate_tensors()
    yamnet.set_tensor(yamnet_inp['index'], chunk)
    yamnet.invoke()
    return yamnet.get_tensor(yamnet_out[0]['index']).mean(axis=0)

def run_npu(embedding):
    embedding.reshape(1, 1024).astype(np.float32).tofile('/tmp/emb_input.bin')
    result = subprocess.run([
        'sudo', '/usr/sbin/neuronrt', '-m', 'hw',
        '-a', DLA_PATH,
        '-b', '100', '-i', '/tmp/emb_input.bin', '-o', '/tmp/npu_output.bin'
    ], capture_output=True, text=True)
    if result.returncode != 0:
        return 0.0
    output = np.fromfile('/tmp/npu_output.bin', dtype=np.float32)
    return float(output[0]) if len(output) > 0 else 0.0

# Countdown
print("\nGet ready to scream!")
time.sleep(1)
print("3...")
time.sleep(1)
print("2...")
time.sleep(1)
print("1... GO!")
time.sleep(0.5)
print(f"Recording {RECORD_SECONDS} seconds...")

result = subprocess.run([
    'arecord', '-D', MIC_DEVICE,
    '-c', '2', '-r', str(SAMPLE_RATE_HW),
    '-f', 'S16_LE', '-d', str(RECORD_SECONDS),
    '--quiet', '-'
], capture_output=True)

if result.returncode != 0 or len(result.stdout) == 0:
    print("ERROR: Could not record audio")
    exit(1)

print("Recording done! Running analysis...")

# Process audio
audio_raw = np.frombuffer(result.stdout, dtype=np.int16)
audio_mono = audio_raw[::2].astype(np.float32) / 32768.0
audio_16k = signal.resample_poly(audio_mono, 1, 3).astype(np.float32)
total_samples = len(audio_16k)
time_raw = np.linspace(0, RECORD_SECONDS, len(audio_mono))

# Run inference
window_times, scream_scores = [], []
for start in range(0, max(1, total_samples - CHUNK_SAMPLES), HOP_SAMPLES):
    chunk = audio_16k[start:start+CHUNK_SAMPLES]
    if len(chunk) < CHUNK_SAMPLES:
        chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))
    t = (start + CHUNK_SAMPLES/2) / SAMPLE_RATE_MODEL
    emb = get_embedding(chunk)
    score = run_npu(emb)
    window_times.append(t)
    scream_scores.append(score)
    print(f"  t={t:.1f}s: score={score:.3f} {'SCREAM!' if score > THRESHOLD else ''}")

window_times = np.array(window_times)
scream_scores = np.array(scream_scores)
above = scream_scores > THRESHOLD

# Spectrogram
f_spec, t_spec, Sxx = signal.spectrogram(
    audio_16k, fs=SAMPLE_RATE_MODEL, nperseg=512, noverlap=256)
Sxx_db = 10 * np.log10(Sxx + 1e-10)

# RMS energy
frame_size = 1600
rms_values, rms_times = [], []
for i in range(0, len(audio_16k) - frame_size, frame_size):
    rms_values.append(np.sqrt(np.mean(audio_16k[i:i+frame_size]**2)))
    rms_times.append((i + frame_size/2) / SAMPLE_RATE_MODEL)

# ZCR
zcr_values, zcr_times = [], []
for i in range(0, len(audio_16k) - frame_size, frame_size):
    zcr_values.append(np.mean(np.abs(np.diff(np.sign(audio_16k[i:i+frame_size])))) / 2)
    zcr_times.append((i + frame_size/2) / SAMPLE_RATE_MODEL)

# Create output folder
timestamp = time.strftime('%Y%m%d_%H%M%S')
out_dir = f'/home/jordanlee/scream_analysis_{timestamp}'
os.makedirs(out_dir, exist_ok=True)

def save_plot(fig, name):
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {name}")

detected_count = int(np.sum(above))
duration = total_samples / SAMPLE_RATE_MODEL

# Graph 1 — Waveform
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(time_raw, audio_mono, color='steelblue', linewidth=0.5, alpha=0.8)
for t, score in zip(window_times, scream_scores):
    if score > THRESHOLD:
        ax.axvspan(t - 0.5, t + 0.5, alpha=0.3, color='red')
ax.set_title('Raw Audio Waveform', fontsize=13, fontweight='bold')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude')
ax.set_xlim(0, RECORD_SECONDS)
ax.grid(True, alpha=0.3)
save_plot(fig, '1_waveform.png')

# Graph 2 — Merged scream probability + spectrogram
fig, ax = plt.subplots(figsize=(12, 5))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#0d1117')
freq_mask = f_spec <= 4000
f_scaled = f_spec[freq_mask] / 4000.0
Sxx_plot = np.clip(Sxx_db[freq_mask, :], -100, -30)
score_interp = np.interp(t_spec, window_times, scream_scores,
                         left=scream_scores[0], right=scream_scores[-1])
Sxx_masked = Sxx_plot.copy().astype(float)
for col_idx in range(len(t_spec)):
    cutoff_idx = np.searchsorted(f_scaled, score_interp[col_idx])
    Sxx_masked[cutoff_idx:, col_idx] = np.nan
ax.pcolormesh(t_spec, f_scaled, Sxx_masked, shading='gouraud',
              cmap='inferno', vmin=-100, vmax=-30, zorder=2)
ax.plot(window_times, scream_scores, color='white', linewidth=2.5, zorder=10)
ax.axhline(THRESHOLD, color='#ff8800', linestyle='--', linewidth=1.8,
           label=f'Threshold ({THRESHOLD})', zorder=8)
ax.scatter(window_times[above], scream_scores[above], color='#ff3333',
           s=120, zorder=12, label='Scream detected', edgecolors='white', linewidths=1.5)
ax.scatter(window_times[~above], scream_scores[~above], color='#33cc55',
           s=80, zorder=12, label='No scream', edgecolors='white', linewidths=1.2)
for t, s in zip(window_times, scream_scores):
    ax.text(t, s + 0.04, f'{s:.2f}', ha='center', va='bottom',
            fontsize=8, color='white', fontweight='bold', zorder=13)
ax.set_xlim(0, duration)
ax.set_ylim(0, 1.12)
ax.set_xlabel('Time (s)', color='white')
ax.set_ylabel('Scream Score / Frequency (0-4 kHz)', color='white')
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(['0.0\n(0 kHz)', '0.25\n(1 kHz)', '0.5\n(2 kHz)',
                    '0.75\n(3 kHz)', '1.0\n(4 kHz)'], color='white', fontsize=8)
ax.tick_params(colors='white')
ax.spines[:].set_color('#333344')
ax.legend(loc='upper right', facecolor='#1a1a2e', edgecolor='#555',
          labelcolor='white', fontsize=9)
ax.set_title('Scream Probability + Frequency Content', fontsize=13,
             fontweight='bold', color='white')
save_plot(fig, '2_scream_probability_merged.png')

# Graph 3 — Spectrogram
fig, ax = plt.subplots(figsize=(12, 4))
img = ax.pcolormesh(t_spec, f_spec/1000, Sxx_db, shading='gouraud', cmap='inferno')
ax.set_title('Spectrogram', fontsize=13, fontweight='bold')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Frequency (kHz)')
ax.set_ylim(0, 4)
plt.colorbar(img, ax=ax, label='Power (dB)')
save_plot(fig, '3_spectrogram.png')

# Graph 4 — RMS Energy
fig, ax = plt.subplots(figsize=(12, 4))
ax.fill_between(rms_times, rms_values, alpha=0.4, color='orange')
ax.plot(rms_times, rms_values, color='darkorange', linewidth=2)
ax.set_title('RMS Energy (Volume Over Time)', fontsize=13, fontweight='bold')
ax.set_xlabel('Time (s)')
ax.set_ylabel('RMS Amplitude')
ax.set_xlim(0, duration)
ax.grid(True, alpha=0.3)
save_plot(fig, '4_rms_energy.png')

# Graph 5 — ZCR
fig, ax = plt.subplots(figsize=(12, 4))
ax.fill_between(zcr_times, zcr_values, alpha=0.4, color='purple')
ax.plot(zcr_times, zcr_values, color='purple', linewidth=2)
ax.set_title('Zero Crossing Rate (High = Screamy/Noisy)', fontsize=13, fontweight='bold')
ax.set_xlabel('Time (s)')
ax.set_ylabel('ZCR')
ax.set_xlim(0, duration)
ax.grid(True, alpha=0.3)
save_plot(fig, '5_zero_crossing_rate.png')

print(f"\nAll graphs saved to: {out_dir}")
print(f"Result: {'SCREAM DETECTED' if detected_count > 0 else 'No scream detected'}")
print(f"Max score: {np.max(scream_scores):.4f} | Threshold: {THRESHOLD}")
