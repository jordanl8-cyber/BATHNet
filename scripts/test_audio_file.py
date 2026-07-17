# SPDX-FileCopyrightText: 2026 VIA Technologies, Inc. VISVIA Software Team
#
# SPDX-License-Identifier: MIT

"""
Test scream detection on a WAV file.

Usage: python3 test_audio_file.py path/to/audio.wav
"""

import sys
import numpy as np
import wave
import subprocess
import tensorflow as tf
from scipy import signal
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

THRESHOLD = 0.8
SAMPLE_RATE_MODEL = 16000
CHUNK_SAMPLES = 15600
HOP_SAMPLES = 8000

YAMNET_PATH = '/home/jordanlee/yamnet_embedding.tflite'
DLA_PATH = '/home/jordanlee/scream_detector_v13.dla'

if len(sys.argv) < 2:
    print("Usage: python3 test_audio_file.py path/to/audio.wav")
    sys.exit(1)

wav_path = sys.argv[1]

print(f"Loading models...")
yamnet = tf.lite.Interpreter(YAMNET_PATH)
yamnet.allocate_tensors()
yamnet_inp = yamnet.get_input_details()[0]
yamnet_out = yamnet.get_output_details()

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

print(f"Testing: {wav_path}\n")

with wave.open(wav_path, 'rb') as f:
    n_channels = f.getnchannels()
    sr = f.getframerate()
    audio = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)

if n_channels == 2:
    audio = audio[::2]
audio = audio.astype(np.float32) / 32768.0

if sr != SAMPLE_RATE_MODEL:
    audio = signal.resample_poly(audio, SAMPLE_RATE_MODEL, sr).astype(np.float32)

total_samples = len(audio)
duration = total_samples / SAMPLE_RATE_MODEL
print(f"Duration: {duration:.1f}s | Sample rate: {sr}Hz | Channels: {n_channels}")
print(f"{'─'*50}")

scores = []
times = []
for start in range(0, max(1, total_samples - CHUNK_SAMPLES), HOP_SAMPLES):
    chunk = audio[start:start+CHUNK_SAMPLES]
    if len(chunk) < CHUNK_SAMPLES:
        chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))
    t = (start + CHUNK_SAMPLES/2) / SAMPLE_RATE_MODEL
    emb = get_embedding(chunk)
    score = run_npu(emb)
    scores.append(score)
    times.append(t)
    bar = '█' * int(score * 30) + '░' * (30 - int(score * 30))
    status = ' 🚨 SCREAM!' if score > THRESHOLD else ''
    print(f"t={t:5.1f}s  [{bar}]  {score:.3f}{status}")

print(f"{'─'*50}")
max_score = max(scores)
detected = max_score > THRESHOLD
print(f"Max score:  {max_score:.4f}")
print(f"Threshold:  {THRESHOLD}")
print(f"Result:     {'🚨 SCREAM DETECTED' if detected else '✓ No scream detected'}")
