"""
Live Scream Detection Web Dashboard
VIA VAB-5000 (MediaTek Genio 700 / MDLA 3.0 NPU)

Run with: python3 live_scream_web.py
Then open: http://jordan-vab.local:5000
"""

import numpy as np
import subprocess
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
import threading
import queue
import time
import os
import collections
import io
from flask import Flask, Response, render_template_string, jsonify

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ── Configuration ─────────────────────────────────────────
THRESHOLD = 0.8          # Scream detection threshold (0-1)
SAMPLE_RATE_HW = 48000   # Hardware mic sample rate
SAMPLE_RATE_MODEL = 16000 # Model input sample rate
CHUNK_SAMPLES = 15600    # 0.975s at 16kHz
WINDOW_SECONDS = 30      # Rolling window width
RECORD_SECONDS = 1       # Audio chunk duration

YAMNET_PATH = '/home/jordanlee/yamnet_embedding.tflite'
DLA_PATH = '/home/jordanlee/scream_detector_v13.dla'
MIC_DEVICE = 'plughw:Device,0'

# ── Load models ───────────────────────────────────────────
print("Loading models...")
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

# ── Shared state ──────────────────────────────────────────
audio_queue = queue.Queue(maxsize=5)
running = True
times = collections.deque(maxlen=WINDOW_SECONDS)
scores = collections.deque(maxlen=WINDOW_SECONDS)
audio_buffer = collections.deque(maxlen=WINDOW_SECONDS * SAMPLE_RATE_MODEL)
start_time = time.time()
latest_status = {'score': 0.0, 'detected': False, 'time': 0.0}
cached_graph = io.BytesIO()
graph_lock = threading.Lock()

# ── Audio threads ─────────────────────────────────────────
def record_thread():
    while running:
        result = subprocess.run([
            'arecord', '-D', MIC_DEVICE,
            '-c', '2', '-r', str(SAMPLE_RATE_HW),
            '-f', 'S16_LE', '-d', str(RECORD_SECONDS),
            '--quiet', '-'
        ], capture_output=True)
        if result.returncode == 0 and len(result.stdout) > 0:
            try:
                audio_queue.put_nowait(result.stdout)
            except queue.Full:
                pass

def process_thread():
    while running:
        try:
            raw = audio_queue.get(timeout=2.0)
        except queue.Empty:
            continue
        audio = np.frombuffer(raw, dtype=np.int16)
        audio = audio[::2].astype(np.float32) / 32768.0
        audio_16k = signal.resample_poly(audio, 1, 3).astype(np.float32)
        audio_buffer.extend(audio_16k)
        chunk = audio_16k[:CHUNK_SAMPLES] if len(audio_16k) >= CHUNK_SAMPLES else \
                np.pad(audio_16k, (0, CHUNK_SAMPLES - len(audio_16k)))
        emb = get_embedding(chunk)
        score = run_npu(emb)
        t = time.time() - start_time
        times.append(t)
        scores.append(score)
        latest_status['score'] = score
        latest_status['detected'] = score > THRESHOLD
        latest_status['time'] = t
        print(f"\rt={t:.1f}s score={score:.3f} "
              f"{'🚨 SCREAM!' if score > THRESHOLD else '  quiet  '}",
              end='', flush=True)

def graph_thread():
    """Regenerate graph every 2 seconds in background."""
    global cached_graph
    while running:
        if len(times) < 2:
            time.sleep(1)
            continue

        t_arr = np.array(times)
        s_arr = np.array(scores)
        audio_arr = np.array(audio_buffer, dtype=np.float32)

        fig, ax = plt.subplots(figsize=(14, 6))
        fig.patch.set_facecolor('#0d1117')
        ax.set_facecolor('#0d1117')

        # Spectrogram overlay
        if len(audio_arr) > 512:
            f_spec, t_spec, Sxx = signal.spectrogram(
                audio_arr, fs=SAMPLE_RATE_MODEL, nperseg=512, noverlap=384)
            Sxx_db = 10 * np.log10(Sxx + 1e-10)
            Sxx_db = np.clip(Sxx_db, -100, -30)
            freq_mask = f_spec <= 4000
            f_scaled = f_spec[freq_mask] / 4000.0
            Sxx_plot = Sxx_db[freq_mask, :]
            t_offset = t_arr[-1] - (len(audio_arr) / SAMPLE_RATE_MODEL)
            t_spec_abs = t_spec + t_offset
            score_interp = np.interp(t_spec_abs, t_arr, s_arr,
                                     left=s_arr[0], right=s_arr[-1])
            Sxx_masked = Sxx_plot.copy().astype(float)
            for col_idx in range(len(t_spec_abs)):
                cutoff_idx = np.searchsorted(f_scaled, score_interp[col_idx])
                Sxx_masked[cutoff_idx:, col_idx] = np.nan
            ax.pcolormesh(t_spec_abs, f_scaled, Sxx_masked,
                         shading='gouraud', cmap='inferno',
                         vmin=-100, vmax=-30, zorder=2)

        # Score line and dots
        above = s_arr > THRESHOLD
        ax.plot(t_arr, s_arr, color='white', linewidth=2.5, zorder=10)
        ax.axhline(THRESHOLD, color='#ff8800', linestyle='--',
                   linewidth=1.8, label=f'Threshold ({THRESHOLD})', zorder=8)
        ax.scatter(t_arr[above], s_arr[above], color='#ff3333',
                   s=120, zorder=12, label='Scream detected',
                   edgecolors='white', linewidths=1.5)
        ax.scatter(t_arr[~above], s_arr[~above], color='#33cc55',
                   s=80, zorder=12, label='No scream',
                   edgecolors='white', linewidths=1.2)
        for t, s in zip(t_arr[-10:], s_arr[-10:]):
            ax.text(t, s + 0.04, f'{s:.2f}', ha='center', va='bottom',
                    fontsize=8, color='white', fontweight='bold', zorder=13)

        # Rolling window
        x_max = max(t_arr[-1] + 1, WINDOW_SECONDS)
        x_min = x_max - WINDOW_SECONDS
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, 1.12)
        ax.set_xlabel('Time (s)', color='white', fontsize=11)
        ax.set_ylabel('Scream Score / Frequency (0-4 kHz)', color='white', fontsize=10)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['0.0\n(0 kHz)', '0.25\n(1 kHz)', '0.5\n(2 kHz)',
                            '0.75\n(3 kHz)', '1.0\n(4 kHz)'], color='white', fontsize=8)
        ax.tick_params(colors='white')
        ax.spines[:].set_color('#333344')
        ax.grid(True, alpha=0.12, color='white', axis='y')
        ax.legend(loc='upper left', facecolor='#1a1a2e',
                  edgecolor='#555', labelcolor='white', fontsize=9)

        last_score = s_arr[-1]
        status_color = '#ff3333' if last_score > THRESHOLD else '#33cc55'
        status_text = 'SCREAM DETECTED' if last_score > THRESHOLD else 'No scream'
        ax.text(0.99, 0.97, f'{status_text}  {last_score:.2f}',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=11, fontweight='bold', color=status_color,
                bbox=dict(boxstyle='round', facecolor='#1a1a2e',
                          edgecolor=status_color, alpha=0.9))
        ax.set_title('Live Scream Detection — VIA VAB-5000 (MTK MDLA 3.0 NPU)',
                     fontsize=12, fontweight='bold', color='white')

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                    facecolor='#0d1117')
        plt.close(fig)
        buf.seek(0)
        with graph_lock:
            cached_graph = buf

        time.sleep(2)

# ── Flask web app ─────────────────────────────────────────
app = Flask(__name__)

HTML = '''<!DOCTYPE html>
<html>
<head>
    <title>Live Scream Detection</title>
    <style>
        body { background: #0d1117; color: white; font-family: Arial;
               text-align: center; margin: 0; padding: 20px; }
        h1 { color: #ff8800; font-size: 20px; margin-bottom: 5px; }
        #status { font-size: 26px; font-weight: bold; margin: 10px;
                  padding: 10px 20px; border-radius: 8px; display: inline-block; }
        #graph { max-width: 100%; border-radius: 8px; margin: 10px auto; }
        #info { color: #aaa; font-size: 12px; margin: 5px; }
    </style>
    <script>
        function refreshStatus() {
            fetch('/status').then(r => r.json()).then(data => {
                const el = document.getElementById('status');
                if (data.detected) {
                    el.style.color = '#ff3333';
                    el.style.border = '2px solid #ff3333';
                    el.innerHTML = '&#128680; SCREAM DETECTED &nbsp; ' + data.score.toFixed(3);
                } else {
                    el.style.color = '#33cc55';
                    el.style.border = '2px solid #33cc55';
                    el.innerHTML = '&#10003; No scream &nbsp; ' + data.score.toFixed(3);
                }
                document.getElementById('info').innerHTML =
                    'Time: ' + data.time.toFixed(1) + 's &nbsp;|&nbsp; ' +
                    'Threshold: ''' + str(THRESHOLD) + ''' &nbsp;|&nbsp; Model: scream_detector_v13';
            });
        }
        function refreshGraph() {
            var newImg = new Image();
            newImg.onload = function() {
                document.getElementById('graph').src = newImg.src;
            };
            newImg.src = '/graph?' + Date.now();
        }
        setInterval(refreshStatus, 1000);
        setInterval(refreshGraph, 2500);
        refreshStatus();
        refreshGraph();
    </script>
</head>
<body>
    <h1>Live Scream Detection &mdash; VIA VAB-5000 (MTK MDLA 3.0 NPU)</h1>
    <div id="status">Loading...</div><br>
    <img id="graph" src="/graph" style="max-width:100%;"/>
    <div id="info"></div>
</body>
</html>'''

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/graph')
def graph():
    with graph_lock:
        cached_graph.seek(0)
        data = cached_graph.read()
    if not data:
        return Response(b'', mimetype='image/png')
    return Response(data, mimetype='image/png',
                    headers={'Cache-Control': 'no-cache'})

@app.route('/status')
def status():
    return jsonify(latest_status)

# ── Start ─────────────────────────────────────────────────
for target in [record_thread, process_thread, graph_thread]:
    threading.Thread(target=target, daemon=True).start()

print("\nOpen in your browser: http://jordan-vab.local:5000")
print("Press Ctrl+C to stop\n")
app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
