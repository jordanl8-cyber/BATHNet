# SPDX-FileCopyrightText: 2026 VIA Technologies, Inc. VISVIA Software Team
#
# SPDX-License-Identifier: MIT

"""
Download AudioSet clips for scream detection training.

Downloads:
- ~380 scream clips
- ~800 background noise clips
- ~800 vocal/speech clips

Usage: python3 download_audioset.py
Requires: yt-dlp, ffmpeg
"""

import os
import random
import subprocess
import urllib.request

SCREAM_IDS = {'/m/03qc9zr', '/m/07qfr4h', '/m/07plz5l', '/m/07r04'}
NEGATIVE_IDS = {
    '/m/0bgv8', '/m/07yv9', '/m/0838f', '/m/01b_21',
    '/m/09x0r', '/m/07pp_mv', '/m/06_y0by', '/m/0k4j',
}
VOCAL_IDS = {
    '/m/09x0r', '/m/05zppz', '/m/02zsn', '/m/0ytgt',
    '/m/01h8n0', '/m/0l14jd', '/m/01swy6', '/m/0l14gg',
    '/m/0l14md', '/m/04rlf', '/m/0hdsk', '/m/01j3sz',
    '/m/07pws3f', '/m/09hlz4', '/m/02fxyj',
}

def download_csvs():
    base = "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv"
    for name in ["balanced_train_segments.csv", "eval_segments.csv"]:
        if not os.path.exists(name):
            print(f"Downloading {name}...")
            urllib.request.urlretrieve(f"{base}/{name}", name)

def parse_csv(filepath):
    rows = []
    with open(filepath) as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split(', ')
            if len(parts) >= 4:
                ytid, start, end = parts[0], float(parts[1]), float(parts[2])
                labels = set(parts[3].replace('"', '').split(','))
                rows.append((ytid, start, end, labels))
    return rows

def download_clip(ytid, start, end, out_path):
    if os.path.exists(out_path):
        return True
    duration = end - start
    cmd = [
        'yt-dlp', '-x', '--audio-format', 'wav',
        '--postprocessor-args', f'ffmpeg:-ar 44100 -ac 1 -ss {start} -t {duration}',
        '-o', out_path, '-q',
        f'https://www.youtube.com/watch?v={ytid}'
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        return result.returncode == 0
    except Exception:
        return False

print("Downloading AudioSet CSV files...")
download_csvs()

print("Parsing clips...")
all_rows = parse_csv('balanced_train_segments.csv') + parse_csv('eval_segments.csv')

scream_clips = [(y,s,e) for y,s,e,l in all_rows if l & SCREAM_IDS]
neg_clips = [(y,s,e) for y,s,e,l in all_rows if (l & NEGATIVE_IDS) and not (l & SCREAM_IDS)]
vocal_clips = [(y,s,e) for y,s,e,l in all_rows if (l & VOCAL_IDS) and not (l & SCREAM_IDS)]
print(f"Found: {len(scream_clips)} scream, {len(neg_clips)} negative, {len(vocal_clips)} vocal")

for folder in ['audioset_screams', 'audioset_negatives', 'audioset_vocals']:
    os.makedirs(folder, exist_ok=True)

random.seed(42)
random.shuffle(neg_clips)
random.shuffle(vocal_clips)

# Download screams
print("\nDownloading scream clips...")
done = 0
for i, (ytid, s, e) in enumerate(scream_clips):
    out = f'audioset_screams/{ytid}_{int(s)}.wav'
    if download_clip(ytid, s, e, out):
        done += 1
    if (i+1) % 50 == 0:
        print(f"  {i+1}/{len(scream_clips)} attempted, {done} downloaded")
print(f"Scream clips: {done}")

# Download negatives
print("\nDownloading negative clips...")
done = 0
for i, (ytid, s, e) in enumerate(neg_clips[:1500]):
    out = f'audioset_negatives/{ytid}_{int(s)}.wav'
    if download_clip(ytid, s, e, out):
        done += 1
    if (i+1) % 100 == 0:
        print(f"  {i+1}/1500 attempted, {done} downloaded")
    if done >= 800:
        break
print(f"Negative clips: {done}")

# Download vocals
print("\nDownloading vocal clips...")
done = 0
for i, (ytid, s, e) in enumerate(vocal_clips[:1500]):
    out = f'audioset_vocals/{ytid}_{int(s)}.wav'
    if download_clip(ytid, s, e, out):
        done += 1
    if (i+1) % 100 == 0:
        print(f"  {i+1}/1500 attempted, {done} downloaded")
    if done >= 800:
        break
print(f"Vocal clips: {done}")

print(f"\nDone! Check audioset_screams/, audioset_negatives/, audioset_vocals/")
