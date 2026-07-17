# SPDX-FileCopyrightText: 2026 VIA Technologies, Inc. VISVIA Software Team
#
# SPDX-License-Identifier: MIT

"""
Train Scream Detector v13
Standard binary classifier on 96K sample dataset.
No distillation — pure hard label training on large diverse dataset.

Usage: python3 train_v13.py
Output: scream_detector_v13.tflite
"""

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             confusion_matrix, roc_auc_score,
                             classification_report)

print("Loading data...")
X = np.load('X_full_clap.npy')
y_hard = np.load('y_hard_full_clap.npy')

print(f"Dataset: {X.shape}")
print(f"Screaming: {y_hard.sum()}, NotScreaming: {(y_hard==0).sum()}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_hard, test_size=0.15, random_state=42, stratify=y_hard)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42, stratify=y_train)

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

scream_weight = len(y_train) / (2 * y_train.sum())
not_scream_weight = len(y_train) / (2 * (y_train==0).sum())
print(f"Class weights: scream={scream_weight:.3f}, not_scream={not_scream_weight:.3f}")

# Build model
inputs = tf.keras.Input(shape=(1024,))
x = tf.keras.layers.Dense(512, activation='relu')(inputs)
x = tf.keras.layers.Dropout(0.4)(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs, outputs, name='scream_detector_v13')

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.0005),
    loss='binary_crossentropy',
    metrics=['accuracy',
             tf.keras.metrics.AUC(name='auc'),
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall')]
)
model.summary()

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=128,
    class_weight={0: not_scream_weight, 1: scream_weight},
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
    ]
)

# Evaluation
print("\n" + "="*55)
print("FULL EVALUATION - v13")
print("="*55)

y_pred_prob = model.predict(X_test, verbose=0).flatten()
y_pred = (y_pred_prob > 0.5).astype(int)

print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_pred_prob):.4f}")
cm = confusion_matrix(y_test, y_pred)
print(f"TN={cm[0][0]}  FP={cm[0][1]}  FN={cm[1][0]}  TP={cm[1][1]}")
print(classification_report(y_test, y_pred, target_names=['Not Scream', 'Scream']))

print("\nThreshold Analysis:")
for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    y_p = (y_pred_prob > t).astype(int)
    p = precision_score(y_test, y_p, zero_division=0)
    r = recall_score(y_test, y_p, zero_division=0)
    f = f1_score(y_test, y_p, zero_division=0)
    print(f"  t={t}: Precision={p:.3f}  Recall={r:.3f}  F1={f:.3f}")

# Export to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('scream_detector_v13.tflite', 'wb') as f:
    f.write(tflite_model)
print(f"\nSaved scream_detector_v13.tflite ({len(tflite_model)/1024:.1f} KB)")
print("\nNext: compile for NPU with:")
print("  sudo ncc-tflite --arch mdla3.0 --relax-fp32 \\")
print("    -d scream_detector_v13.dla scream_detector_v13.tflite")
