# Feature extraction example
import numpy as np
import librosa

# サンプルのファイルをロード
# y：波形
# sr：サンプリングレート
y, sr = librosa.load(librosa.util.example_audio_file())

# hop lengthは１フレームあたりのサンプル数
# sr = 22050 Hz, hop512 samples ~= 23ms
hop_length = 512

# 音声を調波音と打楽器音に分割
# y_harmonic：調波音
# y_percussive：打楽器音
y_harmonic, y_percussive = librosa.effects.hpss(y)

# ビート情報
# tempo：BPM
# beat_frames：ビートのタイミングのフレーム
tempo, beat_frames = librosa.beat.beat_track(y=y_percussive,
                                             sr=sr)

# MFCC(メル周波数ケプストラム)の算出
mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)

# And the first-order differences (delta features)
mfcc_delta = librosa.feature.delta(mfcc)

# Stack and synchronize between beat events
# This time, we'll use the mean value (default) instead of median
beat_mfcc_delta = librosa.util.sync(np.vstack([mfcc, mfcc_delta]),
                                    beat_frames)

# Compute chroma features from the harmonic signal
chromagram = librosa.feature.chroma_cqt(y=y_harmonic,
                                        sr=sr)

# Aggregate chroma features between beat events
# We'll use the median value of each feature between beat frames
beat_chroma = librosa.util.sync(chromagram,
                                beat_frames,
                                aggregate=np.median)

# Finally, stack all beat-synchronous features together
beat_features = np.vstack([beat_chroma, beat_mfcc_delta])