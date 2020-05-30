import librosa
import librosa.display

import numpy as np
import matplotlib.pyplot as plt


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

# And the first-order differences (delta features)?
mfcc_delta = librosa.feature.delta(mfcc)

# mfccとmfcc_deltaをペアにする(np.vstack)
# これをビートに同期させる
# これによりビート間でmfccの平均を取る等可能に
beat_mfcc_delta = librosa.util.sync(np.vstack([mfcc, mfcc_delta]),
                                    beat_frames)

# 調波音からクロマ(12半音)情報の取得
# これはある時刻の調波音に含まれる12半音(C,D,...,B)の強度を表す
chromagram = librosa.feature.chroma_cqt(y=y_harmonic,
                                        sr=sr)

# *クロマグラムの表示                                        
plt.figure(figsize=(6,4))
librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma')
plt.title('Chromagram')
plt.colorbar()
plt.tight_layout()
plt.show()

# 今回はクロマ情報をビートに同期させる
# 'aggregate'は「集計」という意味、おそらくビート間でクロマ情報の代表値を算出する方法？
beat_chroma = librosa.util.sync(chromagram,
                                beat_frames,
                                aggregate=np.median)

# ビートに同期したクロマ情報, mfcc, mfcc_deltaをすべて連結
beat_features = np.vstack([beat_chroma, beat_mfcc_delta])