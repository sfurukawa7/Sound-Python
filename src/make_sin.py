import numpy as np
import soundfile as sf

SAMPLING_RATE = 16000
LENGTH = 5
FREQUENCY = 442


def main():
    phases = np.cumsum(2.0*np.pi*FREQUENCY/SAMPLING_RATE*np.ones(int(LENGTH*SAMPLING_RATE)))
    sin = np.sin(phases)
    sf.write('442.wav', sin, SAMPLING_RATE, 'PCM_24')

if __name__ == '__main__':
    main()
