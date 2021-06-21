"""created by L.X
"""
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pickle
from python_speech_features import fbank
from scipy.io import wavfile
import scipy
import os

'''
示例wav无法librosa抽取
'''
def normalize_frames(m,Scale=True):
    if Scale:
        return (m - np.mean(m, axis=0)) / (np.std(m, axis=0) + 2e-12)
    else:
        return (m - np.mean(m, axis=0))

'''
1. enroll.wav and enroll.p should be edited for using to test.wav and test.p, 
2. used for extracting features of enroll and test .wav, for testing
'''

dirct = 'data_aishell/data_aishell/wav/test'
dirList=[]
fileList=[]
files=os.listdir(dirct) 
for f in files:
    filename = dirct+'/'+f+'/test.wav'
    sample_rate = 16000

    audio, sr = librosa.load(filename, sr=sample_rate, mono=True)

    filter_banks, energies = fbank(audio, samplerate=sample_rate, nfilt=40, winlen=0.025)
    print('filter_banks:',filter_banks.shape)
    print('energies:',len(energies))
    filter_banks = 20 * np.log10(np.maximum(filter_banks,1e-5))
    feature = normalize_frames(filter_banks, Scale=False)
    print('feature:',feature.shape)
    librosa.display.specshow(librosa.power_to_db(feature.T),sr=sr, x_axis='time', y_axis='linear')
    plt.title('Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()
    '''
    empty = {}
    empty['feat'] = feature
    empty['label'] = f
    fileopen = 'feat_logfbank_nfilt40_thchs30/test/'+f
    if not os.path.exists(fileopen):
        os.mkdir(fileopen)
    output = open(fileopen+'/test.p','wb')
    pickle.dump(empty, output)
    print('--------enroll-feature--------')
    print(feature.shape)
    '''


