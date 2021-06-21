"""created by L.X
"""
import librosa
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
'''
dirct = 'test_wavs2'
dirList=[]
fileList=[]
files=os.listdir(dirct) 
for f in files:
    filename = dirct+'/'+f+'/test.wav'
    sample_rate = 16000

    audio, sr = librosa.load(filename, sr=sample_rate, mono=True)

    filter_banks, energies = fbank(audio, samplerate=sample_rate, nfilt=40, winlen=0.025)
    filter_banks = 20 * np.log10(np.maximum(filter_banks,1e-5))
    feature = normalize_frames(filter_banks, Scale=False)
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
'''
used for extracting features of train dataset(data_thchs30), .wav->.p
'''
'''
dirct = 'data_aishell/data_aishell/wav/train'
files = os.listdir(dirct)
str_ = '_'
str2 = '.'
for f in files:
    if f.startswith('A') & f.endswith('.wav'):
        fileopen = 'feat_logfbank_nfilt40_thchs30/train/'
        if not os.path.exists(fileopen+f[:f.index(str_)]):
            os.mkdir(fileopen+f[:f.index(str_)])
        filename = dirct + '/' + f
        sample_rate = 16000

        audio, sr = librosa.load(filename, sr=sample_rate, mono=True)

        filter_banks, energies = fbank(audio, samplerate=sample_rate, nfilt=40, winlen=0.025)
        filter_banks = 20 * np.log10(np.maximum(filter_banks,1e-5))
        feature = normalize_frames(filter_banks, Scale=False) 
        empty = {}
        empty['feat'] = feature
        empty['label'] = f[:f.index(str_)]
        output = open(fileopen + f[:f.index(str_)] + '/' + f[:f.index(str2)] + '.p','wb')
        pickle.dump(empty, output)
        print('--------enroll-feature--------')
        print(feature.shape)
'''

'''

used for extracting features of train dataset(sata_aishell), .wav->.p

'''
'''
dirct = 'data_aishell/data_aishell/wav/train'
files = os.listdir(dirct)
str2 = '.'

for f in files:
    wavfiles = os.listdir(dirct + '/' + f)
    fileopen = 'feat_logfbank_nfilt40_aishell/train/'
    if not os.path.exists(fileopen+f):
        os.mkdir(fileopen+f)
    for wav in wavfiles:
        print('--------enroll-feature--------')
        print(f,'/',wav)
        try:
            filename = dirct + '/' + f + '/' +wav
            sample_rate = 16000

            audio, sr = librosa.load(filename, sr=sample_rate, mono=True)

            filter_banks, energies = fbank(audio, samplerate=sample_rate, nfilt=40, winlen=0.025)
            filter_banks = 20 * np.log10(np.maximum(filter_banks,1e-5))
            feature = normalize_frames(filter_banks, Scale=False) 
            """
            VAD
            """
            start_sec, end_sec = 0.5, 0.5
            start_frame = int(start_sec / 0.01)
            end_frame = len(feature) - int(end_sec / 0.01)
            ori_feat = feature
            feature = feature[start_frame:end_frame,:]
            if(len(feature) < 40 or len(ori_feat) < 40):
                continue
            empty = {}
            empty['feat'] = ori_feat
            empty['label'] = f
            output = open(fileopen + f + '/' + wav[:wav.index(str2)] + '.p','wb')
            pickle.dump(empty, output)
            print(feature.shape)  
        except IndexError:
            continue
        continue
'''

'''

used for extracting features of test dataset(sata_aishell), .wav->.p

'''

'''
dirct = 'data_aishell/data_aishell/wav/test'
files = os.listdir(dirct)
str2 = '.'
for f in files:
    wavfiles = os.listdir(dirct + '/' + f)
    fileopen = 'feat_logfbank_nfilt40_aishell/test/'
    if not os.path.exists(fileopen+f):
        os.mkdir(fileopen+f)
    for wav in wavfiles:
        print('--------enroll-feature--------')
        print(f,'/',wav)
        try:

            filename = dirct + '/' + f + '/' +wav

            sample_rate = 16000



            audio, sr = librosa.load(filename, sr=sample_rate, mono=True)



            filter_banks, energies = fbank(audio, samplerate=sample_rate, nfilt=40, winlen=0.025)

            filter_banks = 20 * np.log10(np.maximum(filter_banks,1e-5))

            feature = normalize_frames(filter_banks, Scale=False) 

            """
            VAD
            """

            start_sec, end_sec = 0.5, 0.5

            start_frame = int(start_sec / 0.01)

            end_frame = len(feature) - int(end_sec / 0.01)

            ori_feat = feature

            feature = feature[start_frame:end_frame,:]

            if(len(feature) < 40 or len(ori_feat) < 40):

                continue

            empty = {}

            empty['feat'] = ori_feat

            empty['label'] = f

            output = open(fileopen + f + '/' + wav[:wav.index(str2)] + '.p','wb')

            pickle.dump(empty, output)

            print(feature.shape)  

        except IndexError:

            continue

        continue
'''
'''

used for extracting features of test dataset(voxceleb1), .wav->.p

'''


dirct = 'voxceleb1/voxceleb1/test'
files = os.listdir(dirct)
str2 = '.'
for f in files:#f id
    wavfiles = os.listdir(dirct + '/' + f)
    fileopen = 'feat_logfbank_nfilt40_voxceleb1/test/'
    if not os.path.exists(fileopen+f):
        os.mkdir(fileopen+f)
    for wavfile in wavfiles:# wavfile dfhjdfhgkhd
        wavid_files = os.listdir(dirct + '/' + f +'/'+ wavfile)
        for wav in wavid_files:# 00001.wav
            
            print('--------enroll-feature--------')
            try:
                filename = dirct + '/' + f + '/' + wavfile +'/' + wav
                print('filename:',filename)

                sample_rate = 16000



                audio, sr = librosa.load(filename, sr=sample_rate, mono=True)



                filter_banks, energies = fbank(audio, samplerate=sample_rate, nfilt=40, winlen=0.025)

                filter_banks = 20 * np.log10(np.maximum(filter_banks,1e-5))

                feature = normalize_frames(filter_banks, Scale=False) 

                """
                VAD
                """

                start_sec, end_sec = 0.5, 0.5

                start_frame = int(start_sec / 0.01)

                end_frame = len(feature) - int(end_sec / 0.01)

                ori_feat = feature

                feature = feature[start_frame:end_frame,:]

                if(len(feature) < 40 or len(ori_feat) < 40):

                    continue

                empty = {}

                empty['feat'] = ori_feat

                empty['label'] = f

                output = open(fileopen + f + '/' +wavfile + wav[:wav.index(str2)] + '.p','wb')

                pickle.dump(empty, output)

                print(feature.shape)  

            except IndexError:

                continue

            continue



        
            
        














'''
filename = 'test_wavs/cryz/test.wav'
sample_rate = 16000

audio, sr = librosa.load(filename, sr=sample_rate, mono=True)

filter_banks, energies = fbank(audio, samplerate=sample_rate, nfilt=40, winlen=0.025)
filter_banks = 20 * np.log10(np.maximum(filter_banks,1e-5))
feature = normalize_frames(filter_banks, Scale=False)
empty = {}
empty['feat'] = feature
empty['label'] = 'cryz'
output = open('feat_logfbank_nfilt40/test/cryz/test.p','wb')
pickle.dump(empty, output)
print('--------enroll-feature--------')
print(feature.shape)

'''
