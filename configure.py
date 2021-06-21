# Feature path
TRAIN_FEAT_TIBMD_DIR = 'feat_logfbank_nfilt40_TIBMD/train'
TEST_FEAT_TIBMD_DIR = 'feat_logfbank_nfilt40_TIBMD/test'
TRAIN_FEAT_AISHELL_DIR = 'feat_logfbank_nfilt40_aishell/train'
TEST_FEAT_AISHELL_DIR = 'feat_logfbank_nfilt40_aishell/test'
TRAIN_FEAT_VOXCELEB1_DIR = 'feat_logfbank_nfilt40_voxceleb1/train'
TEST_FEAT_VOXCELEB1_DIR = 'feat_logfbank_nfilt40_voxceleb1/test'

# Context window size
NUM_WIN_SIZE = 100 #10

# Settings for feature extraction
USE_LOGSCALE = True
USE_DELTA = False
USE_SCALE = False
SAMPLE_RATE = 16000
FILTER_BANK = 40

TEST_WAV_DIR = 'test_wavs'