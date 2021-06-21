"""created by L.X
"""
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pandas as pd
import math
import os
import configure as c

from DB_wav_reader import read_feats_structure
from SR_Dataset import read_MFB, ToTensorTestInput
from model.model_my import background_resnet
import test_eval as eval_eer

def load_model(use_cuda, log_dir, cp_num, embedding_size, n_classes):
    model = background_resnet(embedding_size=embedding_size, num_classes=n_classes)
    if use_cuda:
        model.cuda()
    print('=> loading checkpoint')
    # original saved file with DataParallel
    checkpoint = torch.load(log_dir + '/checkpoint_' + str(cp_num) + '.pth')
    # create new OrderedDict that does not contain `module.`
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model

def split_enroll_and_test(dataroot_dir):
    DB_all = read_feats_structure(dataroot_dir)
    enroll_DB = pd.DataFrame()
    test_DB = pd.DataFrame()
    
    enroll_DB = DB_all[DB_all['filename'].str.contains('enroll.p')]
    test_DB = DB_all[DB_all['filename'].str.contains('test.p')]
    
    # Reset the index
    enroll_DB = enroll_DB.reset_index(drop=True)
    test_DB = test_DB.reset_index(drop=True)
    return enroll_DB, test_DB

def load_enroll_embeddings(embedding_dir):
    embeddings = {}
    for f in os.listdir(embedding_dir):
        spk = f.replace('.pth','')
        # Select the speakers who are in the 'enroll_spk_list'
        embedding_path = os.path.join(embedding_dir, f)
        tmp_embeddings = torch.load(embedding_path)
        embeddings[spk] = tmp_embeddings
        
    return embeddings

def get_embeddings(use_cuda, filename, model, test_frames):
    print('read_mfb')
    input, label = read_MFB(filename) # input size:(n_frames, n_dims)
    
    tot_segments = math.ceil(len(input)/test_frames) # total number of segments with 'test_frames' 
    activation = 0
    with torch.no_grad():
        for i in range(tot_segments):
            temp_input = input[i*test_frames:i*test_frames+test_frames]
            
            TT = ToTensorTestInput()
            temp_input = TT(temp_input) # size:(1, 1, n_dims, n_frames)
    
            if use_cuda:
                temp_input = temp_input.cuda()
            temp_activation,_ = model(temp_input)
            activation += torch.sum(temp_activation, dim=0, keepdim=True)
    
    activation = l2_norm(activation, 1)
                
    return activation

def l2_norm(input, alpha):
    input_size = input.size()  # size:(n_frames, dim)
    buffer = torch.pow(input, 2)  # 2 denotes a squared operation. size:(n_frames, dim)
    normp = torch.sum(buffer, 1).add_(1e-10)  # size:(n_frames)
    norm = torch.sqrt(normp)  # size:(n_frames)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    # Multiply by alpha = 10 as suggested in https://arxiv.org/pdf/1703.09507.pdf
    output = output * alpha
    return output

def perform_identification(use_cuda, model, embeddings, test_filename, test_frames, spk_list):
    test_embedding = get_embeddings(use_cuda, test_filename, model, test_frames)
    print("test_embedding:",test_embedding.shape)
    max_score = -10**8
    best_spk = None
    for spk in spk_list:
        score = F.cosine_similarity(test_embedding, embeddings[spk])
        score = score.data.cpu().numpy() 
        if score > max_score:
            max_score = score
            best_spk = spk
    #print("Speaker identification result : %s" %best_spk)
    true_spk = test_filename.split('/')[-2].split('_')[0]
    print("\n=== Speaker identification ===")
    print("True speaker : %s\nPredicted speaker : %s\nResult : %s\n" %(true_spk, best_spk, true_spk==best_spk))
    print("score:",max_score)
    return best_spk,true_spk==best_spk

def simulate_eer(use_cuda, model, embeddings, test_filename, test_frames, spk_list):
    test_embedding = get_embeddings(use_cuda, test_filename, model, test_frames)
    # print("test_embedding:",test_embedding.shape)
    scores = []
    labels = []
    true_spk = test_filename.split('/')[-2].split('_')[0]
    for spk in spk_list:
        score = F.cosine_similarity(test_embedding, embeddings[spk])
        score = score.data.cpu().numpy() 
        scores.append(float(score))
        if spk == true_spk:
            labels.append(1)
        else:
            labels.append(0)
    return scores, labels

def main():
    
    log_dir = 'model_saved_voxceleb1_resnet50' # Where the checkpoints are saved
    embedding_dir = 'enroll_embeddings' # Where embeddings are saved
    test_dir = 'feat_logfbank_nfilt40_voxceleb1/test/' # Where test features are saved
    
    # Settings
    use_cuda = True # Use cuda or not
    embedding_size = 512 # Dimension of speaker embeddings
    cp_num = 29 # Which checkpoint to use?
    n_classes = 1211 # How many speakers in training data?
    test_frames = 100 # Split the test utterance 

    # Load model from checkpoint
    model = load_model(use_cuda, log_dir, cp_num, embedding_size, n_classes)

    # Get the dataframe for test DB
    enroll_DB, test_DB = split_enroll_and_test(c.TEST_FEAT_VOXCELEB1_DIR)
    
    # Load enroll embeddings
    embeddings = load_enroll_embeddings(embedding_dir)
    
    dirct = 'feat_logfbank_nfilt40_voxceleb1/test'
    spk_list=os.listdir(dirct) 
    print(spk_list)

    '''
    simulate best spr but cant simulate eer
    '''
    # ACC calculating
    # Set the test speaker
    '''
    num = 0
    for test_speaker in spk_list:
        test_path = os.path.join(test_dir, test_speaker, 'test.p')
        # Perform the test 
        best_spk,flg = perform_identification(use_cuda, model, embeddings, test_path, test_frames, spk_list)
        if(flg):
            num = num +1
    print('correct num:',num)
    '''
    
    '''
    simulate eer
    '''
    
    
    scores_pre = []
    labels_real = []

    for spkr in spk_list:
        test_speaker = spkr
        test_path = os.path.join(test_dir, test_speaker, 'test.p')
        # Simulate EER
        scores, labels = simulate_eer(use_cuda, model, embeddings, test_path, test_frames, spk_list)
        scores_pre = scores_pre+ scores
        labels_real = labels_real+ labels
    
    print('scores_pre:',scores_pre)
    print('labels_real:',labels_real)
    thresholds = np.arange(0, 1.0, 0.001)
    eval_eer.calculate_eer(thresholds, scores_pre, labels_real)
    
       
if __name__ == '__main__':
    main()
