"""created by L.X
"""
import numpy as np
'''
sims1 = [1,0.91,1,1,0.89,0.91]
sims2 =  [0,1,1,1,0.89,0.8]
threshold = 0.9
actual_issame1=[1,0,0,0,0,0]
actual_issame2=[0,1,0,0,0,0]
actual_issame = actual_issame1 + actual_issame2
sims = sims1 + sims2

print(sims)
print(actual_issame)
predict_issame = np.greater(sims, threshold)
print(predict_issame)

true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
print('true_accept')
print(true_accept)
false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
print('false_accept')
print(false_accept)
n_same = np.sum(actual_issame)
print('n_same')
print(n_same)
n_diff = np.sum(np.logical_not(actual_issame))
print('n_diff')
print(n_diff)
if n_diff == 0:
    n_diff = 1    
val = float(true_accept) / float(n_same)
frr = 1 - val
far = float(false_accept) / float(n_diff)
print(frr)
print(far)
'''
'''
Thanks for https://github.com/Walleclipse/Deep_Speaker-speaker_recognition_system/blob/master/eval_metrics.py
'''
def main():
    sims1 = [1,0.91,1,1,0.89,0.91]
    sims2 =  [0,1,1,1,0.89,0.8]
    threshold = 0.9
    actual_issame1=[1,0,0,0,0,0]
    actual_issame2=[0,1,0,0,0,0]
    actual_issame = actual_issame1 + actual_issame2
    sims = sims1 + sims2
    thresholds = np.arange(0, 1.0, 0.0001)
    labels = actual_issame 
    eer = calculate_eer(thresholds, sims,labels)
    print('eer:',eer)
    

def calculate_eer(thresholds, sims, labels):
    nrof_pairs = min(len(labels), len(sims))
    nrof_thresholds = len(thresholds)

    indices = np.arange(nrof_pairs)


    # Find the threshold that gives FAR = far_target
    far_train = np.zeros(nrof_thresholds)
    frr_train = np.zeros(nrof_thresholds)
    eer_index = 0
    eer_diff = 100000000
    thres = []
    for threshold_idx, threshold in enumerate(thresholds):
        frr_train[threshold_idx], far_train[threshold_idx] = calculate_val_far(threshold, sims, labels)
        if abs(frr_train[threshold_idx]-far_train[threshold_idx]) < eer_diff:
            eer_diff = abs(frr_train[threshold_idx]-far_train[threshold_idx])
            eer_index = threshold_idx
            thres.append(threshold)


    frr, far = frr_train[eer_index], far_train[eer_index]
    print('frr:',frr)
    print('far:',far)
    print('threshold:',thres[-1])
    eer = (frr + far)/2
    print('eer_diff:',eer_diff)
    print('eer:',eer)

    return eer

def calculate_val_far(threshold, sims, actual_issame):
    predict_issame = np.greater(sims, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    #print('----------')
    #print('threshold:',threshold)
    #print('true_accept:',true_accept)
    #print('false_accept:',false_accept)
    n_same = np.sum(actual_issame)
    #print('n_same:',n_same)
    n_diff = np.sum(np.logical_not(actual_issame))
    #print('n_diff:',n_diff)
    if n_diff == 0:
        n_diff = 1
    if n_same == 0:
        return 0,0
    val = float(true_accept) / float(n_same)
    frr = 1 - val
    far = float(false_accept) / float(n_diff)
    #print('frr:',frr)
    #print('far:',far)

    return frr, far
if __name__ == '__main__':
    main()

