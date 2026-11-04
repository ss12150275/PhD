import torch
import os
import torch.nn.functional as F
import SimpleITK as sitk
import numpy as np

import pandas as pd
from New_DSC import read_nii

def dice_score(prediction, target):
    smooth = 1e-5
    num_classes = target.size(0)
    prediction = prediction.view(num_classes, -1)
    target = target.view(num_classes, -1)

    intersection = (prediction * target)

    dice = (2. * intersection.sum(1) + smooth) / (prediction.sum(1) + target.sum(1) + smooth)

    return dice


if __name__ == '__main__':
    ############################
    # Parameters
    ############################
    BATCHSIZE = 1
    LEARNINGRATE = 0.0001
    K = 5

    # IMAGEDIR =  'D:/Users/Desktop/NewEva/final_seg/'

    IMAGEDIR = 'E:/exact09/test/label/'
    # IMAGEDIR = 'E:/L_File/SSL_Vessel/QQProcessed/evalu_dataset5/'

    OUTPUTDIR =   'E:/exact09/test_results/2023_4_26/'



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    Architectures = ['5-20']  # default, 1444, ADAM, wcedceloss
    architecture = Architectures[0]

    ############################
    # load data
    ###########################
    label_paths = [os.path.join(IMAGEDIR, x)
                   for x in os.listdir(IMAGEDIR)
                   if x.endswith('.nii.gz')]
    print(label_paths)

    output_paths = [os.path.join(OUTPUTDIR, x)
                    for x in os.listdir(OUTPUTDIR)
                    if x.endswith('nii.gz')]
    print(output_paths)
    assert len(label_paths) == len(output_paths)
    results = {'recall': []}
    for index in range(len(label_paths)):
        # assert label_paths[index].split('/')[-1] == output_paths[index].split('/')[-1].replace(".nii.gz", "_seg.nii.gz")
        assert label_paths[index].split('/')[-1] == output_paths[index].split('/')[-1]
        label_array = read_nii(label_paths[index]).astype('uint8')
        output_array = read_nii(output_paths[index]).astype('uint8')

        # label_array[label_array > 2] = 0
        # output_array[output_array > 2] = 0

        y_pred = np.array(output_array.copy() == 1, dtype='uint8')
        y_truth = np.array(label_array.copy() == 1, dtype='uint8')

        seg_inv, gt_inv = np.logical_not(y_pred), np.logical_not(y_truth)
        true_pos = float(np.logical_and(y_pred, y_truth).sum())  # float for division
        true_neg = np.logical_and(seg_inv, gt_inv).sum()
        false_pos = np.logical_and(y_pred, gt_inv).sum()
        false_neg = np.logical_and(seg_inv, y_truth).sum()

        # 然后根据公式分别计算出这几种指标
        prec = true_pos / (true_pos + false_pos + 1e-6)
        rec = true_pos / (true_pos + false_neg + 1e-6)
        accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg + 1e-6)
        F1 = 2 * true_pos / (2 * true_pos + false_pos + false_neg + 1e-6)
        IoU = true_pos / (true_pos + false_neg + false_pos + 1e-6)

        results['recall'].append(rec)


        # results['dice3'].append(dices[3])

    data_frame = pd.DataFrame(
        data={'recall': results['recall']

              # 'dice3': results['dice3']
              },
        index=range(1, len(label_paths) + 1))
    # data_frame.to_csv(output_paths[0].split('\\')[-2] + '/' + 'evaluation_' + architecture + '.csv', index_label='Index')

    # data_frame.to_csv('./EvaResult/3.29' + '/' + 'Recall_All' + architecture + '.csv',
    #                   index_label='Fusion')

    data_frame.to_csv( OUTPUTDIR + 'Recall_' + architecture + '.csv',
                      index_label='Fusion')



