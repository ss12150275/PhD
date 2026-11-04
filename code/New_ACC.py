import torch
import os
import torch.nn.functional as F
import SimpleITK as sitk
import numpy as np

import pandas as pd


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

    IMAGEDIR =  'E:/L_File/SSL_Vessel/QQProcessed/evalu_d3/'
    # IMAGEDIR = 'E:/L_File/SSL_Vessel/QQProcessed/evalu_dataset5/'

    OUTPUTDIR = 'E:/L_File/SSL_Vessel/runs/exp_MT_label_0.5/new_p/'


    # DATASET = 'RSNA_ASNR_MICCAI_BraTS2021_ValidationData'
    # IMAGEDIR = 'E:/Datasets/BRATS21/' + DATASET

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    Architectures = ['5-20']  # default, 1444, ADAM, wcedceloss
    architecture = Architectures[0]

    ############################
    # load data
    ###########################
    label_paths = [os.path.join(IMAGEDIR, x)
                   for x in os.listdir(IMAGEDIR)
                   if x.endswith('seg.nii.gz')]
    print(label_paths)

    output_paths = [os.path.join(OUTPUTDIR, x)
                    for x in os.listdir(OUTPUTDIR)
                    if x.endswith('nii.gz')]
    print(output_paths)
    assert len(label_paths) == len(output_paths)
    results = {'ACC': []}
    for index in range(len(label_paths)):
        # assert label_paths[index].split('/')[-1] == output_paths[index].split('/')[-1].replace(".nii.gz", "_seg.nii.gz")
        assert label_paths[index].split('/')[-1] == output_paths[index].split('/')[-1]
        label_itk = sitk.ReadImage(label_paths[index])
        output_itk = sitk.ReadImage(output_paths[index])

        label_array = sitk.GetArrayFromImage(label_itk).astype('uint8')
        output_array = sitk.GetArrayFromImage(output_itk).astype('uint8')

        label_array[label_array > 2] = 0
        output_array[output_array > 2] = 0

        y_predV = np.array(output_array.copy() == 2, dtype='uint8')
        y_truthV = np.array(label_array.copy() == 2, dtype='uint8')

        y_predA = np.array(output_array.copy() == 1, dtype='uint8')
        y_truthA = np.array(label_array.copy() == 1, dtype='uint8')

        neg_y_trueV = 1 - y_truthV  # 找到标签里除label外的位置

        neg_y_predV = 1 - y_predV  # 找到预测里除了label外的位置


        neg_y_trueA = 1 - y_truthA  # 找到标签里除label外的位置

        neg_y_predA = 1 - y_predA # 找到预测里除了label外的位置




        Tv = y_truthV.sum()  # 真实标签
        Pv = y_predV.sum()  # 预测结果


        TP = np.sum(y_truthA * y_predA)  # 真阳
        FPa = np.sum(y_truthV * y_predA)  # 假阳（错分对）
        TN = np.sum(y_truthV * y_predV)  # 真阴
        FNa = np.sum(y_truthA * y_predV)  # 假阴（对分错）




        ACC = (TP+TN)/(TP+FNa+TN+FPa)




        results['ACC'].append(ACC)




        # results['dice3'].append(dices[3])

    data_frame = pd.DataFrame(
        data={
              'ACC': results['ACC'],


              # 'dice3': results['dice3']
              },
        index=range(1, len(label_paths) + 1))
    # data_frame.to_csv(output_paths[0].split('\\')[-2] + '/' + 'evaluation_' + architecture + '.csv', index_label='Index')

    # data_frame.to_csv('./EvaResult/3.29' + '/' + 'Recall_' + architecture + '.csv',
    #                   index_label='Fusion')
    data_frame.to_csv(OUTPUTDIR + 'ACC_' + architecture + '.csv',
                      index_label='Fusion')

    #'E:/L_File/3D_lung/outputs/MUti/xiaorong/Distal/3D_Dita