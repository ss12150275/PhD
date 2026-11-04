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

    # IMAGEDIR =  'D:/Users/Desktop/NewEva/final_seg/'

    IMAGEDIR = 'E:/L_File/SSL_Vessel/QQProcessed/evalu_d3/'
    # IMAGEDIR = 'E:/L_File/SSL_Vessel/QQProcessed/evalu_dataset5/'

    OUTPUTDIR =   'E:/L_File/SSL_Vessel/runs/exp_MT_label_0.5/new_p/'



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
    results = {'Recalla': [],'Recallv': [],'prea': [],'prev': [],'F1a': [],'F1v': [],'FPRa': [],'FPRv': [],
               'Trecall': [],'Tpre': [],'TF1': [],'TFPR': [],}
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

        neg_y_trueV = 1 - y_truthV  # 找到标签里除label外的位置

        neg_y_predV = 1 - y_predV  # 找到预测里除了label外的位置

        Tv = y_truthV.sum()  # 真实标签
        Pv = y_predV.sum()  # 预测结果

        TPv = np.sum(y_truthV * y_predV)  # 真阳
        FPv = np.sum(neg_y_trueV * y_predV)  # 假阳（错分对）
        TNv = np.sum(neg_y_trueV * neg_y_predV)  # 真阴
        FNv = np.sum(y_truthV * neg_y_predV)  # 假阴（对分错）

        Recallv = TPv / (TPv + FNv)
        prev = TPv / (TPv + FPv)
        F1v = (2 * prev * Recallv) / (prev + Recallv)
        accv = (TPv + TNv) / (TPv + FPv + TNv + FNv)
        FPRv = FPv / (FPv + TNv)

#####################################################################################
        y_predA = np.array(output_array.copy() == 1, dtype='uint8')
        y_truthA = np.array(label_array.copy() == 1, dtype='uint8')

        neg_y_trueA = 1 - y_truthA  # 找到标签里除label外的位置

        neg_y_predA = 1 - y_predA # 找到预测里除了label外的位置

        Ta = y_truthA.sum()  # 真实标签
        Pa = y_predA.sum()  # 预测结果
        TPa = np.sum(y_truthA * y_predA)  # 真阳
        FPa = np.sum(neg_y_trueA * y_predA)  # 假阳（错分对）
        TNa = np.sum(neg_y_trueA* neg_y_predA)  # 真阴
        FNa = np.sum(y_truthA * neg_y_predA)  # 假阴（对分错）

        Recalla = TPa / (TPa + FNa)
        prea = TPa / (TPa + FPa)
        F1a = (2 * prea * Recalla) / (prea + Recalla)
        acca = (TPa + TNa) / (TPa + FPa + TNa + FNa)
        FPRa = FPa / (FPa + TNa)

        Trecall = (Recalla+Recallv)/2
        Tpre = (prea+prev)/2
        TF1 = (F1a+F1v)/2
        Tacc = (acca+accv)/2
        TFPR = (FPRa+FPRv)/2





        results['Recalla'].append(Recalla)
        results['Recallv'].append(Recallv)
        results['prea'].append(prea)
        results['prev'].append(prev)
        results['F1a'].append(F1a)
        results['F1v'].append(F1v)

        results['FPRa'].append(FPRa)
        results['FPRv'].append(FPRv)

        results['Trecall'].append(Trecall)
        results['Tpre'].append(Tpre)
        results['TF1'].append(TF1)

        results['TFPR'].append(TFPR)



        # results['dice3'].append(dices[3])

    data_frame = pd.DataFrame(
        data={'Recalla': results['Recalla'],
              'Recallv': results['Recallv'],
              'prea': results['prea'],
              'prev': results['prev'],
              'F1a': results['F1a'],
              'F1v': results['F1v'],

              'FPRa': results['FPRa'],
              'FPRv': results['FPRv'],


              'Trecall': results['Trecall'],
              'Tpre': results['Tpre'],
              'TF1': results['TF1'],

              'TFPR': results['TFPR'],

              # 'dice3': results['dice3']
              },
        index=range(1, len(label_paths) + 1))
    # data_frame.to_csv(output_paths[0].split('\\')[-2] + '/' + 'evaluation_' + architecture + '.csv', index_label='Index')

    # data_frame.to_csv('./EvaResult/3.29' + '/' + 'Recall_All' + architecture + '.csv',
    #                   index_label='Fusion')

    data_frame.to_csv( OUTPUTDIR + 'Recall_' + architecture + '.csv',
                      index_label='Fusion')


