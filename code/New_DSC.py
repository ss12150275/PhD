import torch
import os
import torch.nn.functional as F
import SimpleITK as sitk
import numpy as np
import nibabel as nib
import pandas as pd


def dice_score(prediction, target):
    smooth = 1e-5
    num_classes = target.size(0)
    prediction = prediction.view(num_classes, -1)
    target = target.view(num_classes, -1)

    intersection = (prediction * target)

    dice = (2. * intersection.sum(1) + smooth) / (prediction.sum(1) + target.sum(1) + smooth)

    return dice

def read_nii(path):
    img_file = nib.load(path)
    array = img_file.get_fdata()
    return array

if __name__ == '__main__':
    ############################
    # Parameters
    ############################
    BATCHSIZE = 1
    LEARNINGRATE = 0.0001
    K = 5



    IMAGEDIR =  'E:/exact09/test/label/'
    # IMAGEDIR = 'E:/L_File/SSL_Vessel/QQProcessed/evalu_dataset5/'

    # OUTPUTDIR = 'E:/L_File/SSL_Vessel/runs/exp_MT_label_0.5/predictions/muti/'
    OUTPUTDIR = 'E:/exact09/test_results/2023_4_27/'


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    Architectures = ['4_28']  # default, 1444, ADAM, wcedceloss
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
                   if x.endswith('.nii.gz')]
    print(output_paths)
    assert len(label_paths) == len(output_paths)
    results = {'dice0': [], 'dice1': [],  'Tdice': []}
    for index in range(len(label_paths)):
        print(label_paths[index].split('/')[-1], output_paths[index].split('/')[-1])
        assert label_paths[index].split('/')[-1] == \
               output_paths[index].split('/')[-1]#.replace(".nii.gz","_seg.nii.gz")
        # print(label_paths[index].split('/')[-1],output_paths[index].split('/')[-1])
        # assert label_paths[index].split('/')[-1][:-13] == output_paths[index].split('/')[-1][:-10]
        # label_itk = sitk.ReadImage(label_paths[index])
        # output_itk = sitk.ReadImage(output_paths[index])
        #
        # label_array = sitk.GetArrayFromImage(label_itk).astype('uint8')
        # output_array = sitk.GetArrayFromImage(output_itk).astype('uint8')

        label_array = read_nii(label_paths[index]).astype('uint8')
        output_array = read_nii(output_paths[index]).astype('uint8')
        label_array[label_array > 0] = 1
        output_array[output_array > 0] = 1

        #
        # print("jiancha_output:",output_array.shape)
        # print("jiancha_label:", label_array.shape)

        label_tensor = torch.from_numpy(label_array).long()
        output_tensor = torch.from_numpy(output_array).long()
        print('output_tensor,label_tensor',output_tensor.shape,label_tensor.shape)
        label_onehot = torch.nn.functional.one_hot(label_tensor, num_classes=2).permute(3, 0, 1, 2).contiguous()
        output_onehot = torch.nn.functional.one_hot(output_tensor, num_classes=2).permute(3, 0, 1, 2).contiguous()

        # print("onehot_output:", output_onehot.shape)
        # print("onehout_label:", label_onehot.shape)

        dices = dice_score(prediction=output_onehot, target=label_onehot).numpy()
        print(index, dices)
        # results['dice0'].append(dices[0])

        results['dice1'].append(dices[1])
        # results['dice2'].append(dices[2])
        # results['Tdice'].append((dices[1]+dices[2])/2)
        # results['dice3'].append(dices[3])

    data_frame = pd.DataFrame(
        data={
              'dice': results['dice1']
              },
        index=range(1, len(label_paths) + 1))
    # data_frame.to_csv(output_paths[0].split('\\')[-2] + '/' + 'evaluation_' + architecture + '.csv', index_label='Index')

    data_frame.to_csv(OUTPUTDIR + 'Dice_evaluation_' + architecture + '.csv',
                      index_label='Z')


