import os
import time

import nibabel as nib
import glob
import numpy as np
import config
import torch
import logging
import sys
import torch.nn as nn
from scipy import ndimage
import cv2
from skimage import measure
from preprocess.preprocess_operation import universal3Dlargestregion
import skimage.transform as T


def norm_img(image):  # 归一化像素值到（0，1）之间，且将溢出值取边界值
    MIN_BOUND = -1024.0
    MAX_BOUND = -300.0  # -700
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image


def normalize_img(img):
    mean = np.mean(img)
    std = np.std(img)
    img -= mean
    img /= std
    return img


def normalize_output(result):
    r_sum = np.sum(result, axis=-1)

    r_sum[r_sum == 0] = 1  # hmm this should not be necessary - but it is :(
    r_sum = np.repeat(r_sum[:, :, :, np.newaxis], 1, -1)  # repeat the sum along channel axis (to allow division)

    result = np.divide(result, r_sum)


def downscale_scan(scan, sz=512, z_length=64, order=0):
    """下采样三维矩阵，scan:nii数组；sz:xy轴的维度；z_length：z轴切片数；order：采样方式，0为最近邻采样"""
    zoom_seq = np.array([sz, sz, z_length]) / np.array(scan.shape)
    s = ndimage.interpolation.zoom(scan, zoom_seq, order=order, prefilter=order)
    return s


def tst_net(net, device):
    # imgs = os.listdir("D:/Dada/airwaydata/originaldata/test")
    # print(imgs)
    test_results = os.path.join('./result/test_patch_result')
    if not os.path.exists(test_results):
        os.makedirs(test_results)

    BLOCK_SIZE = [128, 128, 32]


    # [[0, 0, 80], [0, 0, 96], [0, 0, 112]]保存不同的[i,j,k]的位置  为拼接提供位置索引

    test_images = np.ndarray((1, 1, BLOCK_SIZE[0], BLOCK_SIZE[1], BLOCK_SIZE[2]), dtype=np.float32)

    imgs = glob.glob(r"D:/Dada/airwaydata/originaldata/test/test_patch" + "/*." + "nii.gz")
    print(len(imgs))

    for imgname in imgs:
        midname = imgname[imgname.rindex("\\") + 1:]  # 返回最后一个位置的索引加1
        """nii打开"""
        print(midname)

        nii_file = nib.load(r'D:/Dada/airwaydata/originaldata/test/test_patch' + '/' + midname)
        affine = nii_file.affine
        nii_datas = nii_file.get_data()

        nii_datas = nii_datas.astype('float')

        array_img = nib.Nifti1Image(nii_datas, affine=None)
        # 检查预处理完的文件
        nib.save(array_img, './ct1.nii.gz')


        # 保留原坐标信息
        ori_shape = nii_datas.shape
        DATA_SIZE = ori_shape

        # test_data = np.ndarray((512, 512, ori_shape[2]), dtype=np.float32)
        # test_data[:, :, :] = nii_datas
        test_data = nii_datas



        test_data_1 = nib.Nifti1Image(test_data, affine=None)
        nib.save(test_data_1, './ct2.nii.gz')

        result = np.ndarray((DATA_SIZE[0], DATA_SIZE[1], DATA_SIZE[2]))

        patch = test_data[0:0 + BLOCK_SIZE[0], 0:0 + BLOCK_SIZE[1], 0:0 + BLOCK_SIZE[2]]
        patch = T.resize(patch.astype("float"), BLOCK_SIZE)
        # 多做一步resize,确保输入一样
        array_img = nib.Nifti1Image(patch, affine=None)
        # 切块检查
        nib.save(array_img, './tespatch.nii.gz')
        print('切块尺寸：', patch.shape)
        if ((patch.shape[0] == BLOCK_SIZE[0]) and (patch.shape[1] == BLOCK_SIZE[1]) and (
                patch.shape[2] == BLOCK_SIZE[2])):
            # 前两个位置是索引，后面为索引切片对应的层/
            test_images[0, 0, 0:BLOCK_SIZE[0], 0:BLOCK_SIZE[1], 0:BLOCK_SIZE[2]] = patch
            images = torch.from_numpy(test_images)
            images = images.to(device=device, dtype=torch.float32)

            test_pred, _, _ = net(images)

            # test_pred = net(images)
            print('test_pred',test_pred.shape)

            test_pred = nn.Softmax(dim=1)(test_pred)
            print('test_pred_after_softmax:', test_pred.shape)
            test_output1 = test_pred.squeeze().cpu().detach().numpy()
            print('test_output', test_output1.shape)
            # 筛选出网络预测值Prediction中最大概率值对应的class ID,axis=1 为对应的类的维度，即n
            test_pred_argmax = np.argmax(test_output1, axis=0).astype("float")
            print('test_pred_argmax', test_pred_argmax.shape)
            result = test_pred_argmax

            print('the shape of test_per_argmax:', test_pred_argmax.shape, 'result', result.shape)

        # result = normalize_output(result)
        result[result > 0] = 1

        new_image = nib.Nifti1Image(result, affine)
        nib.save(new_image, os.path.join(test_results, midname))


if __name__ == '__main__':
    args = config.args

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # setting GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    print('加载模型')
    net = torch.load('./output/{}/best.pkl'.format(args.save))

    # restore_path1 = "D:/gzq/MICCAI/output/0414/"
    # restore_path = restore_path1 + "best0.pkl"
    # net = torch.load(restore_path)
    time1 = time.time()

    net.to(device=device)
    try:
        tst_net(net=net, device=device)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'preprocess/INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
    time2 = time.time()
    print('total time', time2 - time1)
