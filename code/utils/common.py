# -*- coding:utf-8 -*-
import SimpleITK as sitk
import numpy as np
from scipy import ndimage
import nibabel as nib
import glob
import skimage.transform as T


MIN_BOUND = -1024.0
MAX_BOUND = -300.0  #-700

def resize(scan, resizeshape, order=0):
    """采样三维矩阵，scan:nii数组；sz:xy轴的维度；z_length：z轴切片数；order：采样方式，0为最近邻采样"""
    if(len(resizeshape)==3):
        zoom_seq = np.array([resizeshape[0], resizeshape[1], resizeshape[2]], dtype='float') / np.array(scan.shape,
                                                                                                        dtype='float')
    else:
        zoom_seq = np.array([resizeshape[0], resizeshape[1]], dtype='float') / np.array(scan.shape,
                                                                                                        dtype='float')
    # zoom_seq = np.array([resizeshape[0], resizeshape[1], resizeshape[2]], dtype='float') / np.array(scan.shape, dtype='float')
    s = ndimage.interpolation.zoom(scan, zoom_seq, order=order, prefilter=order)

    return s


def norm_img(image): # 归一化像素值到（0，1）之间，且将溢出值取边界值
    MIN_BOUND = np.min(image)
    MAX_BOUND = np.max(image)
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


def sitk_read_row(img_path, resize_scale=1): # 读取3D图像并resale（因为一般医学图像并不是标准的[1,1,1]scale）
    # nda = sitk.ReadImage(img_path)
    # nda = sitk.GetArrayFromImage(nda)  # channel first
    nda = nib.load(img_path)
    nda = nda.get_data()
    # print('nda的大小',nda.shape)
    # nda = np.array(nda.copy() > 0)
    #####进行上采样和下采样
    # nda=ndimage.zoom(nda,[resize_scale,resize_scale,resize_scale],order=0) #rescale

    return nda

def nib_read_rowdata(filename,crop_size):
    # print(filename)
    f_nii=nib.load(filename)
    f_img = f_nii.get_fdata()
    # print('duquwenjian',f_img.shape)

    f_img=T.resize(f_img.astype("float"),crop_size)
    # print('duquwenjian2', f_img.shape)
    #
    # seg_img = resize(seg_img.astype("float"), crop_size)#mode='constant'

    return f_img
def nib_read_rowlabel(filename,crop_size):
    # print(filename)
    seg_nii = nib.load(filename)
    seg_img = seg_nii.get_fdata()
    seg_img=np.rint(seg_img).astype("float")
    # if (seg_img.max() == 6):
    # seg_img[seg_img == 6] = 1
    # print(seg_img.shape,'=============')
    seg_img = seg_img.reshape(seg_img.shape[0],seg_img.shape[1],seg_img.shape[2])
    seg_img = resize(seg_img.astype("float"), crop_size) #mode='constant'

    return seg_img

def nib_read_rowdata_2d(filename,crop_size):
    # print(filename)
    f_nii=nib.load(filename)
    f_img = f_nii.get_data()


    return f_img

def nib_read_rowlabel_2d(filename,crop_size):
    # print(filename)
    seg_nii = nib.load(filename)
    seg_img = seg_nii.get_data()
    seg_img=np.rint(seg_img).astype("float")
    # if (seg_img.max() == 6):
    seg_img[seg_img == 6] = 1



    return seg_img

def load_slicer2(path,mode="train"):
    f_name = path
    seg = f_name.replace("ct", "seg")
    p, gdname = os.path.split(seg)
    f_nii = nib.load(f_name)
    f_img = f_nii.get_data()
    if mode=="test":
        seg_img=np.ones_like(f_img)
    else:
        seg_nii = nib.load(seg)
        seg_img = seg_nii.get_data()

    # exposure.equalize_hist()
    return f_img, seg_img, gdname


def make_one_hot_3d(x, n): # 对输入的volume数据x，对每个像素值进行one-hot编码
    one_hot = np.zeros([x.shape[0], x.shape[1], x.shape[2], n]) # 创建one-hot编码后shape的zero张量
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            for v in range(x.shape[2]):
                # print('当前体素点的值',int(x[i, j, v]))
                one_hot[i, j, v, int(x[i, j, v])] = 1 # 给相应类别的位置置位1，模型预测结果也应该是这个shape

    return one_hot

def make_one_hot_2d(x, n): # 对输入的volume数据x，对每个像素值进行one-hot编码
    one_hot = np.zeros([x.shape[0], x.shape[1], n]) # 创建one-hot编码后shape的zero张量
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            # print('当前体素点的值',int(x[i, j, v]))
            one_hot[i, j, int(x[i, j])] = 1 # 给相应类别的位置置位1，模型预测结果也应该是这个shape

    return one_hot

import random


def random_crop_2d(img, label, crop_size):
    random_x_max = img.shape[0] - crop_size[0]
    random_y_max = img.shape[1] - crop_size[1]

    if random_x_max < 0 or random_y_max < 0:
        return None

    x_random = random.randint(0, random_x_max)
    y_random = random.randint(0, random_y_max)

    crop_img = img[x_random:x_random + crop_size[0], y_random:y_random + crop_size[1]]
    crop_label = label[x_random:x_random + crop_size[0], y_random:y_random + crop_size[1]]

    return crop_img, crop_label


def random_crop_3d(img, label, crop_size):
    random_x_max = img.shape[0] - crop_size[0]
    random_y_max = img.shape[1] - crop_size[1]
    random_z_max = img.shape[2] - crop_size[2]
    print('随机采样点边界',random_x_max,random_y_max,random_z_max)

    if random_x_max < 0 or random_y_max < 0 or random_z_max < 0:
        return None,None

    else:
        x_random = random.randint(0, random_x_max)
        y_random = random.randint(0, random_y_max)
        z_random = random.randint(0, random_z_max)

        crop_img = img[x_random:x_random + crop_size[0], y_random:y_random + crop_size[1],
                   z_random:z_random + crop_size[2]]
        crop_label = label[x_random:x_random + crop_size[0], y_random:y_random + crop_size[1],
                     z_random:z_random + crop_size[2]]

        return crop_img, crop_label
import os
def load_file_name_list2(file_path):
    print('获取文件名称')
    filename_list = os.listdir(os.path.join(file_path,'data'))
    print(filename_list)
    return filename_list


def load_file_name_list(file_path):
    file_name_list = []
    with open(file_path, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline().strip()  # 整行读取数据
            if not lines:
                break
                pass
            file_name_list.append(lines)
            pass
    print('文件名称',file_name_list)
    return file_name_list

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# def adjust_learning_rate(optimizer, epoch, lr = 0.0001):
#     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#     lr = lr * (0.1 ** (epoch // 30))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


# if __name__=='__main__':
#     fixd_path = r'D:/Dada/airwaydata/originaldata/total_airway_data/LIDC-EXACT(public)/EXACT'
#     filename_list = load_file_name_list2(fixd_path)
#     label = nib_read_rowlabel( r"D:\Dada\airwaydata\originaldata\total_airway_data\LIDC-EXACT(public)\EXACT\label\case01.nii.gz", [64, 64, 64])
#
#     print('独热编码前：', label.shape)
#     onehotlabels = make_one_hot_3d(label,3)
#     print('独热编码后：', onehotlabels.shape)