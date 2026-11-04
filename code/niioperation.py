#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/11/15 17:09
#@Author: dada
#@File  : niioperation.py

import nibabel as nib
import numpy as np
from scipy import ndimage
import os
import glob


def downscale_scan(scan, sz=512, z_length=64, order=0):
    """采样三维矩阵，scan:nii数组；sz:xy轴的维度；z_length：z轴切片数；order：采样方式，0为最近邻采样"""
    zoom_seq = np.array([sz, sz, z_length], dtype='float') / np.array(scan.shape, dtype='float')
    s = ndimage.interpolation.zoom(scan, zoom_seq, order=order, prefilter=order)

    return s


def resizenii(inputnii, outputnii):
    """nii处理模板"""
    # 读取文件
    img = nib.load(inputnii)
    img_arra = img.get_data()
    print(img_arra.dtype)
    # img_arra = (img_arra.astype(np.uint8))
    affine = img.affine
    head = img.header

    img_arra[img_arra > 0] = 1
    """
    此处加入对nii的矩阵的操作


    """
    img_arra = downscale_scan(img_arra, sz=512, z_length=337, order=0)  # resize
    # 保存文件
    array_img = nib.Nifti1Image(img_arra, affine=affine,header=head)
    nib.save(array_img, outputnii)


# cur_dir = os.getcwd()
# inputnii = (cur_dir + '/chemask_a.nii.gz')
#
# outputnii = './chemask_a_256.nii.gz'
#
# resizenii(inputnii, outputnii)
nii_path = 'C:/Users/ASUS/Desktop/result'
niis = glob.glob(nii_path + "/*." + 'nii.gz')
print("排序前", niis)
# niis = sorted(niis)
# print("排序后", niis)
for nii in niis:
    # print(nii)
    print(nii.split("\\")[1])
    # print("E:/doctordata/Exact09/256/data/" + nii.split("\\")[1])
    resizenii(nii, "C:/Users/ASUS/Desktop/result/" + nii.split("\\")[1])
#     print("done")




