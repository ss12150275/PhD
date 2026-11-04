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


def norm_img(image): # 归一化像素值到（0，1）之间，且将溢出值取边界值
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
    r_sum = np.sum(result,axis=-1)

    r_sum[r_sum==0] = 1 #hmm this should not be necessary - but it is :(
    r_sum = np.repeat(r_sum[:,:,:,np.newaxis],1,-1) #repeat the sum along channel axis (to allow division)

    result = np.divide(result,r_sum)


def downscale_scan(scan, sz=512, z_length=64, order=0):
    """下采样三维矩阵，scan:nii数组；sz:xy轴的维度；z_length：z轴切片数；order：采样方式，0为最近邻采样"""
    zoom_seq = np.array([sz, sz, z_length]) / np.array(scan.shape)
    s = ndimage.interpolation.zoom(scan, zoom_seq, order=order, prefilter=order)
    return s

def imgpre(img_data,sizexy):
    upper = -300
    lower = -1024

    #对应的把数据放缩到最大最小之间

    img_arra = np.clip(img_data, lower, upper)
    #最大最小归一化
    img_arra = (img_arra - lower) / (upper - lower) * 255

    img_arra = (img_arra.astype(np.uint8))
    # print('the value of image',img_arra[:,:,100])
    # f1 = nib.Nifti1Image(img_arra, affine=None)
    # nib.save(f1, './origin_pre.nii.gz')
    #创建指定的空容器
    a = img_data.shape[2]
    nii_arr = np.zeros([sizexy, sizexy, a])
    for i in range(a):  # 二维操作和三维操作速度略快一点点
        #遍历的对每一个切面做预处理
        im = img_arra[:, :, i]
        # im = ndimage.rotate(im, -90)  # 角度负数为顺时针旋转
        # im = im[:, ::-1]  # 列翻转
        # 预处理程序开始
        im3 = cv2.resize(im, (sizexy, sizexy), interpolation=cv2.INTER_NEAREST)  # cv的resize可设置插值算法

        #################################
        deal = im3.copy()


        #阈值截断,二值化
        deal[deal < 250] = 0
        deal[deal >= 250] = 1  # 即找出白框部分


        labels = measure.label(deal, connectivity=2)  # 找白色区域的8连通域，并给予每个连通域标号
        jj = measure.regionprops(labels)  # 这里是取得labels的属性，属性有许多
        num = labels.max()  # 找白色部分的连通域有几个
        del_array = np.array([0] * (num + 1))
        for k in range(num):  # 这里是找最大的那个白色连通域的标号
            if k == 0:
                initial_area = jj[0].area
                save_index = 1  # 初始保留第一个连通域
            else:
                k_area = jj[k].area  # 将元组转换成array
                if initial_area < k_area:
                    initial_area = k_area
                    save_index = k + 1  # 因为python从0开始，而连通域标记是从1开始
        del_array[save_index] = 1
        del_mask = del_array[labels]
        out = deal * del_mask * 255  # 原始白框乘上最大白框*255
        out = (out.astype(np.uint8))  # 找出了最大的白框部分

        im1 = out.copy()
        # imageio.imwrite('./preprocessing/im1' + '%06d' % i + '.tif', im1)
        h, w = out.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(im1, mask, (0, 0), 255)
        cv2.floodFill(im1, mask, (sizexy - 2, sizexy - 2), 255)
        cv2.floodFill(im1, mask, (sizexy - 2, 0), 255)
        cv2.floodFill(im1, mask, (0, sizexy - 2), 255)
        # # imageio.imwrite('./preprocessing/im1111' + '%06d' % i + '.tif', im1)
        im1 = cv2.bitwise_not(im1)
        im1 = (im1.astype(np.uint8))  # 这里把1转换成255了
        nii_arr[:, :, i] = im1  # 预处理与否这里要改
        # imageio.imwrite('./testimage/%03d' % i + '.tif', im2)
    # print(nii_arr.dtype)
    nii_arr = downscale_scan(nii_arr, sz=sizexy, z_length=a, order=0)  # 255
    nii_arr = np.where(nii_arr > 0, 1, 0)  # 1
    nii_arr = universal3Dlargestregion(nii_arr)
    # print(nii_arr.dtype)
    # print(img_data.dtype)
    img_data = downscale_scan(img_data, sz=sizexy, z_length=a, order=0)
    # print('the value of prepocessing image', img_data[:, :, 100])
    # print(img_data.dtype)
    img_data = nii_arr * img_data
    img_data[img_data < -1024] = -1024  # 将-3000+归到-1024
    print("preprocessing airway done")
    # print('the value of prepocessing image',img_data[:,:,100])

    return img_data

def universal3Dlargestregion(deal):
    """找到3D丈量最大连通域,输出为值为1的mask.deal:输入的3D张量"""
    labels = measure.label(deal, connectivity=3)  # 找白色区域的8连通域，并给予每个连通域标号，connectivity为ndarry的维数，三维数组故为3
    jj = measure.regionprops(labels)  # 这里是取得labels的属性，属性有许多
    num = labels.max()  # 找白色部分的连通域有几个
    del_array = np.array([0] * (num + 1))
    for k in range(num):  # 这里是找最大的那个白色连通域的标号
        if k == 0:
            initial_area = jj[0].area
            save_index = 1  # 初始保留第一个连通域
        else:
            k_area = jj[k].area  # 将元组转换成array
            if initial_area < k_area:
                initial_area = k_area
                save_index = k + 1  # python从0开始，而连通域标记是从1开始
    del_array[save_index] = 1
    del_mask = del_array[labels]
    return del_mask

def largest3D(data):
    labels, num = measure.label(data, return_num=True)
    sizes = measure.label(data, connectivity=1)
    regions = measure.regionprops(labels, sizes)
    max_region = max(regions, key=lambda x: x.area)
    max_label = max_region.label
    data[labels != max_label] = 0
    return data

def tst_net(net,device):
    # imgs = os.listdir("D:/Dada/airwaydata/originaldata/test")
    # print(imgs)
    test_results = os.path.join(r'E:\exact09\test_results\2023_4_27')
    if not os.path.exists(test_results):
        os.makedirs(test_results)

    BLOCK_SIZE = args.crop_size

    Batch_size = 2
    #[[0, 0, 80], [0, 0, 96], [0, 0, 112]]保存不同的[i,j,k]的位置  为拼接提供位置索引
    #创建列表，用于切块填充，利用索引填充到对应位置
    test_images = np.ndarray((Batch_size, 1, BLOCK_SIZE[0], BLOCK_SIZE[1], BLOCK_SIZE[2]), dtype=np.float32)




    # imgs = glob.glob("D:/Dada/airwaydata/originaldata/test" + "/*." + "nii.gz")
    imgs = glob.glob(r"E:\exact09\test\image" + "/*." + "nii.gz")
    print(len(imgs))

    print(imgs)
    for imgname in imgs:

        midname = imgname[imgname.rindex("\\") + 1:]  # 返回最后一个位置的索引加1
        """nii打开"""
        print(midname)

        nii_file = nib.load(r'E:\exact09\test\image' + '/' + midname)
        affine = nii_file.affine
        nii_datas = nii_file.get_fdata()
        print('------------',nii_datas.shape)
        h = nii_datas.shape[2]
        print('------------', h)
        nii_datas = downscale_scan(nii_datas, sz=512, z_length=256)#不能添加，预测结果浆糊一样
        nii_datas = imgpre(nii_datas, 512)
        nii_datas = nii_datas.astype('float')

        # array_img = nib.Nifti1Image(nii_datas, affine=None)
        # #检查预处理完的文件
        # nib.save(array_img, './ct1.nii.gz')

        nii_datas[nii_datas < -1024] = -1024
        nii_datas[nii_datas > -300] = -300


        nii_datas = norm_img(nii_datas)

        nii_datas = normalize_img(nii_datas)

        # 保留原坐标信息
        ori_shape = nii_datas.shape
        DATA_SIZE = ori_shape


        # test_data = np.ndarray((512, 512, ori_shape[2]), dtype=np.float32)
        # test_data[:, :, :] = nii_datas
        test_data = nii_datas


        result = np.ndarray((DATA_SIZE[0], DATA_SIZE[1], DATA_SIZE[2]))

        n = 0
        idx = []

        # w = test_data[280:280 + BLOCK_SIZE[0], 234:234 + BLOCK_SIZE[1], 150:150 + BLOCK_SIZE[2]]
        # array_img = nib.Nifti1Image(w, affine=None)
        # nib.save(array_img, './ct1233.nii.gz')
        # test_1 = np.ndarray((1, 1, BLOCK_SIZE[0], BLOCK_SIZE[1], BLOCK_SIZE[2]), dtype=np.float32)
        # test_1[0,0,:,:,:] = w
        # images = torch.from_numpy(test_1)
        # images = images.to(device=device, dtype=torch.float32)
        # test_pred, _, _ = net(images)
        # print('test_pred', test_pred.shape)
        # # 输出的图像大小 1*2*1*100*100
        # test_pred = nn.Softmax(1)(test_pred)
        # test_output1 = test_pred.squeeze().cpu().detach().numpy()
        # test_pred_argmax = np.argmax(test_output1, axis=0).astype("float")
        # array_img = nib.Nifti1Image(test_pred_argmax, affine=None)
        # nib.save(array_img, './tes1233.nii.gz')

        for i in range(0, DATA_SIZE[0], int(BLOCK_SIZE[0] / 2)):
            for j in range(0, DATA_SIZE[1], int(BLOCK_SIZE[1] / 2)):
                for k in range(0, DATA_SIZE[2], int(BLOCK_SIZE[2] / 2)):

                    if (i + BLOCK_SIZE[0] > DATA_SIZE[0]):
                        i = DATA_SIZE[0] - BLOCK_SIZE[0]
                    if (j + BLOCK_SIZE[1] > DATA_SIZE[1]):
                        j = DATA_SIZE[1] - BLOCK_SIZE[1]
                    if (k + BLOCK_SIZE[2] > DATA_SIZE[2]):
                        k = DATA_SIZE[2] - BLOCK_SIZE[2]

                    if n < Batch_size:#四个一批次预测，
                        patch = test_data[i:i + BLOCK_SIZE[0], j:j + BLOCK_SIZE[1], k:k + BLOCK_SIZE[2]]
                        #这个时候resize有无必要
                        patch = T.resize(patch.astype("float"), BLOCK_SIZE)
                        #多做一步resize,确保输入一样
                        # array_img = nib.Nifti1Image(patch, affine=None)
                        #切块检查
                        # nib.save(array_img, './tespatch.nii.gz')
                        print('切块尺寸：',patch.shape)
                        if ((patch.shape[0] == BLOCK_SIZE[0]) and (patch.shape[1] == BLOCK_SIZE[1]) and (patch.shape[2] == BLOCK_SIZE[2])):
                            print('判断patch尺寸和block size的大小：', BLOCK_SIZE[0], BLOCK_SIZE[1], BLOCK_SIZE[2], i, j, k)

                            #前两个位置是索引，后面为索引切片对应的层/
                            test_images[n, 0, 0:BLOCK_SIZE[0], 0:BLOCK_SIZE[1], 0:BLOCK_SIZE[2]] = patch

                            #拼接
                            curr_idx = [i,j,k]
                            print('curr_idx:',curr_idx)

                            idx.append(curr_idx)
                            print('idx,i,j,k:',idx,i,j,k)
                            n += 1
                            #如果切块数不能被整除，n还是会向上叠加，但是在没有被整除的部分，idx列表的内容不会继续增加，则传入else的idx不为四，
                    else:
                        # print('test_images;test_data',test_images.shape,test_data.shape)
                        n = 0
                        #超过4后，重新赋值，开始预测
                        # idx = []

                        # array_img = nib.Nifti1Image(test_images[0, 0, :, :, :], affine=None)
                        # nib.save(array_img, './tespatch.nii.gz')

                        images = torch.from_numpy(test_images)
                        images = images.to(device=device, dtype=torch.float32)

                        test_pred, _, _= net(images)

                        # test_pred = net(images)
                        # print('test_pred',test_pred.shape)
                        # 输出的图像大小 1*2*1*100*100
                        test_pred = nn.Softmax(dim=1)(test_pred)
                        print('test_pred_after_softmax:',test_pred.shape)
                        test_output1 = test_pred.squeeze().cpu().detach().numpy()
                        print('test_output',test_output1.shape)
                        #筛选出网络预测值Prediction中最大概率值对应的class ID,axis=1 为对应的类的维度，即n
                        test_pred_argmax = np.argmax(test_output1, axis=1).astype("float")
                        print('test_pred_argmax',test_pred_argmax.shape)
                        # 拼接结果
                        for id_num in range(Batch_size):
                            print('id_num',id_num)
                            i1,j1,k1 = idx[id_num]
                            print('i1,j1,k1:',i1,j1,k1)

                            #拼接 +=
                            result[i1:i1 + BLOCK_SIZE[0], j1:j1 + BLOCK_SIZE[1], k1:k1 + BLOCK_SIZE[2]] += test_pred_argmax[id_num]

                            print('the shape of test_per_argmax:',test_pred_argmax.shape,'result',result.shape)
                        idx = []
        # result = normalize_output(result)
        print('result.shape',result.shape)
        result = downscale_scan(result, sz=512, z_length=h)
        print('result.shape', result.shape)
        result[result>0] = 1
        # result = universal3Dlargestregion(result)

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
    # net = torch.load('./output/stage1/{}/best1.pkl'.format(args.save))
    net = torch.load('./output/stage1/2023_4_27/best.pkl')
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
    print('total time',time2-time1)



