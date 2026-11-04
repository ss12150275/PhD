import cv2
import numpy as np
from scipy import ndimage
import nibabel as nib
from skimage import data,exposure
import matplotlib.pyplot as plt
import time
import math
from skimage import measure


def otsu1(img):           #######适用于普通图像的 OTSU 法
    resize = img.shape
    min = np.min(img)
    max = np.max(img)
    otsu = np.ndarray(resize, dtype=np.int)
    # edg_y = np.ndarray(resize, dtype=np.int)
    for i in range(resize[2]):
        image = img[:, :, i]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        th = cv2.threshold(gray,0,max,cv2.THRESH_OTSU)
        otsu[:, :, i] = th
    return otsu

def laplacian(img):    ######拉普拉斯变换
    """计算图像的 laplacian，适用于三维图像
                                Inputs:
                                    Required:
                                        img: 二维图像
                                Returns:
                                    final_img:经过拉普拉斯处理后的图像
                    """
    img = np.array(img, dtype='uint8')
    resize = img.shape
    final_img = np.ndarray(resize, dtype=np.int)
    # edg_y = np.ndarray(resize, dtype=np.int)
    for i in range(resize[0]):
        image = img[i, :, :]
        gray_lap = cv2.Laplacian(image, cv2.CV_16S, ksize=1)
        dst = cv2.convertScaleAbs(gray_lap)
        final_img[i, :, :] = dst
    return final_img

def hist_cal(img,min,max):
    """计算图像的 直方图，适用于二维图像
                            Inputs:
                                Required:
                                    img: 直方图均衡化，二维图像
                                    min：图像最小值
                                    max：图像最大值
                            Returns:
                                histogram:直方图统计值
                """
    img = np.array(img)
    value_range = max-min + 1
    # print(min,max,value_range)
    heigh,width=img.shape
    histogram=np.zeros(int(value_range))
    for i in range(heigh):
        for j in range(width):
            gray = img[i][j]
            histogram[int(gray-min)] += 1#出现某个像素值则加1，最终统计出各个像素值的频数

    return histogram

def EqualizeHistogram(img,histogram,min,max,LIMIT = 200):
    """直方图均衡化，二维图像
                        Inputs:
                            Required:
                                img: 直方图均衡化，二维图像
                                min：图像最小值
                                max：图像最大值
                                LIMIT：对比度的限制值
                        Returns:
                            LUT,Equalize_img:前者用于AHE、CLAHE的插值，后者用于直方图均衡化直接使用
            """
    img = np.array(img)
    heigh, width = img.shape
    pixcount=heigh * width
    dataOfSrc = img
    dataOfDst=np.zeros([heigh, width])
    steal = 0
    value_range = int(max - min + 1)
    # for k in range(value_range):
    #     if histogram[k]>LIMIT:
    #         histogram[k]=LIMIT
    #         steal+= histogram[k]-LIMIT
    # bonus=math.floor(steal/value_range)
    # for l in range(value_range):
    #     histogram[l]+=bonus
    pro_den=histogram/pixcount
    sum=0
    LUT=np.zeros(value_range)
    for i in range(value_range):
        sum+=pro_den[i]
        LUT[i]=sum * (value_range-1)
    for m in range(heigh):
        for n in range(width):
            dataOfDst[m][n]=LUT[int(dataOfSrc[m][n]-min)]  ###改动
    Equalize_img=dataOfDst.reshape([heigh, width])
    Equalize_img=Equalize_img.astype(np.uint8)
    return LUT,Equalize_img

def CLAHE(block_number,img_gray):
    """限制对比度的直方图均衡化，分块处理 二维图像
                    Inputs:
                        Required:
                            block_number: 切分的块数
                            img_gray：输入的原图像数据
                    Returns:
                        img_ahe_liner:处理后的图像数据
        """
    heigh, width = img_gray.shape
    img_ahe_liner = np.zeros([heigh, width])
    # print('shape',img_ahe_liner.shape)
    block_height = heigh // block_number
    block_width = width // block_number
    # print('height,width',heigh, width,block_height,block_width)
    min = int(np.min(img_gray))
    max = int(np.max(img_gray))
    value_range = int(max - min + 1)
    # print(value_range)

    all_LUT = np.zeros([block_number, block_number, value_range])  # 存储累积分布函数
    # print(all_LUT.shape)
    # 求得各个块的累积分布函数
    for i in range(block_number):
        for j in range(block_number):
            block = img_gray[i * block_height:(i + 1) * block_height, j * block_width:(j + 1) * block_width]
            hist_block = hist_cal(block,min,max)
            all_LUT[i, j], Equalize_hist = EqualizeHistogram(block, hist_block,min,max)
    for m in range(heigh):
        for n in range(width):
            # four coners
            # 在角上的像素点为累积分布函数得到的值
            if m <= block_height / 2 and n <= block_width / 2:
                block_m = 0
                block_n = 0
                img_ahe_liner[m, n] = all_LUT[block_m, block_n][int(img_gray[m, n]-min)]  ###
            elif m <= block_height / 2 and n >= width - block_width / 2:
                block_m = 0
                block_n = block_number - 1
                img_ahe_liner[m, n] = all_LUT[block_m, block_n][int(img_gray[m, n]-min)]
            elif n <= block_width / 2 and m >= heigh - block_height / 2:
                block_m = block_number - 1
                block_n = 0
                img_ahe_liner[m, n] = all_LUT[block_m, block_n][int(img_gray[m, n]-min)]
            elif m >= heigh - block_height / 2 and n >= width - block_width / 2:
                block_m = block_number - 1
                block_n = block_number - 1
                img_ahe_liner[m, n] = all_LUT[block_m, block_n][int(img_gray[m, n]-min)]
            # four edges except coners
            # 四个边采取线性插值
            elif n <= block_width / 2:
                block_m_1 = math.floor((m - block_height / 2) / block_height)
                block_n_1 = 0
                block_m_2 = block_m_1 + 1
                block_n_2 = block_n_1
                u = np.float((m - (block_m_1 * block_height + block_height / 2)) / (block_height))
                v = 1 - u
                img_ahe_liner[m, n] = v * all_LUT[block_m_1, block_n_1][int(img_gray[m, n]-min)] + \
                                      u * all_LUT[block_m_2, block_n_2][int(img_gray[m, n]-min)]
            elif m <= block_height / 2:
                block_m_1 = 0
                block_n_1 = math.floor((n - block_width / 2) / block_width)
                block_m_2 = block_m_1
                block_n_2 = block_n_1 + 1
                u = np.float((n - (block_n_1 * block_width + block_width / 2)) / (block_width))
                v = 1 - u
                img_ahe_liner[m, n] = v * all_LUT[block_m_1, block_n_1][int(img_gray[m, n]-min)] + \
                                      u * all_LUT[block_m_2, block_n_2][int(img_gray[m, n]-min)]
            elif m >= heigh - block_height / 2:
                block_m_1 = block_number - 1
                block_n_1 = math.floor((n - block_width / 2) / block_width)
                block_m_2 = block_m_1
                block_n_2 = block_n_1 + 1
                u = np.float((n - (block_n_1 * block_width + block_width / 2)) / (block_width))
                v = 1 - u
                img_ahe_liner[m, n] = v * all_LUT[block_m_1, block_n_1][int(img_gray[m, n]-min)] + \
                                      u * all_LUT[block_m_2, block_n_2][int(img_gray[m, n]-min)]
            elif n >= width - block_width / 2:
                block_m_1 = math.floor((m - block_height / 2) / block_height)
                block_n_1 = block_number - 1
                block_m_2 = block_m_1 + 1
                block_n_2 = block_n_1
                u = np.float((m - (block_m_1 * block_height + block_height / 2)) / (block_height))
                v = 1 - u
                img_ahe_liner[m, n] = v * all_LUT[block_m_1, block_n_1][int(img_gray[m, n]-min)] + \
                                      u * all_LUT[block_m_2, block_n_2][int(img_gray[m, n]-min)]
            # 中间区域采取双线性插值
            else:
                block_m_1 = math.floor((m - block_height / 2) / block_height)
                block_n_1 = math.floor((n - block_width / 2) / block_width)
                block_m_2 = block_m_1 + 1
                block_n_2 = block_n_1 + 1
                u = np.float((m - (block_m_1 * block_height + block_height / 2)) / (block_height))
                v = np.float((n - (block_n_1 * block_width + block_width / 2)) / (block_width))
                img_ahe_liner[m, n] = (1 - u) * (1 - v) * all_LUT[block_m_1, block_n_1][int(img_gray[m, n]-min)] + \
                                      v * (1 - u) * all_LUT[block_m_1, block_n_2][int(img_gray[m, n]-min)] + \
                                      u * (1 - v) * all_LUT[block_m_2, block_n_1][int(img_gray[m, n]-min)] + \
                                      u * v * all_LUT[block_m_2, block_n_2][int(img_gray[m, n]-min)]
    return img_ahe_liner.astype(np.uint8)

def clahe_img(array,block_number=8):
    """限制对比度的直方图均衡化，用于三维数据
        Inputs:
            Required:
                array:输入的数据矩阵
                block_numbers:切分的块数
        Returns:
            clahes:处理后的图像数据
        """
    dim = array.shape
    # print(dim)

    clahes = np.ndarray((dim[0], dim[1], dim[2]))

    for i in range(dim[2]):  ###dim[2]
        img = array[:, :, i]
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # clahe_img = clahe.apply(img)
        clahe = CLAHE(block_number, img)
        # print(clahe.shape)
        clahes[:, :, i] = clahe
    return clahes

def otsu(gray,min_value = None,max_value = None):
    """OTSU 图像增强算法
            Inputs:
                Required:
                    array: 输入的数据矩阵
                    min_value：设置的下限值
                    max_value：设置的上限值
            Returns:
                final_img:处理后的图像数据
            """
    pixel_number = gray.shape[0] * gray.shape[1]
    mean_weigth = 1.0/pixel_number
    max = np.max(gray)
    min = np.min(gray)
    value_range = max - min + 1
    # 发现bins必须写到257，否则255这个值只能分到[254,255)区间
    his = hist_cal(gray,min,max)

    # print(his)
    final_thresh = -1
    final_value = -1
    intensity_arr = np.arange(min,max + 1)
    # print('inrensity',intensity_arr)
    for t in range(1,value_range):      #bins[1:-1]: # This goes from 1 to 254 uint8 range (Pretty sure wont be those values)
        pcb = np.sum(his[:t])
        pcf = np.sum(his[t:])
        Wb = pcb * mean_weigth
        Wf = pcf * mean_weigth

        mub = np.sum(intensity_arr[:t]*his[:t]) / float(pcb)
        muf = np.sum(intensity_arr[t:]*his[t:]) / float(pcf)
        # print('值',Wb,Wf,mub,muf)
        #print mub, muf
        value = Wb * Wf * (mub - muf) ** 2
        # print('value',value)

        if value > final_value:
            final_thresh = intensity_arr[t]
            final_value = value
    final_img = gray.copy()
    print(final_thresh)
    if min_value is None:
        min_thresh = min
    else:
        min_thresh = min_value
    if max_value is None:
        max_thresh = max
    else:
        max_thresh = max_value

    final_img[gray > final_thresh] = min_thresh
    final_img[gray < final_thresh] = max_thresh
    # cv.imshow("otsu image", final_img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    return final_img

def gamma_transformation(input_image, c, gamma):
    """伽马变换
                Inputs:
                    Required:
                        input_image: 输入的数据矩阵
                        c：伽马变换超参数
                        gamma：伽马值
                Returns:
                    output_imgae:处理后的图像数据
    """
    input_image_cp = np.copy(input_image)  # 输入图像的副本

    output_imgae = c * np.power(input_image_cp.astype(int), gamma)  # 输出图像

    return output_imgae

def window_level(ct_array, normal=False):
    """
    return: trucated image according to window center and window width
    and normalized to [0,1]
    """
    windowWidth = np.max(ct_array) - np.min(ct_array)
    windowCenter = (np.max(ct_array) - np.min(ct_array)) / 2
    minWindow = float(windowCenter) - 0.5 * float(windowWidth)
    newimg = (ct_array - minWindow) / float(windowWidth)
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    if not normal:
        newimg = (newimg * 255).astype('uint8')
    return newimg

def downscale_scan(scan, sz=512, z_length=64, order=0):
    """下采样三维矩阵，scan:nii数组；sz:xy轴的维度；z_length：z轴切片数；order：采样方式，0为最近邻采样"""
    zoom_seq = np.array([sz, sz, z_length]) / np.array(scan.shape)
    s = ndimage.interpolation.zoom(scan, zoom_seq, order=order, prefilter=order)
    return s

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

def find_3D_object_voxel_list(vol):
    """This function creates a centerline from the segmented volume (vol)

    Inputs:
        Required:
            vol: 3D binary volume where -
                0: background
                1: object to be skeletonized/centerline extracted
    Returns:
        nz_x, nz_y, nz_z
            lists of non-zero x, y, z co-ordinates

    Dependencies:
        numpy
    """
    nz_coords = np.nonzero(vol)
    nz_x, nz_y, nz_z = [nz_coords[i].tolist() for i in range(3)]
    return nz_x, nz_y, nz_z

def imgpre(img_data,sizexy):
    upper = -300
    lower = -1024
    img_arra = np.clip(img_data, lower, upper)
    img_arra = (img_arra - lower) / (upper - lower) * 255
    img_arra = (img_arra.astype(np.uint8))
    # print('the value of image',img_arra[:,:,100])
    # f1 = nib.Nifti1Image(img_arra, affine=None)
    # nib.save(f1, './origin_pre.nii.gz')
    a = img_data.shape[2]
    nii_arr = np.zeros([sizexy, sizexy, a])
    # save_index = 1
    for i in range(a):  # 二维操作和三维操作速度略快一点点
        im = img_arra[:, :, i]
        # im = ndimage.rotate(im, -90)  # 角度负数为顺时针旋转
        # im = im[:, ::-1]  # 列翻转
        # 预处理程序开始
        im3 = cv2.resize(im, (sizexy, sizexy), interpolation=cv2.INTER_NEAREST)  # cv的resize可设置插值算法

        #################################
        deal = im3.copy()
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
        # imageio.imwrite('./preprocessing/im1111' + '%06d' % i + '.tif', im1)
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

if __name__=="__main__":
    nii_path = 'F:/airway_seg/nii/ChenQiaoHua.nii.gz'

    datas = nib.load(nii_path)
    affine = datas.affine
    data_file = datas.get_data()
    dim = data_file.shape
    print(dim)

    #
    # temp = clahe_img(data_file,block_number=8)
    new_data = imgpre(data_file, 512)
    new_data[new_data > -700] = -700
    new_data[new_data < -1024] = -1024
    revise_data = 0 - new_data
    # out_name = 'F:/airway_seg/nii/rrr.nii.gz'
    # array_img = nib.Nifti1Image(revise_data, affine=affine)
    # nib.save(array_img, out_name)

    x3, y3, z3 = find_3D_object_voxel_list(revise_data)
    print('边界值信息', min(x3), max(x3), min(y3), max(y3), min(z3), max(z3))

    # revise_data_region = new_data[min(x3): max(x3), min(y3): max(y3), min(z3): max(z3)]

    # final_data1 = window_level(data_file,normal=True)
    time_1 = time.time()
    final_data2 = clahe_img(new_data,block_number=1)
    time_2 = time.time()
    print('clahe time',time_2 - time_1)

    # revise_data[min(x3): max(x3), min(y3): max(y3), min(z3): max(z3)] = final_data2

    # win_data = window_level(data_file,normal=True)


######取反操作
    # revise_data = 0 - new_data
    # x3, y3, z3 = find_3D_object_voxel_list(revise_data)
    # print('边界值信息', min(x3), max(x3), min(y3), max(y3), min(z3), max(z3))
    # print('最值',np.max(revise_data),np.min(revise_data))

    out_name = 'F:/airway_seg/nii/clahe_1.nii.gz'
    array_img = nib.Nifti1Image(final_data2, affine=affine)
    nib.save(array_img, out_name)

