import glob
import os
import imageio
import cv2
from skimage import measure
import config
from utils.common import *
from preprocess_operation import universal3Dlargestregion

from sklearn.utils import shuffle

def imgpre(img_data,sizexy):
    upper = -300
    lower = -1024

    #对应的把数据放缩到最大最小之间

    img_arra = np.clip(img_data, lower, upper)
    #最大最小归一化
    img_arra = (img_arra - lower) / (upper - lower) * 255

    img_arra = (img_arra.astype(np.uint8))   #整型变换--运行速度？
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
        #阈值截断
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
    img_data[img_data < -1024] = -1024  # 将-3000+归到-1024   (需要的HU值大于-1024)
    # img_data[img_data > -300] = -300  #2022.5.23添加
    print("preprocessing airway done")
    # print('the value of prepocessing image',img_data[:,:,100])

    return img_data

def resize(scan, resizeshape, order=0):
    """采样三维矩阵，scan:nii数组；sz:xy轴的维度；z_length：z轴切片数；order：采样方式，0为最近邻采样"""
    if (len(resizeshape) == 3):
        zoom_seq = np.array([resizeshape[0], resizeshape[1], resizeshape[2]], dtype='float') / np.array(scan.shape,
                                                                                                        dtype='float')
    else:
        zoom_seq = np.array([resizeshape[0], resizeshape[1]], dtype='float') / np.array(scan.shape,
                                                                                        dtype='float')
    # zoom_seq = np.array([resizeshape[0], resizeshape[1], resizeshape[2]], dtype='float') / np.array(scan.shape, dtype='float')
    s = ndimage.interpolation.zoom(scan, zoom_seq, order=order, prefilter=order)

    return s

def read_nii(path):
    img_file = nib.load(path)
    array = img_file.get_fdata()
    return array

def downscale_scan(scan, sz=512, z_length=64, order=0):
    """下采样三维矩阵，scan:nii数组；sz:xy轴的维度；z_length：z轴切片数；order：采样方式，0为最近邻采样"""
    zoom_seq = np.array([sz, sz, z_length]) / np.array(scan.shape)
    s = ndimage.interpolation.zoom(scan, zoom_seq, order=order, prefilter=order)
    return s

# def random_sample(img_array, label_array, data_size, patch_size, ct_num, nums, path, modalitys):
#     # array = [array1,array2]
#     # for num in range(nums):
#     num = 0
#     while (num < nums):
#         startx = random.randint(0, data_size[0] - patch_size[0])
#         starty = random.randint(0, data_size[1] - patch_size[1])
#         startz = random.randint(0, data_size[2] - patch_size[2])
#         endx = startx + patch_size[0]
#         endy = starty + patch_size[1]
#         endz = startz + patch_size[2]
#         patch1 = img_array[startx:endx, starty:endy, startz:endz]  # 本质上为索引
#         patch2 = label_array[startx:endx, starty:endy, startz:endz]
#         a = np.where(patch2 > 0)
#         if (len(a[0]) > 100):
#             o = nib.Nifti1Image(patch1, affine=None)  # 将索引后的转换为图像
#             os.makedirs(os.path.join(path, modalitys[0]), exist_ok=True)
#             os.makedirs(os.path.join(path, modalitys[1]), exist_ok=True)
#             nib.save(o, os.path.join(path, modalitys[0], 'patch_' + 'ct' + str(ct_num) + '_' + str(num) + '.nii.gz'))
# 
#             b = nib.Nifti1Image(patch2, affine=None)  # 将索引后的转换为图像
#             nib.save(b, os.path.join(path, modalitys[1], 'patch_' + 'ct' + str(ct_num) + '_' + str(num) + '.nii.gz'))
# 
#             num += 1
def setDir(filepath):
    '''
    如果文件夹不存在就创建，如果文件存在就清空！
    :param filepath:需要创建的文件夹路径
    :return:
    '''
    if os.path.exists(filepath):
        print('路径{}已存在'.format(filepath))
    else:
        print('创建目标路径')
        os.makedirs(filepath)

def random_sample(img_array, label_array, data_size, patch_size, nums, path, modalitys,image_name):
    # array = [array1,array2]
    # for num in range(nums):
    num = 0
    while (num < nums):
        startx = random.randint(0, data_size[0] - patch_size[0])
        starty = random.randint(0, data_size[1] - patch_size[1])
        startz = random.randint(0, data_size[2] - patch_size[2])
        endx = startx + patch_size[0]
        endy = starty + patch_size[1]
        endz = startz + patch_size[2]
        patch1 = img_array[startx:endx, starty:endy, startz:endz]  # 本质上为索引
        patch2 = label_array[startx:endx, starty:endy, startz:endz]
        a = np.where(patch2 > 0)
        # print('判断条件',a)
        # setDir(os.path.join(path, modalitys[0]))
        # setDir(os.path.join(path, modalitys[1]))
        os.makedirs(os.path.join(path, modalitys[0]), exist_ok=True)
        os.makedirs(os.path.join(path, modalitys[1]), exist_ok=True)

        if (len(a[0]) > 0.02*patch1.shape[0]*patch1.shape[1]*patch1.shape[2]):



            o = nib.Nifti1Image(patch1, affine=None)  # 将索引后的转换为图像
            nib.save(o, os.path.join(path, modalitys[0],  str(image_name)+'_patch_' + '_' + str(num) + '.nii.gz'))

            b = nib.Nifti1Image(patch2, affine=None)  # 将索引后的转换为图像
            nib.save(b, os.path.join(path, modalitys[1],  str(image_name)+'_patch_' + '_' + str(num) + '.nii.gz'))

            num += 1


def slide_window_sample(img_array, label_array, data_size, patch_size, path, modalitys, img_name, stride=0.5, ):
    num = 0
    strides = np.array(patch_size, dtype=np.float) * stride
    print('strids', strides)
    for i in range(0, data_size[0], int(strides[0])):
        for j in range(0, data_size[1], int(strides[1])):
            for k in range(0, data_size[2], int(strides[2])):
                if k + patch_size[2] > data_size[2]:
                    k = data_size[2] - patch_size[2]
                # print('coordinates',i,j,k,)
                # num +=1
                patch_img = img_array[i:i + patch_size[0], j:j + patch_size[1], k:k + patch_size[2]]
                patch_label = label_array[i:i + patch_size[0], j:j + patch_size[1], k:k + patch_size[2]]
                a = np.where(patch_label > 0)
                if len(a[0]) > 100:
                    # print("dddddddddddddddddddddddddddddddddddddddddddddddddddd")
                    # if "lid" in xxx:
                    if not os.path.exists(os.path.join(path, modalitys[0])):
                        os.makedirs(os.path.join(path, modalitys[0]), exist_ok=True)
                    if not os.path.exists(os.path.join(path, modalitys[1])):
                        os.makedirs(os.path.join(path, modalitys[1]), exist_ok=True)

                    o = nib.Nifti1Image(patch_img, affine=None)  # 将索引后的转换为图像
                    nib.save(o, os.path.join(path, modalitys[0], img_name + '_' + 'patch_' + str(num) + '.nii.gz'))

                    b = nib.Nifti1Image(patch_label, affine=None)  # 将索引后的转换为图像
                    nib.save(b, os.path.join(path, modalitys[1], img_name + '_' + 'patch_' + str(num) + '.nii.gz'))

                    num += 1

def largestestregion2(img_arra, shuzhi = 1,save_num = 1):
    """找最大连通域,shuzhi:将最大联通域置为什么值,如气管为6"""
    deal= (img_arra.astype(np.uint8))
    #deal=np.asarray(img)
    deal[deal > 0] = 1  # 转换为二值图像
    del_mask = save_largestregion(deal,save_num = save_num)
    out = deal * del_mask * shuzhi # 原始白框乘上最大白框*255   修改为乘6(为了肺气管的mask)
    out = (out.astype(np.uint8))  # 找出了最大的白框部分
    return out

def save_largestregion(deal, save_num):
    """找到3D丈量最大连通域,输出为值为1的mask.deal:输入的3D张量"""
    labels = measure.label(deal, connectivity=3)  # 找白色区域的8连通域，并给予每个连通域标号，connectivity为ndarry的维数，三维数组故为3
    jj = measure.regionprops(labels)  # 这里是取得labels的属性，属性有许多
    save_indexs = []
    region_size = []
    num = labels.max()  # 找白色部分的连通域有几个
    print('白色区域数量', num)
    del_array = np.array([0] * (num + 1))
    for k in range(num):  # 这里是找最大的那个白色连通域的标号save
        save_indexs.append(k + 1)
        region_size.append(jj[k].area)
    region_size, save_indexs = zip(*sorted(zip(region_size, save_indexs)))
    print(region_size)
    print(save_indexs)
    for i in range(save_num):
        index = int(-(i + 1))
        print(index)
        del_array[save_indexs[index]] = 1
    deal = del_array[labels]
    return deal


def read_img_array(img):
    # 单例输入，序号提取
    print(img)
    # 文件名提取
    img_name = img[img.rindex("\\") + 1:]
    img_name = img_name.split('.nii.gz')[0]

    # img的预处理部分
    img_array = read_nii(img)
    h = img_array.shape[2]
    img_array = downscale_scan(img_array, sz=512, z_length=256)
    img_array = imgpre(img_array, 512)
    img_array = img_array.astype('float')
    # img_nii = nib.Nifti1Image(img_array, affine=None)  # 将索引后的转换为图像

    # nib.save(img_nii, img_name+'.nii.gz')

    img_array[img_array < -1024] = -1024
    img_array[img_array > -300] = -300
    # print('check------1',img_array.min(),img_array.max())
    img_array = norm_img(img_array)
    # print('check------2', img_array.min(), img_array.max())

    img_array = normalize_img(img_array)
    # print('check------3', img_array.min(), img_array.max())

    return img_name, img_array

def read_label_array(label):
    label_name = label[label.rindex("\\") + 1:]
    label_name = label_name.split('.nii.gz')[0]
    label_array = read_nii(label)  # 索引顺序和标签顺序差1
    label_shape = label_array.shape
    # label_array = np.reshape(label_array, (512, 512, label_shape[2], 1))
    # label_array = largestestregion2(label_array, shuzhi=1, save_num=1)
    label_array = largestestregion2(label_array, shuzhi=1, save_num=1)
    label_array = downscale_scan(label_array, sz=512, z_length=256, order=0)

    label_array[label_array > 0] = 1
    label_array[label_array <= 0] = 0
    return label_name, label_array




def train_data(data_path,patch_path):
    # image_list = get_img_list(data_path)
    # label_list = get_label_list()
    image_list = os.listdir(os.path.join(data_path,'data'))
    label_list = os.listdir(os.path.join(data_path,'label'))

    # print('原图个数：label个数：', len(image_list), len(label_list))
    # print('bb_00', image_list)
    # print('cc_00', label_list)

    image_list, label_list = shuffle(image_list, label_list, random_state=0)
    # 绝对路径的加载
    print('aa', len(image_list), len(label_list))
    print('bb', image_list)
    print('cc', label_list)
    k = 0
    for img, label in zip(image_list, label_list):

        if k < int(len(image_list) * 8 / 10):
            print('训练集加载')
            # patch_path = r"D:\gzq\airway_patch_vision2\train"
            patch_path_1 = os.path.join(patch_path,'train')
            img_name, img_array = read_img_array(os.path.join(data_path, 'data', img))
            label_name, label_array = read_label_array(os.path.join(data_path, 'label', label))

            print(img_array.shape,label_array.shape,'check')
            assert img_array.shape == label_array.shape

            # label_array_shape_z = label_array.shape[2]
            # img_array = downscale_scan(img_array, sz=512, z_length=label_array_shape_z)
            modalitys = ['data', 'label']
            slide_window_sample(img_array, label_array, img_array.shape, [128, 128, 32], patch_path, modalitys, img_name,
                                stride=0.4)
            # slide_window_sample(img_array, label_array, img_array.shape, args.crop_size, patch_path, modalitys,
            #                     img_name,
            #                     stride=0.5)
            # random_sample(img_array, label_array, img_array.shape, [128, 128, 32], 50, patch_path_1, modalitys, img_name)
        else:
            print('验证集加载')
            # patch_path = r"D:\gzq\airway_patch_vision2\val"
            patch_path_2 = os.path.join(patch_path,'val')
            img_name, img_array = read_img_array(os.path.join(data_path,'data',img))
            label_name, label_array = read_label_array(os.path.join(data_path,'label',label))

            print(img_array.shape, label_array.shape, 'check')
            assert img_array.shape == label_array.shape

            # label_array_shape_z = label_array.shape[2]
            # img_array = downscale_scan(img_array, sz=512, z_length=label_array_shape_z)
            modalitys = ['data', 'label']
            slide_window_sample(img_array, label_array, img_array.shape, [128, 128, 32], patch_path, modalitys, img_name,
                                stride=0.4)
            # random_sample(img_array, label_array, img_array.shape, [128, 128, 32], 50, patch_path_2, modalitys, img_name)

        k += 1


def train_data_manual(data_path,patch_path):

        print('训练集加载')
        train_list = os.listdir(os.path.join(data_path,'data','train'))
        for train_name in train_list:
            # patch_path = r"D:\gzq\airway_patch_vision2\train"
            patch_path_1 = os.path.join(patch_path,'train')

            img_name, img_array = read_img_array(os.path.join(data_path, 'data','train',train_name))
            label_name, label_array = read_label_array(os.path.join(data_path,'label','train', train_name))

            print(img_array.shape, label_array.shape, 'check')
            # img_array_shape_z = img_array.shape[2]
            # label_array = np.resize(label_array, img_array.shape)

            assert img_array.shape[0] == label_array.shape[0]
            assert img_array.shape[1] == label_array.shape[1]
            assert img_array.shape[2] == label_array.shape[2]

            modalitys = ['data', 'label']
            slide_window_sample(img_array, label_array, img_array.shape, [128, 128, 32], patch_path_1, modalitys, img_name,
                                stride=0.4)
            # random_sample(img_array, label_array, img_array.shape, [128, 128, 32], 100, patch_path_1, modalitys, img_name)

        print('验证集加载')

        val_list = os.listdir(os.path.join(data_path,'data','val'))
        for val_name in val_list:
            # patch_path = r"D:\gzq\airway_patch_vision2\val"
            patch_path_2 = os.path.join(patch_path,'val')
            img_name, img_array = read_img_array(os.path.join(data_path, 'data', 'val', val_name))
            label_name, label_array = read_label_array(os.path.join(data_path, 'label', 'val', val_name))

            print(img_array.shape, label_array.shape, 'check')
            img_array_shape_z = img_array.shape[2]
            label_array = np.resize(label_array, img_array.shape)

            assert img_array.shape[0] == label_array.shape[0]
            assert img_array.shape[1] == label_array.shape[1]
            assert img_array.shape[2] == label_array.shape[2]


            modalitys = ['data', 'label']
            slide_window_sample(img_array, label_array, img_array.shape, [128, 128, 32], patch_path_2, modalitys, img_name,
                                stride=0.4)
            # random_sample(img_array, label_array, img_array.shape, [128, 128, 32], 100, patch_path_2, modalitys, img_name)


if __name__ == '__main__':
    # 分别读取，保存的位置设为一样的
    args = config.args
    data_path = r"E:\exact09"
    patch_path = r"E:\exact09\patch2023_4_27"
    train_data_manual(data_path, patch_path)
