# -*- coding:utf-8 -*-
from utils.common import *
from scipy import ndimage
import numpy as np
from torchvision import transforms as T
import torch,os
from torch.utils.data import Dataset, DataLoader


class Lits_DataSet(Dataset):
    def __init__(self, crop_size,resize_scale, dataset_path,mode=None,deep_supervise = False):
        self.crop_size = crop_size
        self.resize_scale=resize_scale
        self.dataset_path = dataset_path
        self.mode = mode
        self.supervision = deep_supervise
        self.n_labels = 2

        #数据读取
        if mode=='train':
            #####调用该函数从txt文件中读取相应的文件名称并存放到filename_list列表中
            # self.filename_list = load_file_name_list(os.path.join(dataset_path,'train_name_list.txt'))
            ######该函数直接从
            # self.filename_list = load_file_name_list2(self.dataset_path)[0:12]
            self.filename_list = load_file_name_list2(self.dataset_path)
        elif mode =='val':
            # self.filename_list = load_file_name_list(os.path.join(dataset_path, 'val_name_list.txt'))
            self.filename_list = load_file_name_list2(self.dataset_path)
        elif mode == 'test':
            self.filename_list = load_file_name_list2(self.dataset_path)
        else:
            raise TypeError('Dataset mode error!!! ')


    def __getitem__(self, index):
        # data = data.transpose(0, 4, 1, 2, 3)
        # target = target.transpose(0, 4, 1, 2, 3)
        if self.mode == "test":
            #test没有label，不做one-hot
            data, patchname = self.get_train_batch_by_index(crop_size=self.crop_size, index=index,
                                                                    resize_scale=self.resize_scale)
            return torch.from_numpy(data), patchname
        else:
            if self.supervision:
                data, target, target2, patchname = self.get_train_batch_by_index(crop_size=self.crop_size, index=index,
                                                                        resize_scale=self.resize_scale)
                return torch.from_numpy(data), torch.from_numpy(target), torch.from_numpy(target2)
            else:
                data, target, patchname = self.get_train_batch_by_index(crop_size=self.crop_size, index=index,
                                                                        resize_scale=self.resize_scale)
                return torch.from_numpy(data), torch.from_numpy(target)



    def __len__(self):
        return len(self.filename_list)

    def get_train_batch_by_index(self, crop_size, index, resize_scale=1):#返回文件名，
        # print('标签',index)
        resize = (crop_size[0], crop_size[1], crop_size[2])
        train_imgs = np.zeros([1, crop_size[0], crop_size[1], crop_size[2]])
        if self.mode == "test":
            '''#测试下，仅仅获取图的array和文件名'''
            img, patchname = self.get_np_data_3d2(self.filename_list[index], resize)
            sub_img = img
            '''扩展一个维度后，给对应的维度赋值'''
            train_imgs[0] = sub_img
            # print('得到图像')
            return train_imgs.astype("float"), patchname
        else:
            '''训练和验证的时候，取原图的array和label以及文件名，以及对label进行one-hot编码'''
            resize = (crop_size[0], crop_size[1], crop_size[2])
            # print('1',self.filename_list[index])
            img, label, patchname = self.get_np_data_3d2(self.filename_list[index], resize)
            # print('label.shape',label.shape,img.dtype)
            sub_img = img
            sub_label = label
            # sub_img, sub_label = random_crop_3d(img, label, crop_size)      ######从图像中裁剪相应大小的块，一般我们不需要
            # print('裁剪后的图像大小',sub_img.shape)
            ####在多分类问题中需要做one-hot编码，其目的是使得每一个类别所有通道上该类别的值为1，其他为0
            # print('独热编码前：',sub_label.shape)
            sub_label_onehot = make_one_hot_3d(sub_label, self.n_labels)
            # print('独热编码后：',sub_label_onehot.shape)
            train_labels = sub_label_onehot.transpose(3, 0, 1, 2)
            train_imgs[0] = sub_img
            # print('train.shape',train_imgs.shape)
            if self.supervision:
                return train_imgs.astype("float"), train_labels.astype("float"),sub_label.astype("float"), patchname
            else:
                return train_imgs.astype("float"), train_labels.astype("float"), patchname

            # print('得到图像')

    #暂无调用
    def get_np_data_3d(self, filename, resize_scale=1):
        patchname = filename
        data_np = sitk_read_row(self.dataset_path + '/data/' + filename,
                                resize_scale=resize_scale)
        data_np=norm_img(data_np)        #####对数据进行归一化处理
        label_np = sitk_read_row(self.dataset_path + '/label/' + filename,
                                 resize_scale=resize_scale)
        return data_np, label_np,patchname


    def get_np_data_3d2(self, filename, resize):
        if self.mode == 'test':
            f_img = nib_read_rowdata(self.dataset_path + '/data/' + filename, resize)
            #f_img = nib_read_rowdata(self.dataset_path  + filename, resize)
            # T.warp
            # f_img = normalize_img(f_img)
            dataname = filename
            return f_img, dataname
        else:
            #train和val有label需要多做一步one-hot
            # print('路径校对',self.dataset_path + '/data/' + filename)
            # print('文件名',filename)
            f_img = nib_read_rowdata(self.dataset_path + '/data/' + filename, resize)
            #f_img = nib_read_rowdata(self.dataset_path  + filename, resize)
            # T.warp
            # f_img = normalize_img(f_img)
            # print('the shape of data',f_img.shape)
            label = nib_read_rowlabel(self.dataset_path + '/label/' + filename, resize)
            # print(label.shape,'-----------')
            dataname = filename
            return f_img, label, dataname






####测试代码
# def main():
#     fixd_path  = r'/home/nwy/Music/airway beifen/3DUNet-Pytorch-master/fixed/val'
#     dataset = Lits_DataSet([64, 64, 64],1,fixd_path,mode='val')  #batch size
#     data_loader=DataLoader(dataset=dataset,batch_size=2,num_workers=1, shuffle=True)
    # for batch_idx, (data, target) in enumerate(data_loader):
    #     print('数据大小',data.shape, target.shape)

def main():
    fixd_path  = r'D:/Dada/airwaydata/originaldata/total_airway_data/LIDC-EXACT(public)/EXACT'
    #fixd_path = r'E:\exact09'
    dataset = Lits_DataSet([64, 64, 64],1,fixd_path,mode='val')  #batch size
    data_loader=DataLoader(dataset=dataset,batch_size=2,num_workers=1, shuffle=True)
def process1():
    # from models.as_se_unet import *
    # dataser_dir = r'F:\gzq\ATM_dataset\patch_train'
    # Laplace = Lits_DataSet([128, 128, 32], 1, dataser_dir, mode='train',deep_supervise=True)
    # device = torch.device('cpu' if torch.cuda.is_available() else 'cuda')
    # UnetModel = as_at_UNet3D_deep(1, [16, 32, 64, 96, 128, 256, 512], 2,net_mode='3d').to(device)
    #
    # for data_item in Laplace:
    #     data, one_label, label = data_item
    #     element_set = torch.unique(data)
    #     # print(element_set)
    #     data = data.unsqueeze()
    #     out = UnetModel(data)
    pass
import matplotlib.pyplot as plt
def main2():
    #dataser_dir = r'E:\exact09\patch\train'
    dataser_dir = r'F:\gzq\ATM_dataset\patch_train'
    Laplace = Lits_DataSet([128, 128, 32], 1, dataser_dir, mode='train',deep_supervise=True)
    device = torch.device('cpu' if torch.cuda.is_available() else 'cuda')


    for data_item in Laplace:
        data, one_label, label = data_item
        print('data----',)
        print(data.max(),data.min())
        print(one_label.max(), one_label.min())
        print(label.max(), label.min())
        data=data.cpu().numpy()
        one_label = one_label.cpu().numpy()
        label = label.cpu().numpy()
        print('数据检查',data.shape,one_label.shape,label.shape)
        # plt.subplot(3,1,1)
        # plt.imshow(data[0,:,:,0])
        # plt.subplot(3,1,2)
        # plt.imshow(one_label[1,:,:,0])
        # plt.subplot(3,1,3)
        # plt.imshow(label[:,:,0])
        # plt.show()
        # plt.show(data[0,:,:])
        # print(data.shape, one_label.shape)



if __name__ == '__main__':
    # main()
    # process1()
    main2()