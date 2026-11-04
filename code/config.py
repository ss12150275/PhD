# -*- coding:utf-8 -*-
import argparse


parser = argparse.ArgumentParser(description='Hyper-parameters management')

# Hardware options
parser.add_argument('--n_threads', type=int, default=6,
                    help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')

# data in/out and dataset
parser.add_argument('--trainset_path',default=r'E:\exact09\patch2023_4_27\train',
                    help='fixed trainset root path')
parser.add_argument('--valset_path',default = r'E:\exact09\patch2023_4_27\val',
                    help='fixed valset root path')

parser.add_argument('--dataset_trainpath',default = r'F:\gzq\EXACT\path_dadaway',
                    help='fixed trainset root path')
parser.add_argument('--dataset_valpath',default = r'F:\gzq\EXACT\path_dadaway',
                    help='fixed trainset root path')

parser.add_argument('--unlabel_path',default = '/media/lab312ai-3060/C14D581BDA18EBFA/Dada/airwaydata/xiehe/1114total/train',
                    help='fixed trainset root path')

# parser.add_argument('--valset_path',default = '/media/lab312ai/C14D581BDA18EBFA/Dada/airwaydata/fixedcoor/unlabel/val',
#                     help='fixed valset root path')
parser.add_argument('--testset_path',default = 'D:/Dada/Airway3D-project/fixed2021/test',
                    help='fixed testset root path')
                    
#2D  data in/out and dataset
parser.add_argument('--d_dataset_path',default = r'F:\gzq\DRAVC2022\A. Segmentation\1. Original Images\a. Training Set patch',
                    help='fixed trainset root path')
parser.add_argument('--d_unlabel_path',default = r'F:\gzq\DRAVC2022\A. Segmentation\Original_Images\val_set',
                    help='fixed trainset root path')
parser.add_argument('--d_valset_path',default = r'F:\gzq\DRAVC2022\A. Segmentation\1. Original Images\b. Valing Set patch',
                    help='fixed valset root path')
parser.add_argument('--d_testset_path',default = '/media/root/C14D581BDA18EBFA/Dada/airwaydata/2dfixed/test',
                    help='fixed testset root path')


parser.add_argument('--originaldata_path',default = r'D:\Dada\airwaydata\originaldata\total_airway_data\LIDC-EXACT(public)\EXACT',
                    help='fixed originaldata root path')
# parser.add_argument('--originaltestdata_path',default = 'D:/Dada/airwaydata/originaldata/test',
#                     help='fixed originaltestdata root path')
parser.add_argument('--originaltestdata_path',default = r'D:\Dada\airwaydata\originaldata\test\new_data_preprocess',
                    help='fixed originaltestdata root path')

parser.add_argument('--npy_path',default = './npydata/2021',
                    help='fixed npydata root path')

parser.add_argument('--save',default='2023_4_27',  ##as_at_unet_deep_1114_64     2022_5_25
                    help='save path of trained model')
parser.add_argument('--save_datapath',default='D:/Dada/Airway3D-project/fixed2021/output',
                    help='save path of predict output')
parser.add_argument('--resize_scale', type=float, default=1,
                    help='resize scale for input data')
parser.add_argument('--crop_size', type=list, default=[128, 128, 32],
                    help='patch size of train samples after resize')
####2d
parser.add_argument('--crop_size_2d', type=list, default=[512,512],
                    help='patch size of train samples after resize')
parser.add_argument('--data_size', type=list, default=[1024,1024],
                    help='patch size of train samples after resize')
parser.add_argument('--save_datapath_2d',default='D:/Dada/Airway3D-project/2dfixed_dr/output',
                    help='save path of predict output')
#####

parser.add_argument('--batch_size', type=list, default=2,
                    help='batch size of trainset')

# train
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')

parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids')

args = parser.parse_args()


