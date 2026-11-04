

import SimpleITK as sitk
import os

Path = r'D:/Dada/airwaydata\originaldata\test\new_data'  # DICOM存放的文件夹
data_path = os.listdir(Path)
save_path = r'D:\Dada\airwaydata\originaldata\test\new_data_preprocess'
for i in data_path:

    print(i)

    folderPath = os.path.join(Path,i)
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(folderPath)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()  # 获取到文件
    sitk.WriteImage(image, os.path.join(save_path,i + '.nii.gz'))



    # for folder in os.listdir(folderPath):
    #     imagePath = os.path.join(folderPath ,folder)  # dicom图片所在文件夹
    #     reader = sitk.ImageSeriesReader()
    #     dicom_names = reader.GetGDCMSeriesFileNames(imagePath)
    #     reader.SetFileNames(dicom_names)
    #     image = reader.Execute()  # 获取到文件
    #     sitk.WriteImage(image, folderPath + folder + '_image.nii.gz')
    # print("完成")


