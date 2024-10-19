import os
import shutil


def rename_and_copy_files(root_folder, target_folder, my_index):
    # 如果目标文件夹B不存在，则创建它
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 获取所有子文件夹
    subfolders = [f.path for f in os.scandir(root_folder) if f.is_dir()]

    # 遍历子文件夹，并从000开始编号
    for idx, subfolder in enumerate(sorted(subfolders)):
        idx+=my_index
        folder_number = f"{idx:03d}"
        print(f"Folder: {subfolder}, Number: {folder_number}")

        # 获取当前子文件夹中的所有文件
        files = [f for f in os.listdir(subfolder) if f.endswith('.nii.gz')]

        for file in files:
            # 根据文件名的结尾确定新的文件名
            if file.endswith('_t1.nii.gz'):
                new_name = f"BraTS_{folder_number}_0000.nii.gz"
            elif file.endswith('_t2.nii.gz'):
                new_name = f"BraTS_{folder_number}_0001.nii.gz"
            elif file.endswith('_t1ce.nii.gz'):
                new_name = f"BraTS_{folder_number}_0002.nii.gz"
            elif file.endswith('_flair.nii.gz'):
                new_name = f"BraTS_{folder_number}_0003.nii.gz"
            elif file.endswith('_seg.nii.gz'):
                new_name = f"BraTS_{folder_number}.nii.gz"
            else:
                continue  # 忽略不符合条件的文件

            # 获取旧文件的完整路径和新文件的完整路径
            old_file_path = os.path.join(subfolder, file)
            if new_name[-11:-8] == "000":
                new_file_path = os.path.join(target_folder, new_name)
            else:
                new_file_path = os.path.join(target_folder[-8:]+'labelsTr', new_name)

            # 复制并重命名文件到目标文件夹
            shutil.copy(old_file_path, new_file_path)
            print(f"Copied and Renamed: {old_file_path} -> {new_file_path}")


# 使用示例：假设文件夹路径为'/path/to/root_folder'，目标文件夹为'/path/to/B'
root_folder = '/home/sun/data/BraTS/MICCAI_BraTS_2018_Data_Training/LGG'
target_folder = '/home/sun/data/nnUNetv1/nnUNet_raw/nnUNet_raw_data/Task501_BraTS/imagesTr'
rename_and_copy_files(root_folder, target_folder,300)

root_folder = '/home/sun/data/BraTS/MICCAI_BraTS_2018_Data_Training/HGG'
rename_and_copy_files(root_folder, target_folder,0)


import os
import nibabel as nib
import numpy as np


def modify_nifti_files(folder_path):
    # 遍历文件夹中的所有文件
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.nii.gz'):
            file_path = os.path.join(folder_path, file_name)
            print(f"Processing file: {file_path}")

            # 加载 .nii.gz 文件
            img = nib.load(file_path)
            data = img.get_fdata()

            # 将像素值 4 修改为 3
            data[data == 4] = 3

            # 创建新的 NIfTI 图像
            new_img = nib.Nifti1Image(np.round(data).astype(int), img.affine, img.header)

            # 保存修改后的文件，覆盖原文件
            nib.save(new_img, file_path)
            print(f"Modified file saved: {file_path}")


# 使用示例：假设文件夹路径为 '/path/to/folder'
folder_path = '/home/sun/data/nnUNetv1/nnUNet_raw/nnUNet_raw_data/Task501_BraTS/labelsTr'
modify_nifti_files(folder_path)

# 使用示例：假设文件夹路径为 '/path/to/folder'
folder_path = '/home/sun/data/nnUNetv1/nnUNet_raw/nnUNet_raw_data/Task501_BraTS/labelsTs'
modify_nifti_files(folder_path)