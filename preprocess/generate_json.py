import nibabel as nib
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *

from nnunet.dataset_conversion.utils import generate_dataset_json

if True:
    task_id = 501
    task_name = 'BraTS'
    # base = f'C:/nnUNet/nnUNet_raw_data/{tasks_2D[num_task]}'
    base = f'/home/sun/data/nnUNetv1/nnUNet_raw/nnUNet_raw_data/Task{task_id}_{task_name}'


    resampled_path = join(base, "labelsTr")
    nii_all = nib.load(os.path.join(resampled_path, f"{task_name}_003.nii.gz"))
    arr_all = np.array(nii_all.dataobj)
    numb = np.unique(arr_all)
    label = {}
    # numb = [0,1,2,3]
    for i in range(len(numb)):
        label[i] = int(numb[i])
    print(label)

    generate_dataset_json(output_file=join(base, 'dataset.json'),
                          imagesTr_dir=join(base, 'imagesTr'),
                          imagesTs_dir=join(base, 'imagesTs'),
                          # imagesTs_dir=None,
                          modalities=('MRI',),
                          labels=label,
                          dataset_name=task_name,
                          license='nope',
                          dataset_release='0')
    print("done!")