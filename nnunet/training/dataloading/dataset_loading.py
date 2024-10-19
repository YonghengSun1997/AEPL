#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import pdb
from collections import OrderedDict
import numpy as np
from multiprocessing import Pool

from batchgenerators.dataloading.data_loader import SlimDataLoaderBase

from nnunet.configuration import default_num_threads
from nnunet.paths import preprocessing_output_dir
from batchgenerators.utilities.file_and_folder_operations import *


def get_case_identifiers(folder):
    case_identifiers = [i[:-4] for i in os.listdir(folder) if i.endswith("npz") and (i.find("segFromPrevStage") == -1)]
    return case_identifiers


def get_case_identifiers_from_raw_folder(folder):
    case_identifiers = np.unique(
        [i[:-12] for i in os.listdir(folder) if i.endswith(".nii.gz") and (i.find("segFromPrevStage") == -1)])
    return case_identifiers


def convert_to_npy(args):
    if not isinstance(args, tuple):
        key = "data"
        npz_file = args
    else:
        npz_file, key = args
    if not isfile(npz_file[:-3] + "npy"):
        a = np.load(npz_file)[key]
        np.save(npz_file[:-3] + "npy", a)


def save_as_npz(args):
    if not isinstance(args, tuple):
        key = "data"
        npy_file = args
    else:
        npy_file, key = args
    d = np.load(npy_file)
    np.savez_compressed(npy_file[:-3] + "npz", **{key: d})


def unpack_dataset(folder, threads=default_num_threads, key="data"):
    """
    unpacks all npz files in a folder to npy (whatever you want to have unpacked must be saved unter key)
    :param folder:
    :param threads:
    :param key:
    :return:
    """
    p = Pool(threads)
    npz_files = subfiles(folder, True, None, ".npz", True)
    p.map(convert_to_npy, zip(npz_files, [key] * len(npz_files)))
    p.close()
    p.join()


def pack_dataset(folder, threads=default_num_threads, key="data"):
    p = Pool(threads)
    npy_files = subfiles(folder, True, None, ".npy", True)
    p.map(save_as_npz, zip(npy_files, [key] * len(npy_files)))
    p.close()
    p.join()


def delete_npy(folder):
    case_identifiers = get_case_identifiers(folder)
    npy_files = [join(folder, i + ".npy") for i in case_identifiers]
    npy_files = [i for i in npy_files if isfile(i)]
    for n in npy_files:
        os.remove(n)


def load_dataset(folder, num_cases_properties_loading_threshold=1000):
    # we don't load the actual data but instead return the filename to the np file.
    print('loading dataset')
    case_identifiers = get_case_identifiers(folder)
    case_identifiers.sort()
    dataset = OrderedDict()
    for c in case_identifiers:
        dataset[c] = OrderedDict()
        dataset[c]['data_file'] = join(folder, "%s.npz" % c)

        # dataset[c]['properties'] = load_pickle(join(folder, "%s.pkl" % c))
        dataset[c]['properties_file'] = join(folder, "%s.pkl" % c)

        if dataset[c].get('seg_from_prev_stage_file') is not None:
            dataset[c]['seg_from_prev_stage_file'] = join(folder, "%s_segs.npz" % c)

    if len(case_identifiers) <= num_cases_properties_loading_threshold:
        print('loading all case properties')
        for i in dataset.keys():
            dataset[i]['properties'] = load_pickle(dataset[i]['properties_file'])

    return dataset


def crop_2D_image_force_fg(img, crop_size, valid_voxels):
    """
    img must be [c, x, y]
    img[-1] must be the segmentation with segmentation>0 being foreground
    :param img:
    :param crop_size:
    :param valid_voxels: voxels belonging to the selected class
    :return:
    """
    assert len(valid_voxels.shape) == 2

    if type(crop_size) not in (tuple, list):
        crop_size = [crop_size] * (len(img.shape) - 1)
    else:
        assert len(crop_size) == (len(
            img.shape) - 1), "If you provide a list/tuple as center crop make sure it has the same len as your data has dims (3d)"

    # we need to find the center coords that we can crop to without exceeding the image border
    lb_x = crop_size[0] // 2
    ub_x = img.shape[1] - crop_size[0] // 2 - crop_size[0] % 2
    lb_y = crop_size[1] // 2
    ub_y = img.shape[2] - crop_size[1] // 2 - crop_size[1] % 2

    if len(valid_voxels) == 0:
        selected_center_voxel = (np.random.random_integers(lb_x, ub_x),
                                 np.random.random_integers(lb_y, ub_y))
    else:
        selected_center_voxel = valid_voxels[np.random.choice(valid_voxels.shape[1]), :]

    selected_center_voxel = np.array(selected_center_voxel)
    for i in range(2):
        selected_center_voxel[i] = max(crop_size[i] // 2, selected_center_voxel[i])
        selected_center_voxel[i] = min(img.shape[i + 1] - crop_size[i] // 2 - crop_size[i] % 2,
                                       selected_center_voxel[i])

    result = img[:, (selected_center_voxel[0] - crop_size[0] // 2):(
            selected_center_voxel[0] + crop_size[0] // 2 + crop_size[0] % 2),
             (selected_center_voxel[1] - crop_size[1] // 2):(
                     selected_center_voxel[1] + crop_size[1] // 2 + crop_size[1] % 2)]
    return result


class DataLoader3D(SlimDataLoaderBase):
    def __init__(self, data, patch_size, final_patch_size, batch_size, has_prev_stage=False,
                 oversample_foreground_percent=0.0, memmap_mode="r", pad_mode="edge", pad_kwargs_data=None,
                 pad_sides=None):
        """
        This is the basic data loader for 3D networks. It uses preprocessed data as produced by my (Fabian) preprocessing.
        You can load the data with load_dataset(folder) where folder is the folder where the npz files are located. If there
        are only npz files present in that folder, the data loader will unpack them on the fly. This may take a while
        and increase CPU usage. Therefore, I advise you to call unpack_dataset(folder) first, which will unpack all npz
        to npy. Don't forget to call delete_npy(folder) after you are done with training?
        Why all the hassle? Well the decathlon dataset is huge. Using npy for everything will consume >1 TB and that is uncool
        given that I (Fabian) will have to store that permanently on /datasets and my local computer. With this strategy all
        data is stored in a compressed format (factor 10 smaller) and only unpacked when needed.
        :param data: get this with load_dataset(folder, stage=0). Plug the return value in here and you are g2g (good to go)
        :param patch_size: what patch size will this data loader return? it is common practice to first load larger
        patches so that a central crop after data augmentation can be done to reduce border artifacts. If unsure, use
        get_patch_size() from data_augmentation.default_data_augmentation
        :param final_patch_size: what will the patch finally be cropped to (after data augmentation)? this is the patch
        size that goes into your network. We need this here because we will pad patients in here so that patches at the
        border of patients are sampled properly
        :param batch_size:
        :param num_batches: how many batches will the data loader produce before stopping? None=endless
        :param seed:
        :param stage: ignore this (Fabian only)
        :param random: Sample keys randomly; CAREFUL! non-random sampling requires batch_size=1, otherwise you will iterate batch_size times over the dataset
        :param oversample_foreground: half the batch will be forced to contain at least some foreground (equal prob for each of the foreground classes)
        """
        super(DataLoader3D, self).__init__(data, batch_size, None)
        if pad_kwargs_data is None:
            pad_kwargs_data = OrderedDict()
        self.pad_kwargs_data = pad_kwargs_data
        self.pad_mode = pad_mode
        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.has_prev_stage = has_prev_stage
        self.patch_size = patch_size
        self.list_of_keys = list(self._data.keys())
        # need_to_pad denotes by how much we need to pad the data so that if we sample a patch of size final_patch_size
        # (which is what the network will get) these patches will also cover the border of the patients
        self.need_to_pad = (np.array(patch_size) - np.array(final_patch_size)).astype(int)
        if pad_sides is not None:
            if not isinstance(pad_sides, np.ndarray):
                pad_sides = np.array(pad_sides)
            self.need_to_pad += pad_sides
        self.memmap_mode = memmap_mode
        self.num_channels = None
        self.pad_sides = pad_sides
        self.data_shape, self.seg_shape = self.determine_shapes()

    def get_do_oversample(self, batch_idx):
        return not batch_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))

    def determine_shapes(self):
        if self.has_prev_stage:
            num_seg = 2
        else:
            num_seg = 1

        k = list(self._data.keys())[0]
        if isfile(self._data[k]['data_file'][:-4] + ".npy"):
            case_all_data = np.load(self._data[k]['data_file'][:-4] + ".npy", self.memmap_mode)
        else:
            case_all_data = np.load(self._data[k]['data_file'])['data']
        num_color_channels = case_all_data.shape[0] - 1
        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, num_seg, *self.patch_size)
        return data_shape, seg_shape

    def generate_train_batch(self):
        # pdb.set_trace()
        selected_keys = np.random.choice(self.list_of_keys, self.batch_size, True, None)
        # print(selected_keys)
        data = np.zeros(self.data_shape, dtype=np.float32)
        seg = np.zeros(self.seg_shape, dtype=np.float32)
        case_properties = []
        for j, i in enumerate(selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            if self.get_do_oversample(j):
                force_fg = True
            else:
                force_fg = False

            if 'properties' in self._data[i].keys():
                properties = self._data[i]['properties']
            else:
                properties = load_pickle(self._data[i]['properties_file'])
            case_properties.append(properties)

            # cases are stored as npz, but we require unpack_dataset to be run. This will decompress them into npy
            # which is much faster to access
            if isfile(self._data[i]['data_file'][:-4] + ".npy"):
                case_all_data = np.load(self._data[i]['data_file'][:-4] + ".npy", self.memmap_mode)
            else:
                case_all_data = np.load(self._data[i]['data_file'])['data']

            # If we are doing the cascade then we will also need to load the segmentation of the previous stage and
            # concatenate it. Here it will be concatenates to the segmentation because the augmentations need to be
            # applied to it in segmentation mode. Later in the data augmentation we move it from the segmentations to
            # the last channel of the data
            if self.has_prev_stage:
                if isfile(self._data[i]['seg_from_prev_stage_file'][:-4] + ".npy"):
                    segs_from_previous_stage = np.load(self._data[i]['seg_from_prev_stage_file'][:-4] + ".npy",
                                                       mmap_mode=self.memmap_mode)[None]
                else:
                    segs_from_previous_stage = np.load(self._data[i]['seg_from_prev_stage_file'])['data'][None]
                # we theoretically support several possible previsous segmentations from which only one is sampled. But
                # in practice this feature was never used so it's always only one segmentation
                seg_key = np.random.choice(segs_from_previous_stage.shape[0])
                seg_from_previous_stage = segs_from_previous_stage[seg_key:seg_key + 1]
                assert all([i == j for i, j in zip(seg_from_previous_stage.shape[1:], case_all_data.shape[1:])]), \
                    "seg_from_previous_stage does not match the shape of case_all_data: %s vs %s" % \
                    (str(seg_from_previous_stage.shape[1:]), str(case_all_data.shape[1:]))
            else:
                seg_from_previous_stage = None

            # do you trust me? You better do. Otherwise you'll have to go through this mess and honestly there are
            # better things you could do right now

            # (above) documentation of the day. Nice. Even myself coming back 1 months later I have not friggin idea
            # what's going on. I keep the above documentation just for fun but attempt to make things clearer now

            need_to_pad = self.need_to_pad.copy()
            for d in range(3):
                # if case_all_data.shape + need_to_pad is still < patch size we need to pad more! We pad on both sides
                # always
                if need_to_pad[d] + case_all_data.shape[d + 1] < self.patch_size[d]:
                    need_to_pad[d] = self.patch_size[d] - case_all_data.shape[d + 1]

            # we can now choose the bbox from -need_to_pad // 2 to shape - patch_size + need_to_pad // 2. Here we
            # define what the upper and lower bound can be to then sample from them with np.random.randint
            shape = case_all_data.shape[1:]
            lb_x = - need_to_pad[0] // 2
            ub_x = shape[0] + need_to_pad[0] // 2 + need_to_pad[0] % 2 - self.patch_size[0]
            lb_y = - need_to_pad[1] // 2
            ub_y = shape[1] + need_to_pad[1] // 2 + need_to_pad[1] % 2 - self.patch_size[1]
            lb_z = - need_to_pad[2] // 2
            ub_z = shape[2] + need_to_pad[2] // 2 + need_to_pad[2] % 2 - self.patch_size[2]

            # if not force_fg then we can just sample the bbox randomly from lb and ub. Else we need to make sure we get
            # at least one of the foreground classes in the patch
            if not force_fg:
                bbox_x_lb = np.random.randint(lb_x, ub_x + 1)
                bbox_y_lb = np.random.randint(lb_y, ub_y + 1)
                bbox_z_lb = np.random.randint(lb_z, ub_z + 1)
            else:
                # these values should have been precomputed
                if 'class_locations' not in properties.keys():
                    raise RuntimeError("Please rerun the preprocessing with the newest version of nnU-Net!")

                # this saves us a np.unique. Preprocessing already did that for all cases. Neat.
                foreground_classes = np.array(
                    [i for i in properties['class_locations'].keys() if len(properties['class_locations'][i]) != 0])
                foreground_classes = foreground_classes[foreground_classes > 0]

                if len(foreground_classes) == 0:
                    # this only happens if some image does not contain foreground voxels at all
                    selected_class = None
                    voxels_of_that_class = None
                    print('case does not contain any foreground classes', i)
                else:
                    selected_class = np.random.choice(foreground_classes)

                    voxels_of_that_class = properties['class_locations'][selected_class]

                if voxels_of_that_class is not None:
                    selected_voxel = voxels_of_that_class[np.random.choice(len(voxels_of_that_class))]
                    # selected voxel is center voxel. Subtract half the patch size to get lower bbox voxel.
                    # Make sure it is within the bounds of lb and ub
                    bbox_x_lb = max(lb_x, selected_voxel[0] - self.patch_size[0] // 2)
                    bbox_y_lb = max(lb_y, selected_voxel[1] - self.patch_size[1] // 2)
                    bbox_z_lb = max(lb_z, selected_voxel[2] - self.patch_size[2] // 2)
                else:
                    # If the image does not contain any foreground classes, we fall back to random cropping
                    bbox_x_lb = np.random.randint(lb_x, ub_x + 1)
                    bbox_y_lb = np.random.randint(lb_y, ub_y + 1)
                    bbox_z_lb = np.random.randint(lb_z, ub_z + 1)

            bbox_x_ub = bbox_x_lb + self.patch_size[0]
            bbox_y_ub = bbox_y_lb + self.patch_size[1]
            bbox_z_ub = bbox_z_lb + self.patch_size[2]

            # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
            # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
            # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
            # later
            valid_bbox_x_lb = max(0, bbox_x_lb)
            valid_bbox_x_ub = min(shape[0], bbox_x_ub)
            valid_bbox_y_lb = max(0, bbox_y_lb)
            valid_bbox_y_ub = min(shape[1], bbox_y_ub)
            valid_bbox_z_lb = max(0, bbox_z_lb)
            valid_bbox_z_ub = min(shape[2], bbox_z_ub)

            # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
            # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
            # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
            # remove label -1 in the data augmentation but this way it is less error prone)
            case_all_data = np.copy(case_all_data[:, valid_bbox_x_lb:valid_bbox_x_ub,
                                    valid_bbox_y_lb:valid_bbox_y_ub,
                                    valid_bbox_z_lb:valid_bbox_z_ub])
            if seg_from_previous_stage is not None:
                seg_from_previous_stage = seg_from_previous_stage[:, valid_bbox_x_lb:valid_bbox_x_ub,
                                          valid_bbox_y_lb:valid_bbox_y_ub,
                                          valid_bbox_z_lb:valid_bbox_z_ub]

            data[j] = np.pad(case_all_data[:-1], ((0, 0),
                                                  (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                                  (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0)),
                                                  (-min(0, bbox_z_lb), max(bbox_z_ub - shape[2], 0))),
                             self.pad_mode, **self.pad_kwargs_data)

            seg[j, 0] = np.pad(case_all_data[-1:], ((0, 0),
                                                    (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                                    (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0)),
                                                    (-min(0, bbox_z_lb), max(bbox_z_ub - shape[2], 0))),
                               'constant', **{'constant_values': -1})
            if seg_from_previous_stage is not None:
                seg[j, 1] = np.pad(seg_from_previous_stage, ((0, 0),
                                                             (-min(0, bbox_x_lb),
                                                              max(bbox_x_ub - shape[0], 0)),
                                                             (-min(0, bbox_y_lb),
                                                              max(bbox_y_ub - shape[1], 0)),
                                                             (-min(0, bbox_z_lb),
                                                              max(bbox_z_ub - shape[2], 0))),
                                   'constant', **{'constant_values': 0})

        return {'data': data, 'seg': seg, 'properties': case_properties, 'keys': selected_keys}

class DataLoader3D_AEPL(SlimDataLoaderBase):
    def __init__(self, data, patch_size, final_patch_size, batch_size, has_prev_stage=False,
                 oversample_foreground_percent=0.0, memmap_mode="r", pad_mode="edge", pad_kwargs_data=None,
                 pad_sides=None):
        """
        This is the basic data loader for 3D networks. It uses preprocessed data as produced by my (Fabian) preprocessing.
        You can load the data with load_dataset(folder) where folder is the folder where the npz files are located. If there
        are only npz files present in that folder, the data loader will unpack them on the fly. This may take a while
        and increase CPU usage. Therefore, I advise you to call unpack_dataset(folder) first, which will unpack all npz
        to npy. Don't forget to call delete_npy(folder) after you are done with training?
        Why all the hassle? Well the decathlon dataset is huge. Using npy for everything will consume >1 TB and that is uncool
        given that I (Fabian) will have to store that permanently on /datasets and my local computer. With this strategy all
        data is stored in a compressed format (factor 10 smaller) and only unpacked when needed.
        :param data: get this with load_dataset(folder, stage=0). Plug the return value in here and you are g2g (good to go)
        :param patch_size: what patch size will this data loader return? it is common practice to first load larger
        patches so that a central crop after data augmentation can be done to reduce border artifacts. If unsure, use
        get_patch_size() from data_augmentation.default_data_augmentation
        :param final_patch_size: what will the patch finally be cropped to (after data augmentation)? this is the patch
        size that goes into your network. We need this here because we will pad patients in here so that patches at the
        border of patients are sampled properly
        :param batch_size:
        :param num_batches: how many batches will the data loader produce before stopping? None=endless
        :param seed:
        :param stage: ignore this (Fabian only)
        :param random: Sample keys randomly; CAREFUL! non-random sampling requires batch_size=1, otherwise you will iterate batch_size times over the dataset
        :param oversample_foreground: half the batch will be forced to contain at least some foreground (equal prob for each of the foreground classes)
        """
        super(DataLoader3D_AEPL, self).__init__(data, batch_size, None)
        if pad_kwargs_data is None:
            pad_kwargs_data = OrderedDict()
        self.pad_kwargs_data = pad_kwargs_data
        self.pad_mode = pad_mode
        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.has_prev_stage = has_prev_stage
        self.patch_size = patch_size
        self.list_of_keys = list(self._data.keys())
        # need_to_pad denotes by how much we need to pad the data so that if we sample a patch of size final_patch_size
        # (which is what the network will get) these patches will also cover the border of the patients
        self.need_to_pad = (np.array(patch_size) - np.array(final_patch_size)).astype(int)
        if pad_sides is not None:
            if not isinstance(pad_sides, np.ndarray):
                pad_sides = np.array(pad_sides)
            self.need_to_pad += pad_sides
        self.memmap_mode = memmap_mode
        self.num_channels = None
        self.pad_sides = pad_sides
        self.data_shape, self.seg_shape = self.determine_shapes()
        self.birads = {
    "BraTS_000": 0,
    "BraTS_001": 0,
    "BraTS_002": 0,
    "BraTS_003": 0,
    "BraTS_004": 0,
    "BraTS_005": 0,
    "BraTS_006": 0,
    "BraTS_007": 0,
    "BraTS_008": 0,
    "BraTS_009": 0,
    "BraTS_010": 0,
    "BraTS_011": 0,
    "BraTS_012": 0,
    "BraTS_013": 0,
    "BraTS_014": 0,
    "BraTS_015": 0,
    "BraTS_016": 0,
    "BraTS_017": 0,
    "BraTS_018": 0,
    "BraTS_019": 0,
    "BraTS_020": 0,
    "BraTS_021": 0,
    "BraTS_022": 0,
    "BraTS_023": 0,
    "BraTS_024": 0,
    "BraTS_025": 0,
    "BraTS_026": 0,
    "BraTS_027": 0,
    "BraTS_028": 0,
    "BraTS_029": 0,
    "BraTS_030": 0,
    "BraTS_031": 0,
    "BraTS_032": 0,
    "BraTS_033": 0,
    "BraTS_034": 0,
    "BraTS_035": 0,
    "BraTS_036": 0,
    "BraTS_037": 0,
    "BraTS_038": 0,
    "BraTS_039": 0,
    "BraTS_040": 0,
    "BraTS_041": 0,
    "BraTS_042": 0,
    "BraTS_043": 0,
    "BraTS_044": 0,
    "BraTS_045": 0,
    "BraTS_046": 0,
    "BraTS_047": 0,
    "BraTS_048": 0,
    "BraTS_049": 0,
    "BraTS_050": 0,
    "BraTS_051": 0,
    "BraTS_052": 0,
    "BraTS_053": 0,
    "BraTS_054": 0,
    "BraTS_055": 0,
    "BraTS_056": 0,
    "BraTS_057": 0,
    "BraTS_058": 0,
    "BraTS_059": 0,
    "BraTS_060": 0,
    "BraTS_061": 0,
    "BraTS_062": 0,
    "BraTS_063": 0,
    "BraTS_064": 0,
    "BraTS_065": 0,
    "BraTS_066": 0,
    "BraTS_067": 0,
    "BraTS_068": 0,
    "BraTS_069": 0,
    "BraTS_070": 0,
    "BraTS_071": 0,
    "BraTS_072": 0,
    "BraTS_073": 0,
    "BraTS_074": 0,
    "BraTS_075": 0,
    "BraTS_076": 0,
    "BraTS_077": 0,
    "BraTS_078": 0,
    "BraTS_079": 0,
    "BraTS_080": 0,
    "BraTS_081": 0,
    "BraTS_082": 0,
    "BraTS_083": 0,
    "BraTS_084": 0,
    "BraTS_085": 0,
    "BraTS_086": 0,
    "BraTS_087": 0,
    "BraTS_088": 0,
    "BraTS_089": 0,
    "BraTS_090": 0,
    "BraTS_091": 0,
    "BraTS_092": 0,
    "BraTS_093": 0,
    "BraTS_094": 0,
    "BraTS_095": 0,
    "BraTS_096": 0,
    "BraTS_097": 0,
    "BraTS_098": 0,
    "BraTS_099": 0,
    "BraTS_100": 0,
    "BraTS_101": 0,
    "BraTS_102": 0,
    "BraTS_103": 0,
    "BraTS_104": 0,
    "BraTS_105": 0,
    "BraTS_106": 0,
    "BraTS_107": 0,
    "BraTS_108": 0,
    "BraTS_109": 0,
    "BraTS_110": 0,
    "BraTS_111": 0,
    "BraTS_112": 0,
    "BraTS_113": 0,
    "BraTS_114": 0,
    "BraTS_115": 0,
    "BraTS_116": 0,
    "BraTS_117": 0,
    "BraTS_118": 0,
    "BraTS_119": 0,
    "BraTS_120": 0,
    "BraTS_121": 0,
    "BraTS_122": 0,
    "BraTS_123": 0,
    "BraTS_124": 0,
    "BraTS_125": 0,
    "BraTS_126": 0,
    "BraTS_127": 0,
    "BraTS_128": 0,
    "BraTS_129": 0,
    "BraTS_130": 0,
    "BraTS_131": 0,
    "BraTS_132": 0,
    "BraTS_133": 0,
    "BraTS_134": 0,
    "BraTS_135": 0,
    "BraTS_136": 0,
    "BraTS_137": 0,
    "BraTS_138": 0,
    "BraTS_139": 0,
    "BraTS_140": 0,
    "BraTS_141": 0,
    "BraTS_142": 0,
    "BraTS_143": 0,
    "BraTS_144": 0,
    "BraTS_145": 0,
    "BraTS_146": 0,
    "BraTS_147": 0,
    "BraTS_148": 0,
    "BraTS_149": 0,
    "BraTS_150": 0,
    "BraTS_151": 0,
    "BraTS_152": 0,
    "BraTS_153": 0,
    "BraTS_154": 0,
    "BraTS_155": 0,
    "BraTS_156": 0,
    "BraTS_157": 0,
    "BraTS_158": 0,
    "BraTS_159": 0,
    "BraTS_160": 0,
    "BraTS_161": 0,
    "BraTS_162": 0,
    "BraTS_163": 0,
    "BraTS_164": 0,
    "BraTS_165": 0,
    "BraTS_166": 0,
    "BraTS_167": 0,
    "BraTS_168": 0,
    "BraTS_169": 0,
    "BraTS_170": 0,
    "BraTS_171": 0,
    "BraTS_172": 0,
    "BraTS_173": 0,
    "BraTS_174": 0,
    "BraTS_175": 0,
    "BraTS_176": 0,
    "BraTS_177": 0,
    "BraTS_178": 0,
    "BraTS_179": 0,
    "BraTS_180": 0,
    "BraTS_181": 0,
    "BraTS_182": 0,
    "BraTS_183": 0,
    "BraTS_184": 0,
    "BraTS_185": 0,
    "BraTS_186": 0,
    "BraTS_187": 0,
    "BraTS_188": 0,
    "BraTS_189": 0,
    "BraTS_190": 0,
    "BraTS_191": 0,
    "BraTS_192": 0,
    "BraTS_193": 0,
    "BraTS_194": 0,
    "BraTS_195": 0,
    "BraTS_196": 0,
    "BraTS_197": 0,
    "BraTS_198": 0,
    "BraTS_199": 0,
    "BraTS_200": 0,
    "BraTS_201": 0,
    "BraTS_202": 0,
    "BraTS_203": 0,
    "BraTS_204": 0,
    "BraTS_205": 0,
    "BraTS_206": 0,
    "BraTS_207": 0,
    "BraTS_208": 0,
    "BraTS_209": 0,
    "BraTS_300": 1,
    "BraTS_301": 1,
    "BraTS_302": 1,
    "BraTS_303": 1,
    "BraTS_304": 1,
    "BraTS_305": 1,
    "BraTS_306": 1,
    "BraTS_307": 1,
    "BraTS_308": 1,
    "BraTS_309": 1,
    "BraTS_310": 1,
    "BraTS_311": 1,
    "BraTS_312": 1,
    "BraTS_313": 1,
    "BraTS_314": 1,
    "BraTS_315": 1,
    "BraTS_316": 1,
    "BraTS_317": 1,
    "BraTS_318": 1,
    "BraTS_319": 1,
    "BraTS_320": 1,
    "BraTS_321": 1,
    "BraTS_322": 1,
    "BraTS_323": 1,
    "BraTS_324": 1,
    "BraTS_325": 1,
    "BraTS_326": 1,
    "BraTS_327": 1,
    "BraTS_328": 1,
    "BraTS_329": 1,
    "BraTS_330": 1,
    "BraTS_331": 1,
    "BraTS_332": 1,
    "BraTS_333": 1,
    "BraTS_334": 1,
    "BraTS_335": 1,
    "BraTS_336": 1,
    "BraTS_337": 1,
    "BraTS_338": 1,
    "BraTS_339": 1,
    "BraTS_340": 1,
    "BraTS_341": 1,
    "BraTS_342": 1,
    "BraTS_343": 1,
    "BraTS_344": 1,
    "BraTS_345": 1,
    "BraTS_346": 1,
    "BraTS_347": 1,
    "BraTS_348": 1,
    "BraTS_349": 1,
    "BraTS_350": 1,
    "BraTS_351": 1,
    "BraTS_352": 1,
    "BraTS_353": 1,
    "BraTS_354": 1,
    "BraTS_355": 1,
    "BraTS_356": 1,
    "BraTS_357": 1,
    "BraTS_358": 1,
    "BraTS_359": 1,
    "BraTS_360": 1,
    "BraTS_361": 1,
    "BraTS_362": 1,
    "BraTS_363": 1,
    "BraTS_364": 1,
    "BraTS_365": 1,
    "BraTS_366": 1,
    "BraTS_367": 1,
    "BraTS_368": 1,
    "BraTS_369": 1,
    "BraTS_370": 1,
    "BraTS_371": 1,
    "BraTS_372": 1,
    "BraTS_373": 1,
    "BraTS_374": 1
}
            # {'AEPL_003': 6, 'AEPL_004': 2, 'AEPL_005': 4, 'AEPL_007': 6, 'AEPL_009': 0, 'AEPL_010': 4, 'AEPL_022': 6, 'AEPL_025': 2, 'AEPL_028': 6, 'AEPL_032': 6, 'AEPL_033': 3, 'AEPL_035': 2, 'AEPL_036': 4, 'AEPL_038': 6, 'AEPL_042': 2, 'AEPL_043': 3, 'AEPL_044': 4, 'AEPL_045': 3, 'AEPL_047': 6, 'AEPL_049': 4, 'AEPL_050': 6, 'AEPL_496': 6, 'AEPL_541': 6, 'AEPL_559': 2, 'AEPL_562': 4, 'AEPL_567': 6, 'AEPL_568': 4, 'AEPL_573': 2, 'AEPL_577': 6, 'AEPL_580': 6, 'AEPL_581': 6, 'AEPL_582': 6, 'AEPL_584': 6, 'AEPL_586': 6, 'AEPL_588': 2, 'AEPL_591': 6, 'AEPL_593': 6, 'AEPL_596': 6, 'AEPL_597': 6, 'AEPL_598': 6, 'AEPL_604': 6, 'AEPL_605': 6, 'AEPL_607': 6, 'AEPL_612': 6, 'AEPL_613': 6, 'AEPL_615': 4, 'AEPL_618': 6, 'AEPL_619': 6, 'AEPL_622': 6, 'AEPL_623': 2, 'AEPL_628': 6, 'AEPL_632': 6}
    def get_do_oversample(self, batch_idx):
        return not batch_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))

    def determine_shapes(self):
        if self.has_prev_stage:
            num_seg = 2
        else:
            num_seg = 1

        k = list(self._data.keys())[0]
        if isfile(self._data[k]['data_file'][:-4] + ".npy"):
            case_all_data = np.load(self._data[k]['data_file'][:-4] + ".npy", self.memmap_mode)
        else:
            case_all_data = np.load(self._data[k]['data_file'])['data']
        num_color_channels = case_all_data.shape[0] - 1
        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, num_seg, *self.patch_size)
        return data_shape, seg_shape

    def generate_train_batch(self):
        selected_keys = np.random.choice(self.list_of_keys, self.batch_size, True, None)
        data = np.zeros(self.data_shape, dtype=np.float32)
        seg = np.zeros(self.seg_shape, dtype=np.float32)
        case_properties = []
        birads_gt = []
        for j, i in enumerate(selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            birads_gt.append(self.birads[i])

            if self.get_do_oversample(j):
                force_fg = True
            else:
                force_fg = False

            if 'properties' in self._data[i].keys():
                properties = self._data[i]['properties']
            else:
                properties = load_pickle(self._data[i]['properties_file'])
            case_properties.append(properties)

            # cases are stored as npz, but we require unpack_dataset to be run. This will decompress them into npy
            # which is much faster to access
            if isfile(self._data[i]['data_file'][:-4] + ".npy"):
                case_all_data = np.load(self._data[i]['data_file'][:-4] + ".npy", self.memmap_mode)
            else:
                case_all_data = np.load(self._data[i]['data_file'])['data']

            # If we are doing the cascade then we will also need to load the segmentation of the previous stage and
            # concatenate it. Here it will be concatenates to the segmentation because the augmentations need to be
            # applied to it in segmentation mode. Later in the data augmentation we move it from the segmentations to
            # the last channel of the data
            if self.has_prev_stage:
                if isfile(self._data[i]['seg_from_prev_stage_file'][:-4] + ".npy"):
                    segs_from_previous_stage = np.load(self._data[i]['seg_from_prev_stage_file'][:-4] + ".npy",
                                                       mmap_mode=self.memmap_mode)[None]
                else:
                    segs_from_previous_stage = np.load(self._data[i]['seg_from_prev_stage_file'])['data'][None]
                # we theoretically support several possible previsous segmentations from which only one is sampled. But
                # in practice this feature was never used so it's always only one segmentation
                seg_key = np.random.choice(segs_from_previous_stage.shape[0])
                seg_from_previous_stage = segs_from_previous_stage[seg_key:seg_key + 1]
                assert all([i == j for i, j in zip(seg_from_previous_stage.shape[1:], case_all_data.shape[1:])]), \
                    "seg_from_previous_stage does not match the shape of case_all_data: %s vs %s" % \
                    (str(seg_from_previous_stage.shape[1:]), str(case_all_data.shape[1:]))
            else:
                seg_from_previous_stage = None

            # do you trust me? You better do. Otherwise you'll have to go through this mess and honestly there are
            # better things you could do right now

            # (above) documentation of the day. Nice. Even myself coming back 1 months later I have not friggin idea
            # what's going on. I keep the above documentation just for fun but attempt to make things clearer now

            need_to_pad = self.need_to_pad.copy()
            for d in range(3):
                # if case_all_data.shape + need_to_pad is still < patch size we need to pad more! We pad on both sides
                # always
                if need_to_pad[d] + case_all_data.shape[d + 1] < self.patch_size[d]:
                    need_to_pad[d] = self.patch_size[d] - case_all_data.shape[d + 1]

            # we can now choose the bbox from -need_to_pad // 2 to shape - patch_size + need_to_pad // 2. Here we
            # define what the upper and lower bound can be to then sample from them with np.random.randint
            shape = case_all_data.shape[1:]
            lb_x = - need_to_pad[0] // 2
            ub_x = shape[0] + need_to_pad[0] // 2 + need_to_pad[0] % 2 - self.patch_size[0]
            lb_y = - need_to_pad[1] // 2
            ub_y = shape[1] + need_to_pad[1] // 2 + need_to_pad[1] % 2 - self.patch_size[1]
            lb_z = - need_to_pad[2] // 2
            ub_z = shape[2] + need_to_pad[2] // 2 + need_to_pad[2] % 2 - self.patch_size[2]

            # if not force_fg then we can just sample the bbox randomly from lb and ub. Else we need to make sure we get
            # at least one of the foreground classes in the patch
            if not force_fg:
                bbox_x_lb = np.random.randint(lb_x, ub_x + 1)
                bbox_y_lb = np.random.randint(lb_y, ub_y + 1)
                bbox_z_lb = np.random.randint(lb_z, ub_z + 1)
            else:
                # these values should have been precomputed
                if 'class_locations' not in properties.keys():
                    raise RuntimeError("Please rerun the preprocessing with the newest version of nnU-Net!")

                # this saves us a np.unique. Preprocessing already did that for all cases. Neat.
                foreground_classes = np.array(
                    [i for i in properties['class_locations'].keys() if len(properties['class_locations'][i]) != 0])
                foreground_classes = foreground_classes[foreground_classes > 0]

                if len(foreground_classes) == 0:
                    # this only happens if some image does not contain foreground voxels at all
                    selected_class = None
                    voxels_of_that_class = None
                    print('case does not contain any foreground classes', i)
                else:
                    selected_class = np.random.choice(foreground_classes)

                    voxels_of_that_class = properties['class_locations'][selected_class]

                if voxels_of_that_class is not None:
                    selected_voxel = voxels_of_that_class[np.random.choice(len(voxels_of_that_class))]
                    # selected voxel is center voxel. Subtract half the patch size to get lower bbox voxel.
                    # Make sure it is within the bounds of lb and ub
                    bbox_x_lb = max(lb_x, selected_voxel[0] - self.patch_size[0] // 2)
                    bbox_y_lb = max(lb_y, selected_voxel[1] - self.patch_size[1] // 2)
                    bbox_z_lb = max(lb_z, selected_voxel[2] - self.patch_size[2] // 2)
                else:
                    # If the image does not contain any foreground classes, we fall back to random cropping
                    bbox_x_lb = np.random.randint(lb_x, ub_x + 1)
                    bbox_y_lb = np.random.randint(lb_y, ub_y + 1)
                    bbox_z_lb = np.random.randint(lb_z, ub_z + 1)

            bbox_x_ub = bbox_x_lb + self.patch_size[0]
            bbox_y_ub = bbox_y_lb + self.patch_size[1]
            bbox_z_ub = bbox_z_lb + self.patch_size[2]

            # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
            # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
            # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
            # later
            valid_bbox_x_lb = max(0, bbox_x_lb)
            valid_bbox_x_ub = min(shape[0], bbox_x_ub)
            valid_bbox_y_lb = max(0, bbox_y_lb)
            valid_bbox_y_ub = min(shape[1], bbox_y_ub)
            valid_bbox_z_lb = max(0, bbox_z_lb)
            valid_bbox_z_ub = min(shape[2], bbox_z_ub)

            # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
            # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
            # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
            # remove label -1 in the data augmentation but this way it is less error prone)
            case_all_data = np.copy(case_all_data[:, valid_bbox_x_lb:valid_bbox_x_ub,
                                    valid_bbox_y_lb:valid_bbox_y_ub,
                                    valid_bbox_z_lb:valid_bbox_z_ub])
            if seg_from_previous_stage is not None:
                seg_from_previous_stage = seg_from_previous_stage[:, valid_bbox_x_lb:valid_bbox_x_ub,
                                          valid_bbox_y_lb:valid_bbox_y_ub,
                                          valid_bbox_z_lb:valid_bbox_z_ub]

            data[j] = np.pad(case_all_data[:-1], ((0, 0),
                                                  (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                                  (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0)),
                                                  (-min(0, bbox_z_lb), max(bbox_z_ub - shape[2], 0))),
                             self.pad_mode, **self.pad_kwargs_data)

            seg[j, 0] = np.pad(case_all_data[-1:], ((0, 0),
                                                    (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                                    (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0)),
                                                    (-min(0, bbox_z_lb), max(bbox_z_ub - shape[2], 0))),
                               'constant', **{'constant_values': -1})
            if seg_from_previous_stage is not None:
                seg[j, 1] = np.pad(seg_from_previous_stage, ((0, 0),
                                                             (-min(0, bbox_x_lb),
                                                              max(bbox_x_ub - shape[0], 0)),
                                                             (-min(0, bbox_y_lb),
                                                              max(bbox_y_ub - shape[1], 0)),
                                                             (-min(0, bbox_z_lb),
                                                              max(bbox_z_ub - shape[2], 0))),
                                   'constant', **{'constant_values': 0})

        return {'data': data, 'seg': seg, 'properties': case_properties, 'keys': selected_keys, 'birads_gt':birads_gt}

class DataLoader3D_BraTS(SlimDataLoaderBase):
    def __init__(self, data, patch_size, final_patch_size, batch_size, has_prev_stage=False,
                 oversample_foreground_percent=0.0, memmap_mode="r", pad_mode="edge", pad_kwargs_data=None,
                 pad_sides=None):
        """
        This is the basic data loader for 3D networks. It uses preprocessed data as produced by my (Fabian) preprocessing.
        You can load the data with load_dataset(folder) where folder is the folder where the npz files are located. If there
        are only npz files present in that folder, the data loader will unpack them on the fly. This may take a while
        and increase CPU usage. Therefore, I advise you to call unpack_dataset(folder) first, which will unpack all npz
        to npy. Don't forget to call delete_npy(folder) after you are done with training?
        Why all the hassle? Well the decathlon dataset is huge. Using npy for everything will consume >1 TB and that is uncool
        given that I (Fabian) will have to store that permanently on /datasets and my local computer. With this strategy all
        data is stored in a compressed format (factor 10 smaller) and only unpacked when needed.
        :param data: get this with load_dataset(folder, stage=0). Plug the return value in here and you are g2g (good to go)
        :param patch_size: what patch size will this data loader return? it is common practice to first load larger
        patches so that a central crop after data augmentation can be done to reduce border artifacts. If unsure, use
        get_patch_size() from data_augmentation.default_data_augmentation
        :param final_patch_size: what will the patch finally be cropped to (after data augmentation)? this is the patch
        size that goes into your network. We need this here because we will pad patients in here so that patches at the
        border of patients are sampled properly
        :param batch_size:
        :param num_batches: how many batches will the data loader produce before stopping? None=endless
        :param seed:
        :param stage: ignore this (Fabian only)
        :param random: Sample keys randomly; CAREFUL! non-random sampling requires batch_size=1, otherwise you will iterate batch_size times over the dataset
        :param oversample_foreground: half the batch will be forced to contain at least some foreground (equal prob for each of the foreground classes)
        """
        super(DataLoader3D_BraTS, self).__init__(data, batch_size, None)
        if pad_kwargs_data is None:
            pad_kwargs_data = OrderedDict()
        self.pad_kwargs_data = pad_kwargs_data
        self.pad_mode = pad_mode
        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.has_prev_stage = has_prev_stage
        self.patch_size = patch_size
        self.list_of_keys = list(self._data.keys())
        # need_to_pad denotes by how much we need to pad the data so that if we sample a patch of size final_patch_size
        # (which is what the network will get) these patches will also cover the border of the patients
        self.need_to_pad = (np.array(patch_size) - np.array(final_patch_size)).astype(int)
        if pad_sides is not None:
            if not isinstance(pad_sides, np.ndarray):
                pad_sides = np.array(pad_sides)
            self.need_to_pad += pad_sides
        self.memmap_mode = memmap_mode
        self.num_channels = None
        self.pad_sides = pad_sides
        self.data_shape, self.seg_shape = self.determine_shapes()
        self.birads = {'HGG__Brats18_CBICA_AQD_1': 0, 'HGG__Brats18_CBICA_AQR_1': 0, 'LGG__Brats18_TCIA10_420_1': 1, 'LGG__Brats18_TCIA10_325_1': 1, 'LGG__Brats18_TCIA13_650_1': 1, 'HGG__Brats18_TCIA06_409_1': 0, 'HGG__Brats18_2013_22_1': 0, 'HGG__Brats18_TCIA04_328_1': 0, 'LGG__Brats18_TCIA10_330_1': 1, 'HGG__Brats18_CBICA_BFP_1': 0, 'LGG__Brats18_TCIA10_266_1': 1, 'LGG__Brats18_TCIA13_624_1': 1, 'HGG__Brats18_CBICA_AWI_1': 0, 'HGG__Brats18_TCIA03_474_1': 0, 'HGG__Brats18_CBICA_AXJ_1': 0, 'HGG__Brats18_TCIA02_171_1': 0, 'HGG__Brats18_CBICA_AME_1': 0, 'LGG__Brats18_TCIA10_307_1': 1, 'HGG__Brats18_TCIA04_343_1': 0, 'HGG__Brats18_CBICA_ANP_1': 0, 'HGG__Brats18_CBICA_AXO_1': 0, 'HGG__Brats18_TCIA08_113_1': 0, 'LGG__Brats18_2013_0_1': 1, 'LGG__Brats18_TCIA13_654_1': 1, 'HGG__Brats18_CBICA_BFB_1': 0, 'LGG__Brats18_TCIA10_490_1': 1, 'LGG__Brats18_TCIA12_249_1': 1, 'HGG__Brats18_CBICA_AWG_1': 0, 'LGG__Brats18_TCIA10_640_1': 1, 'LGG__Brats18_TCIA09_428_1': 1, 'HGG__Brats18_TCIA02_314_1': 0, 'HGG__Brats18_TCIA02_226_1': 0, 'HGG__Brats18_TCIA01_429_1': 0, 'HGG__Brats18_CBICA_AQU_1': 0, 'HGG__Brats18_2013_3_1': 0, 'LGG__Brats18_TCIA10_625_1': 1, 'LGG__Brats18_2013_29_1': 1, 'LGG__Brats18_TCIA09_141_1': 1, 'LGG__Brats18_TCIA13_633_1': 1, 'HGG__Brats18_2013_2_1': 0, 'HGG__Brats18_TCIA02_430_1': 0, 'HGG__Brats18_CBICA_APR_1': 0, 'HGG__Brats18_TCIA02_491_1': 0, 'HGG__Brats18_CBICA_BHB_1': 0, 'HGG__Brats18_TCIA03_257_1': 0, 'HGG__Brats18_TCIA08_242_1': 0, 'HGG__Brats18_CBICA_AQP_1': 0, 'HGG__Brats18_TCIA08_280_1': 0, 'LGG__Brats18_TCIA13_615_1': 1, 'HGG__Brats18_TCIA02_606_1': 0, 'HGG__Brats18_CBICA_AQY_1': 0, 'HGG__Brats18_TCIA01_425_1': 0, 'HGG__Brats18_TCIA02_300_1': 0, 'HGG__Brats18_TCIA01_150_1': 0, 'HGG__Brats18_TCIA02_370_1': 0, 'LGG__Brats18_TCIA13_634_1': 1, 'LGG__Brats18_TCIA09_620_1': 1, 'HGG__Brats18_CBICA_ASU_1': 0, 'HGG__Brats18_CBICA_AOH_1': 0, 'HGG__Brats18_TCIA02_321_1': 0, 'HGG__Brats18_TCIA02_322_1': 0, 'HGG__Brats18_CBICA_AYW_1': 0, 'HGG__Brats18_TCIA01_499_1': 0, 'HGG__Brats18_2013_18_1': 0, 'HGG__Brats18_TCIA02_283_1': 0, 'HGG__Brats18_TCIA01_390_1': 0, 'HGG__Brats18_TCIA06_184_1': 0, 'HGG__Brats18_TCIA02_394_1': 0, 'HGG__Brats18_CBICA_AVV_1': 0, 'HGG__Brats18_TCIA06_211_1': 0, 'HGG__Brats18_CBICA_AAB_1': 0, 'LGG__Brats18_TCIA10_152_1': 1, 'HGG__Brats18_TCIA08_162_1': 0, 'HGG__Brats18_CBICA_AUR_1': 0, 'HGG__Brats18_TCIA04_111_1': 0, 'HGG__Brats18_CBICA_AQO_1': 0, 'HGG__Brats18_CBICA_AQV_1': 0, 'HGG__Brats18_CBICA_AWH_1': 0, 'HGG__Brats18_CBICA_AYU_1': 0, 'HGG__Brats18_CBICA_AXQ_1': 0, 'HGG__Brats18_TCIA08_167_1': 0, 'LGG__Brats18_2013_6_1': 1, 'HGG__Brats18_CBICA_ASA_1': 0, 'HGG__Brats18_TCIA02_374_1': 0, 'HGG__Brats18_TCIA02_274_1': 0, 'HGG__Brats18_TCIA02_455_1': 0, 'HGG__Brats18_TCIA02_117_1': 0, 'HGG__Brats18_CBICA_ATX_1': 0, 'HGG__Brats18_TCIA02_368_1': 0, 'LGG__Brats18_TCIA12_101_1': 1, 'LGG__Brats18_TCIA12_470_1': 1, 'HGG__Brats18_TCIA03_199_1': 0, 'HGG__Brats18_TCIA01_335_1': 0, 'LGG__Brats18_2013_8_1': 1, 'HGG__Brats18_CBICA_AQQ_1': 0, 'HGG__Brats18_TCIA02_208_1': 0, 'HGG__Brats18_TCIA01_203_1': 0, 'HGG__Brats18_TCIA01_221_1': 0, 'HGG__Brats18_2013_5_1': 0, 'HGG__Brats18_CBICA_AZH_1': 0, 'HGG__Brats18_TCIA03_138_1': 0, 'LGG__Brats18_2013_28_1': 1, 'HGG__Brats18_CBICA_AAL_1': 0, 'LGG__Brats18_TCIA09_255_1': 1, 'LGG__Brats18_TCIA10_387_1': 1, 'HGG__Brats18_TCIA02_605_1': 0, 'HGG__Brats18_2013_11_1': 0, 'LGG__Brats18_TCIA10_175_1': 1, 'LGG__Brats18_TCIA13_645_1': 1, 'LGG__Brats18_2013_15_1': 1, 'HGG__Brats18_CBICA_ABN_1': 0, 'LGG__Brats18_TCIA10_632_1': 1, 'LGG__Brats18_TCIA09_312_1': 1, 'HGG__Brats18_CBICA_ABM_1': 0, 'HGG__Brats18_TCIA05_277_1': 0, 'LGG__Brats18_TCIA10_637_1': 1, 'HGG__Brats18_TCIA02_309_1': 0, 'HGG__Brats18_CBICA_AXN_1': 0, 'HGG__Brats18_2013_26_1': 0, 'HGG__Brats18_CBICA_ASE_1': 0, 'LGG__Brats18_2013_16_1': 1, 'LGG__Brats18_2013_9_1': 1, 'HGG__Brats18_TCIA01_231_1': 0, 'LGG__Brats18_TCIA13_642_1': 1, 'HGG__Brats18_CBICA_ATP_1': 0, 'HGG__Brats18_TCIA01_186_1': 0, 'HGG__Brats18_CBICA_BHK_1': 0, 'HGG__Brats18_2013_12_1': 0, 'LGG__Brats18_TCIA10_442_1': 1, 'HGG__Brats18_TCIA03_338_1': 0, 'LGG__Brats18_TCIA10_202_1': 1, 'HGG__Brats18_CBICA_ANG_1': 0, 'LGG__Brats18_TCIA09_254_1': 1, 'LGG__Brats18_2013_1_1': 1, 'LGG__Brats18_TCIA12_480_1': 1, 'HGG__Brats18_CBICA_ARZ_1': 0, 'HGG__Brats18_TCIA08_469_1': 0, 'HGG__Brats18_TCIA04_437_1': 0, 'HGG__Brats18_TCIA03_265_1': 0, 'HGG__Brats18_CBICA_ALN_1': 0, 'LGG__Brats18_TCIA09_451_1': 1, 'HGG__Brats18_2013_21_1': 0, 'HGG__Brats18_TCIA05_478_1': 0, 'HGG__Brats18_2013_17_1': 0, 'LGG__Brats18_TCIA10_241_1': 1, 'HGG__Brats18_TCIA01_401_1': 0, 'HGG__Brats18_TCIA01_235_1': 0, 'HGG__Brats18_TCIA01_147_1': 0, 'HGG__Brats18_CBICA_ANZ_1': 0, 'HGG__Brats18_TCIA04_149_1': 0, 'LGG__Brats18_TCIA09_462_1': 1, 'HGG__Brats18_2013_25_1': 0, 'HGG__Brats18_CBICA_AXM_1': 0, 'HGG__Brats18_TCIA02_608_1': 0, 'HGG__Brats18_CBICA_ABY_1': 0, 'HGG__Brats18_CBICA_AXW_1': 0, 'HGG__Brats18_CBICA_AQG_1': 0, 'HGG__Brats18_2013_4_1': 0, 'LGG__Brats18_TCIA10_628_1': 1, 'HGG__Brats18_CBICA_ASV_1': 0, 'HGG__Brats18_CBICA_ATD_1': 0, 'HGG__Brats18_TCIA01_180_1': 0, 'HGG__Brats18_CBICA_AQZ_1': 0, 'HGG__Brats18_CBICA_AQT_1': 0, 'HGG__Brats18_TCIA08_234_1': 0, 'HGG__Brats18_CBICA_AUN_1': 0, 'HGG__Brats18_CBICA_BHM_1': 0, 'LGG__Brats18_TCIA10_130_1': 1, 'LGG__Brats18_TCIA12_466_1': 1, 'HGG__Brats18_CBICA_AOP_1': 0, 'LGG__Brats18_TCIA10_449_1': 1, 'HGG__Brats18_TCIA04_479_1': 0, 'HGG__Brats18_TCIA02_377_1': 0, 'HGG__Brats18_CBICA_ALX_1': 0, 'LGG__Brats18_TCIA13_630_1': 1, 'HGG__Brats18_CBICA_AQJ_1': 0, 'HGG__Brats18_CBICA_AAG_1': 0, 'LGG__Brats18_TCIA10_310_1': 1, 'HGG__Brats18_TCIA02_179_1': 0, 'HGG__Brats18_TCIA02_151_1': 0, 'HGG__Brats18_TCIA02_222_1': 0, 'HGG__Brats18_CBICA_AUQ_1': 0, 'LGG__Brats18_TCIA10_351_1': 1, 'HGG__Brats18_TCIA03_419_1': 0, 'HGG__Brats18_CBICA_ABB_1': 0, 'HGG__Brats18_TCIA02_471_1': 0, 'HGG__Brats18_TCIA08_218_1': 0, 'HGG__Brats18_CBICA_AYI_1': 0, 'HGG__Brats18_CBICA_ASO_1': 0, 'HGG__Brats18_CBICA_APY_1': 0, 'HGG__Brats18_CBICA_AOZ_1': 0, 'HGG__Brats18_CBICA_ASW_1': 0, 'LGG__Brats18_TCIA10_639_1': 1, 'HGG__Brats18_TCIA03_121_1': 0, 'LGG__Brats18_TCIA10_109_1': 1, 'HGG__Brats18_CBICA_APZ_1': 0, 'HGG__Brats18_TCIA01_190_1': 0, 'HGG__Brats18_TCIA06_247_1': 0, 'LGG__Brats18_2013_24_1': 1, 'HGG__Brats18_CBICA_ASN_1': 0, 'LGG__Brats18_TCIA10_410_1': 1, 'LGG__Brats18_TCIA09_493_1': 1, 'LGG__Brats18_TCIA10_282_1': 1, 'HGG__Brats18_CBICA_ASK_1': 0, 'HGG__Brats18_CBICA_ARF_1': 0, 'HGG__Brats18_CBICA_AQA_1': 0, 'HGG__Brats18_TCIA02_473_1': 0, 'HGG__Brats18_TCIA01_411_1': 0, 'HGG__Brats18_TCIA08_319_1': 0, 'HGG__Brats18_CBICA_ATF_1': 0, 'HGG__Brats18_CBICA_AAP_1': 0, 'HGG__Brats18_TCIA05_396_1': 0, 'HGG__Brats18_CBICA_AOD_1': 0, 'HGG__Brats18_TCIA01_460_1': 0, 'HGG__Brats18_CBICA_AMH_1': 0, 'LGG__Brats18_TCIA10_393_1': 1, 'HGG__Brats18_TCIA06_372_1': 0, 'HGG__Brats18_CBICA_AQN_1': 0, 'LGG__Brats18_TCIA10_629_1': 1, 'HGG__Brats18_TCIA08_278_1': 0, 'LGG__Brats18_TCIA10_299_1': 1, 'HGG__Brats18_2013_13_1': 0, 'HGG__Brats18_CBICA_ARW_1': 0, 'HGG__Brats18_2013_19_1': 0, 'HGG__Brats18_CBICA_ATV_1': 0, 'LGG__Brats18_TCIA12_298_1': 1, 'HGG__Brats18_2013_23_1': 0, 'HGG__Brats18_2013_7_1': 0, 'LGG__Brats18_TCIA10_276_1': 1, 'HGG__Brats18_2013_14_1': 0, 'HGG__Brats18_TCIA03_375_1': 0, 'LGG__Brats18_TCIA13_618_1': 1, 'HGG__Brats18_TCIA01_412_1': 0, 'HGG__Brats18_TCIA02_607_1': 0, 'HGG__Brats18_CBICA_ASG_1': 0, 'HGG__Brats18_CBICA_AYA_1': 0, 'LGG__Brats18_TCIA10_413_1': 1, 'HGG__Brats18_TCIA08_205_1': 0, 'LGG__Brats18_TCIA13_653_1': 1, 'HGG__Brats18_CBICA_ABE_1': 0, 'HGG__Brats18_TCIA03_296_1': 0, 'LGG__Brats18_TCIA10_346_1': 1, 'LGG__Brats18_TCIA10_408_1': 1, 'HGG__Brats18_TCIA01_201_1': 0, 'HGG__Brats18_CBICA_AZD_1': 0, 'LGG__Brats18_TCIA09_177_1': 1, 'HGG__Brats18_CBICA_ALU_1': 0, 'LGG__Brats18_TCIA13_623_1': 1, 'HGG__Brats18_TCIA01_448_1': 0, 'HGG__Brats18_TCIA01_131_1': 0, 'HGG__Brats18_TCIA03_133_1': 0, 'HGG__Brats18_CBICA_ATB_1': 0, 'HGG__Brats18_TCIA02_168_1': 0, 'HGG__Brats18_TCIA06_165_1': 0, 'HGG__Brats18_CBICA_ANI_1': 0, 'HGG__Brats18_TCIA02_198_1': 0, 'HGG__Brats18_TCIA06_332_1': 0, 'HGG__Brats18_TCIA01_378_1': 0, 'HGG__Brats18_TCIA04_192_1': 0, 'HGG__Brats18_TCIA08_105_1': 0, 'HGG__Brats18_TCIA03_498_1': 0, 'HGG__Brats18_CBICA_AXL_1': 0, 'LGG__Brats18_TCIA13_621_1': 1, 'HGG__Brats18_TCIA02_135_1': 0, 'HGG__Brats18_2013_20_1': 0, 'HGG__Brats18_TCIA06_603_1': 0, 'HGG__Brats18_CBICA_ASH_1': 0, 'HGG__Brats18_CBICA_AOO_1': 0, 'HGG__Brats18_TCIA08_436_1': 0, 'LGG__Brats18_TCIA10_261_1': 1, 'LGG__Brats18_TCIA09_402_1': 1, 'HGG__Brats18_CBICA_AVG_1': 0, 'HGG__Brats18_CBICA_ABO_1': 0, 'HGG__Brats18_TCIA04_361_1': 0, 'HGG__Brats18_2013_10_1': 0, 'HGG__Brats18_CBICA_ASY_1': 0, 'LGG__Brats18_TCIA10_644_1': 1, 'HGG__Brats18_TCIA02_331_1': 0, 'LGG__Brats18_TCIA10_103_1': 1, 'HGG__Brats18_TCIA02_290_1': 0, 'HGG__Brats18_TCIA08_406_1': 0, 'HGG__Brats18_CBICA_AVJ_1': 0, 'HGG__Brats18_TCIA02_118_1': 0, 'HGG__Brats18_TCIA05_444_1': 0, 'HGG__Brats18_2013_27_1': 0}

            # {'AEPL_003': 6, 'AEPL_004': 2, 'AEPL_005': 4, 'AEPL_007': 6, 'AEPL_009': 0, 'AEPL_010': 4, 'AEPL_022': 6, 'AEPL_025': 2, 'AEPL_028': 6, 'AEPL_032': 6, 'AEPL_033': 3, 'AEPL_035': 2, 'AEPL_036': 4, 'AEPL_038': 6, 'AEPL_042': 2, 'AEPL_043': 3, 'AEPL_044': 4, 'AEPL_045': 3, 'AEPL_047': 6, 'AEPL_049': 4, 'AEPL_050': 6, 'AEPL_496': 6, 'AEPL_541': 6, 'AEPL_559': 2, 'AEPL_562': 4, 'AEPL_567': 6, 'AEPL_568': 4, 'AEPL_573': 2, 'AEPL_577': 6, 'AEPL_580': 6, 'AEPL_581': 6, 'AEPL_582': 6, 'AEPL_584': 6, 'AEPL_586': 6, 'AEPL_588': 2, 'AEPL_591': 6, 'AEPL_593': 6, 'AEPL_596': 6, 'AEPL_597': 6, 'AEPL_598': 6, 'AEPL_604': 6, 'AEPL_605': 6, 'AEPL_607': 6, 'AEPL_612': 6, 'AEPL_613': 6, 'AEPL_615': 4, 'AEPL_618': 6, 'AEPL_619': 6, 'AEPL_622': 6, 'AEPL_623': 2, 'AEPL_628': 6, 'AEPL_632': 6}
    def get_do_oversample(self, batch_idx):
        return not batch_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))

    def determine_shapes(self):
        if self.has_prev_stage:
            num_seg = 2
        else:
            num_seg = 1

        k = list(self._data.keys())[0]
        if isfile(self._data[k]['data_file'][:-4] + ".npy"):
            case_all_data = np.load(self._data[k]['data_file'][:-4] + ".npy", self.memmap_mode)
        else:
            case_all_data = np.load(self._data[k]['data_file'])['data']
        num_color_channels = case_all_data.shape[0] - 1
        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, num_seg, *self.patch_size)
        return data_shape, seg_shape

    def generate_train_batch(self):
        selected_keys = np.random.choice(self.list_of_keys, self.batch_size, True, None)
        data = np.zeros(self.data_shape, dtype=np.float32)
        seg = np.zeros(self.seg_shape, dtype=np.float32)
        case_properties = []
        birads_gt = []
        for j, i in enumerate(selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            birads_gt.append(self.birads[i])

            if self.get_do_oversample(j):
                force_fg = True
            else:
                force_fg = False

            if 'properties' in self._data[i].keys():
                properties = self._data[i]['properties']
            else:
                properties = load_pickle(self._data[i]['properties_file'])
            case_properties.append(properties)

            # cases are stored as npz, but we require unpack_dataset to be run. This will decompress them into npy
            # which is much faster to access
            if isfile(self._data[i]['data_file'][:-4] + ".npy"):
                case_all_data = np.load(self._data[i]['data_file'][:-4] + ".npy", self.memmap_mode)
            else:
                case_all_data = np.load(self._data[i]['data_file'])['data']

            # If we are doing the cascade then we will also need to load the segmentation of the previous stage and
            # concatenate it. Here it will be concatenates to the segmentation because the augmentations need to be
            # applied to it in segmentation mode. Later in the data augmentation we move it from the segmentations to
            # the last channel of the data
            if self.has_prev_stage:
                if isfile(self._data[i]['seg_from_prev_stage_file'][:-4] + ".npy"):
                    segs_from_previous_stage = np.load(self._data[i]['seg_from_prev_stage_file'][:-4] + ".npy",
                                                       mmap_mode=self.memmap_mode)[None]
                else:
                    segs_from_previous_stage = np.load(self._data[i]['seg_from_prev_stage_file'])['data'][None]
                # we theoretically support several possible previsous segmentations from which only one is sampled. But
                # in practice this feature was never used so it's always only one segmentation
                seg_key = np.random.choice(segs_from_previous_stage.shape[0])
                seg_from_previous_stage = segs_from_previous_stage[seg_key:seg_key + 1]
                assert all([i == j for i, j in zip(seg_from_previous_stage.shape[1:], case_all_data.shape[1:])]), \
                    "seg_from_previous_stage does not match the shape of case_all_data: %s vs %s" % \
                    (str(seg_from_previous_stage.shape[1:]), str(case_all_data.shape[1:]))
            else:
                seg_from_previous_stage = None

            # do you trust me? You better do. Otherwise you'll have to go through this mess and honestly there are
            # better things you could do right now

            # (above) documentation of the day. Nice. Even myself coming back 1 months later I have not friggin idea
            # what's going on. I keep the above documentation just for fun but attempt to make things clearer now

            need_to_pad = self.need_to_pad.copy()
            for d in range(3):
                # if case_all_data.shape + need_to_pad is still < patch size we need to pad more! We pad on both sides
                # always
                if need_to_pad[d] + case_all_data.shape[d + 1] < self.patch_size[d]:
                    need_to_pad[d] = self.patch_size[d] - case_all_data.shape[d + 1]

            # we can now choose the bbox from -need_to_pad // 2 to shape - patch_size + need_to_pad // 2. Here we
            # define what the upper and lower bound can be to then sample from them with np.random.randint
            shape = case_all_data.shape[1:]
            lb_x = - need_to_pad[0] // 2
            ub_x = shape[0] + need_to_pad[0] // 2 + need_to_pad[0] % 2 - self.patch_size[0]
            lb_y = - need_to_pad[1] // 2
            ub_y = shape[1] + need_to_pad[1] // 2 + need_to_pad[1] % 2 - self.patch_size[1]
            lb_z = - need_to_pad[2] // 2
            ub_z = shape[2] + need_to_pad[2] // 2 + need_to_pad[2] % 2 - self.patch_size[2]

            # if not force_fg then we can just sample the bbox randomly from lb and ub. Else we need to make sure we get
            # at least one of the foreground classes in the patch
            if not force_fg:
                bbox_x_lb = np.random.randint(lb_x, ub_x + 1)
                bbox_y_lb = np.random.randint(lb_y, ub_y + 1)
                bbox_z_lb = np.random.randint(lb_z, ub_z + 1)
            else:
                # these values should have been precomputed
                if 'class_locations' not in properties.keys():
                    raise RuntimeError("Please rerun the preprocessing with the newest version of nnU-Net!")

                # this saves us a np.unique. Preprocessing already did that for all cases. Neat.
                foreground_classes = np.array(
                    [i for i in properties['class_locations'].keys() if len(properties['class_locations'][i]) != 0])
                foreground_classes = foreground_classes[foreground_classes > 0]

                if len(foreground_classes) == 0:
                    # this only happens if some image does not contain foreground voxels at all
                    selected_class = None
                    voxels_of_that_class = None
                    print('case does not contain any foreground classes', i)
                else:
                    selected_class = np.random.choice(foreground_classes)

                    voxels_of_that_class = properties['class_locations'][selected_class]

                if voxels_of_that_class is not None:
                    selected_voxel = voxels_of_that_class[np.random.choice(len(voxels_of_that_class))]
                    # selected voxel is center voxel. Subtract half the patch size to get lower bbox voxel.
                    # Make sure it is within the bounds of lb and ub
                    bbox_x_lb = max(lb_x, selected_voxel[0] - self.patch_size[0] // 2)
                    bbox_y_lb = max(lb_y, selected_voxel[1] - self.patch_size[1] // 2)
                    bbox_z_lb = max(lb_z, selected_voxel[2] - self.patch_size[2] // 2)
                else:
                    # If the image does not contain any foreground classes, we fall back to random cropping
                    bbox_x_lb = np.random.randint(lb_x, ub_x + 1)
                    bbox_y_lb = np.random.randint(lb_y, ub_y + 1)
                    bbox_z_lb = np.random.randint(lb_z, ub_z + 1)

            bbox_x_ub = bbox_x_lb + self.patch_size[0]
            bbox_y_ub = bbox_y_lb + self.patch_size[1]
            bbox_z_ub = bbox_z_lb + self.patch_size[2]

            # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
            # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
            # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
            # later
            valid_bbox_x_lb = max(0, bbox_x_lb)
            valid_bbox_x_ub = min(shape[0], bbox_x_ub)
            valid_bbox_y_lb = max(0, bbox_y_lb)
            valid_bbox_y_ub = min(shape[1], bbox_y_ub)
            valid_bbox_z_lb = max(0, bbox_z_lb)
            valid_bbox_z_ub = min(shape[2], bbox_z_ub)

            # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
            # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
            # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
            # remove label -1 in the data augmentation but this way it is less error prone)
            case_all_data = np.copy(case_all_data[:, valid_bbox_x_lb:valid_bbox_x_ub,
                                    valid_bbox_y_lb:valid_bbox_y_ub,
                                    valid_bbox_z_lb:valid_bbox_z_ub])
            if seg_from_previous_stage is not None:
                seg_from_previous_stage = seg_from_previous_stage[:, valid_bbox_x_lb:valid_bbox_x_ub,
                                          valid_bbox_y_lb:valid_bbox_y_ub,
                                          valid_bbox_z_lb:valid_bbox_z_ub]

            data[j] = np.pad(case_all_data[:-1], ((0, 0),
                                                  (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                                  (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0)),
                                                  (-min(0, bbox_z_lb), max(bbox_z_ub - shape[2], 0))),
                             self.pad_mode, **self.pad_kwargs_data)

            seg[j, 0] = np.pad(case_all_data[-1:], ((0, 0),
                                                    (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                                    (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0)),
                                                    (-min(0, bbox_z_lb), max(bbox_z_ub - shape[2], 0))),
                               'constant', **{'constant_values': -1})
            if seg_from_previous_stage is not None:
                seg[j, 1] = np.pad(seg_from_previous_stage, ((0, 0),
                                                             (-min(0, bbox_x_lb),
                                                              max(bbox_x_ub - shape[0], 0)),
                                                             (-min(0, bbox_y_lb),
                                                              max(bbox_y_ub - shape[1], 0)),
                                                             (-min(0, bbox_z_lb),
                                                              max(bbox_z_ub - shape[2], 0))),
                                   'constant', **{'constant_values': 0})

        return {'data': data, 'seg': seg, 'properties': case_properties, 'keys': selected_keys, 'birads_gt':birads_gt}

class DataLoader2D(SlimDataLoaderBase):
    def __init__(self, data, patch_size, final_patch_size, batch_size, oversample_foreground_percent=0.0,
                 memmap_mode="r", pseudo_3d_slices=1, pad_mode="edge",
                 pad_kwargs_data=None, pad_sides=None):
        """
        This is the basic data loader for 2D networks. It uses preprocessed data as produced by my (Fabian) preprocessing.
        You can load the data with load_dataset(folder) where folder is the folder where the npz files are located. If there
        are only npz files present in that folder, the data loader will unpack them on the fly. This may take a while
        and increase CPU usage. Therefore, I advise you to call unpack_dataset(folder) first, which will unpack all npz
        to npy. Don't forget to call delete_npy(folder) after you are done with training?
        Why all the hassle? Well the decathlon dataset is huge. Using npy for everything will consume >1 TB and that is uncool
        given that I (Fabian) will have to store that permanently on /datasets and my local computer. With this strategy all
        data is stored in a compressed format (factor 10 smaller) and only unpacked when needed.
        :param data: get this with load_dataset(folder, stage=0). Plug the return value in here and you are g2g (good to go)
        :param patch_size: what patch size will this data loader return? it is common practice to first load larger
        patches so that a central crop after data augmentation can be done to reduce border artifacts. If unsure, use
        get_patch_size() from data_augmentation.default_data_augmentation
        :param final_patch_size: what will the patch finally be cropped to (after data augmentation)? this is the patch
        size that goes into your network. We need this here because we will pad patients in here so that patches at the
        border of patients are sampled properly
        :param batch_size:
        :param num_batches: how many batches will the data loader produce before stopping? None=endless
        :param seed:
        :param stage: ignore this (Fabian only)
        :param transpose: ignore this
        :param random: sample randomly; CAREFUL! non-random sampling requires batch_size=1, otherwise you will iterate batch_size times over the dataset
        :param pseudo_3d_slices: 7 = 3 below and 3 above the center slice
        """
        super(DataLoader2D, self).__init__(data, batch_size, None)
        if pad_kwargs_data is None:
            pad_kwargs_data = OrderedDict()
        self.pad_kwargs_data = pad_kwargs_data
        self.pad_mode = pad_mode
        self.pseudo_3d_slices = pseudo_3d_slices
        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.patch_size = patch_size
        self.list_of_keys = list(self._data.keys())
        self.need_to_pad = np.array(patch_size) - np.array(final_patch_size)
        self.memmap_mode = memmap_mode
        if pad_sides is not None:
            if not isinstance(pad_sides, np.ndarray):
                pad_sides = np.array(pad_sides)
            self.need_to_pad += pad_sides
        self.pad_sides = pad_sides
        self.data_shape, self.seg_shape = self.determine_shapes()

    def determine_shapes(self):
        num_seg = 1

        k = list(self._data.keys())[0]
        if isfile(self._data[k]['data_file'][:-4] + ".npy"):
            case_all_data = np.load(self._data[k]['data_file'][:-4] + ".npy", self.memmap_mode)
        else:
            case_all_data = np.load(self._data[k]['data_file'])['data']
        num_color_channels = case_all_data.shape[0] - num_seg
        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, num_seg, *self.patch_size)
        return data_shape, seg_shape

    def get_do_oversample(self, batch_idx):
        return not batch_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))

    def generate_train_batch(self):
        selected_keys = np.random.choice(self.list_of_keys, self.batch_size, True, None)

        data = np.zeros(self.data_shape, dtype=np.float32)
        seg = np.zeros(self.seg_shape, dtype=np.float32)

        case_properties = []
        for j, i in enumerate(selected_keys):
            if 'properties' in self._data[i].keys():
                properties = self._data[i]['properties']
            else:
                properties = load_pickle(self._data[i]['properties_file'])
            case_properties.append(properties)

            if self.get_do_oversample(j):
                force_fg = True
            else:
                force_fg = False

            if not isfile(self._data[i]['data_file'][:-4] + ".npy"):
                # lets hope you know what you're doing
                case_all_data = np.load(self._data[i]['data_file'][:-4] + ".npz")['data']
            else:
                case_all_data = np.load(self._data[i]['data_file'][:-4] + ".npy", self.memmap_mode)

            # this is for when there is just a 2d slice in case_all_data (2d support)
            if len(case_all_data.shape) == 3:
                case_all_data = case_all_data[:, None]

            # first select a slice. This can be either random (no force fg) or guaranteed to contain some class
            if not force_fg:
                random_slice = np.random.choice(case_all_data.shape[1])
                selected_class = None
            else:
                # these values should have been precomputed
                if 'class_locations' not in properties.keys():
                    raise RuntimeError("Please rerun the preprocessing with the newest version of nnU-Net!")

                foreground_classes = np.array(
                    [i for i in properties['class_locations'].keys() if len(properties['class_locations'][i]) != 0])
                foreground_classes = foreground_classes[foreground_classes > 0]
                if len(foreground_classes) == 0:
                    selected_class = None
                    random_slice = np.random.choice(case_all_data.shape[1])
                    print('case does not contain any foreground classes', i)
                else:
                    selected_class = np.random.choice(foreground_classes)

                    voxels_of_that_class = properties['class_locations'][selected_class]
                    valid_slices = np.unique(voxels_of_that_class[:, 0])
                    random_slice = np.random.choice(valid_slices)
                    voxels_of_that_class = voxels_of_that_class[voxels_of_that_class[:, 0] == random_slice]
                    voxels_of_that_class = voxels_of_that_class[:, 1:]

            # now crop case_all_data to contain just the slice of interest. If we want additional slice above and
            # below the current slice, here is where we get them. We stack those as additional color channels
            if self.pseudo_3d_slices == 1:
                case_all_data = case_all_data[:, random_slice]
            else:
                # this is very deprecated and will probably not work anymore. If you intend to use this you need to
                # check this!
                mn = random_slice - (self.pseudo_3d_slices - 1) // 2
                mx = random_slice + (self.pseudo_3d_slices - 1) // 2 + 1
                valid_mn = max(mn, 0)
                valid_mx = min(mx, case_all_data.shape[1])
                case_all_seg = case_all_data[-1:]
                case_all_data = case_all_data[:-1]
                case_all_data = case_all_data[:, valid_mn:valid_mx]
                case_all_seg = case_all_seg[:, random_slice]
                need_to_pad_below = valid_mn - mn
                need_to_pad_above = mx - valid_mx
                if need_to_pad_below > 0:
                    shp_for_pad = np.array(case_all_data.shape)
                    shp_for_pad[1] = need_to_pad_below
                    case_all_data = np.concatenate((np.zeros(shp_for_pad), case_all_data), 1)
                if need_to_pad_above > 0:
                    shp_for_pad = np.array(case_all_data.shape)
                    shp_for_pad[1] = need_to_pad_above
                    case_all_data = np.concatenate((case_all_data, np.zeros(shp_for_pad)), 1)
                case_all_data = case_all_data.reshape((-1, case_all_data.shape[-2], case_all_data.shape[-1]))
                case_all_data = np.concatenate((case_all_data, case_all_seg), 0)

            # case all data should now be (c, x, y)
            assert len(case_all_data.shape) == 3

            # we can now choose the bbox from -need_to_pad // 2 to shape - patch_size + need_to_pad // 2. Here we
            # define what the upper and lower bound can be to then sample from them with np.random.randint

            need_to_pad = self.need_to_pad.copy()
            for d in range(2):
                # if case_all_data.shape + need_to_pad is still < patch size we need to pad more! We pad on both sides
                # always
                if need_to_pad[d] + case_all_data.shape[d + 1] < self.patch_size[d]:
                    need_to_pad[d] = self.patch_size[d] - case_all_data.shape[d + 1]

            shape = case_all_data.shape[1:]
            lb_x = - need_to_pad[0] // 2
            ub_x = shape[0] + need_to_pad[0] // 2 + need_to_pad[0] % 2 - self.patch_size[0]
            lb_y = - need_to_pad[1] // 2
            ub_y = shape[1] + need_to_pad[1] // 2 + need_to_pad[1] % 2 - self.patch_size[1]

            # if not force_fg then we can just sample the bbox randomly from lb and ub. Else we need to make sure we get
            # at least one of the foreground classes in the patch
            if not force_fg or selected_class is None:
                bbox_x_lb = np.random.randint(lb_x, ub_x + 1)
                bbox_y_lb = np.random.randint(lb_y, ub_y + 1)
            else:
                # this saves us a np.unique. Preprocessing already did that for all cases. Neat.
                selected_voxel = voxels_of_that_class[np.random.choice(len(voxels_of_that_class))]
                # selected voxel is center voxel. Subtract half the patch size to get lower bbox voxel.
                # Make sure it is within the bounds of lb and ub
                bbox_x_lb = max(lb_x, selected_voxel[0] - self.patch_size[0] // 2)
                bbox_y_lb = max(lb_y, selected_voxel[1] - self.patch_size[1] // 2)

            bbox_x_ub = bbox_x_lb + self.patch_size[0]
            bbox_y_ub = bbox_y_lb + self.patch_size[1]

            # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
            # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
            # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
            # later
            valid_bbox_x_lb = max(0, bbox_x_lb)
            valid_bbox_x_ub = min(shape[0], bbox_x_ub)
            valid_bbox_y_lb = max(0, bbox_y_lb)
            valid_bbox_y_ub = min(shape[1], bbox_y_ub)

            # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
            # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
            # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
            # remove label -1 in the data augmentation but this way it is less error prone)

            case_all_data = case_all_data[:, valid_bbox_x_lb:valid_bbox_x_ub,
                            valid_bbox_y_lb:valid_bbox_y_ub]

            case_all_data_donly = np.pad(case_all_data[:-1], ((0, 0),
                                                              (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                                              (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0))),
                                         self.pad_mode, **self.pad_kwargs_data)

            case_all_data_segonly = np.pad(case_all_data[-1:], ((0, 0),
                                                                (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                                                (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0))),
                                           'constant', **{'constant_values': -1})

            data[j] = case_all_data_donly
            seg[j] = case_all_data_segonly

        keys = selected_keys
        return {'data': data, 'seg': seg, 'properties': case_properties, "keys": keys}


if __name__ == "__main__":
    t = "Task002_Heart"
    p = join(preprocessing_output_dir, t, "stage1")
    dataset = load_dataset(p)
    with open(join(join(preprocessing_output_dir, t), "plans_stage1.pkl"), 'rb') as f:
        plans = pickle.load(f)
    unpack_dataset(p)
    dl = DataLoader3D(dataset, (32, 32, 32), (32, 32, 32), 2, oversample_foreground_percent=0.33)
    dl = DataLoader3D(dataset, np.array(plans['patch_size']).astype(int), np.array(plans['patch_size']).astype(int), 2,
                      oversample_foreground_percent=0.33)
    dl2d = DataLoader2D(dataset, (64, 64), np.array(plans['patch_size']).astype(int)[1:], 12,
                        oversample_foreground_percent=0.33)
