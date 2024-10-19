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
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss
# from nnunet.training.loss_functions.dice_loss import FocalDiceCE
from nnunet.training.loss_functions.focal_loss import FocalLossV2
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from torch import nn

class FocalDiceCE(nn.Module):
    def __init__(self, batch_dice):
        super(FocalDiceCE, self).__init__()
        self.dc = DC_and_CE_loss({'batch_dice': batch_dice, 'smooth': 1e-5, 'do_bg': False}, {})
        self.focal = FocalLossV2(apply_nonlin=nn.Softmax(dim=1), **{'alpha':0.5, 'gamma':2, 'smooth':1e-5})

    def forward(self, net_output, target):
        dc_loss = self.dc(net_output, target)
        focal_loss = self.focal(net_output, target)
        result = focal_loss + dc_loss
        return result

class nnUNetTrainerV2_SegLoss_FocalDiceCE(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage,
                                              unpack_data, deterministic, fp16)
        print("Setting up self.loss = Focal_loss({'alpha':0.75, 'gamma':2, 'smooth':1e-5})")
        self.batch_dice = batch_dice
        self.loss = FocalDiceCE(self.batch_dice)