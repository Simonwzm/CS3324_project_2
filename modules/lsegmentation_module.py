import types
import time
import random
import clip
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from saliency_dataset import SaliencyDataset
import metrics.metrics as salmetrics
from torchvision.transforms import ToPILImage


from argparse import ArgumentParser

import pytorch_lightning as pl

from data import get_dataset, get_available_datasets

from encoding.models import get_segmentation_model
from encoding.nn import SegmentationLosses

from encoding.utils import batch_pix_accuracy, batch_intersection_union

# add mixed precision
import torch.cuda.amp as amp
import numpy as np

from encoding.utils import SegmentationMetric

class LSegmentationModule(pl.LightningModule):
    def __init__(self, data_path, dataset, batch_size, base_lr, max_epochs, **kwargs):
        super().__init__()

        self.data_path = data_path
        self.batch_size = batch_size
        self.base_lr = base_lr / 16 * batch_size
        self.lr = self.base_lr

        self.epochs = max_epochs
        self.other_kwargs = kwargs
        self.enabled = False #True mixed precision will make things complicated and leading to NAN error
        self.scaler = amp.GradScaler(enabled=self.enabled)
        self.batch_eval_value = {"cc":[], "auc":[], "nss":[], "BCE_Loss": []}
        self.store_out = []
        self.to_pil_image = ToPILImage()
        

    def forward(self, x, text, train_type):
        return self.net(x, text, train_type)

    def evaluate(self, x, target=None):
        pred = self.net.forward(x)
        # print("in lseg_module eval")
        print("ineval")
        # print(pred.size())
        # print(pred[0])
        # exit()
        if isinstance(pred, (tuple, list)):
            pred = pred[0]
        if target is None:
            return pred
        correct, labeled = batch_pix_accuracy(pred.data, target.data)
        inter, union = batch_intersection_union(pred.data, target.data, self.nclass)

        return correct, labeled, inter, union

    def evaluate_random(self, x, labelset, target=None):
        pred = self.net.forward(x, labelset)
        if isinstance(pred, (tuple, list)):
            pred = pred[0]
        if target is None:
            return pred
        correct, labeled = batch_pix_accuracy(pred.data, target.data)
        inter, union = batch_intersection_union(pred.data, target.data, self.nclass)

        return correct, labeled, inter, union
    

    def training_step(self, batch, batch_nb):
        img, target,fixation, text, train_type = batch
        # target = target[:,0:1,:,:]
        # print(img.size())
        # print(target.size())
        # print(text)
        # print(train_type)
        print(fixation.size())
        with amp.autocast(enabled=self.enabled):
            out, out_bin = self(img, text, train_type)
            
            print(out_bin.size())
            if target:
                prob_loss = self.criterion(out, target)
            if fixation:
                bin_loss = self.criterion(out_bin, fixation)
                # use BCE loss between out
            loss = self.scaler.scale(prob_loss) + self.scaler.scale(bin_loss)
        self.log("train_loss", prob_loss)
        # final_output = out[0] if multi_loss else out
        # train_pred, train_gt = self._filter_invalid(final_output, target)
        # if train_gt.nelement() != 0:
        #     self.train_accuracy(train_pred, train_gt)
        self.log("bin_trainloss", bin_loss)
        self.log("all_loss", loss)
        return loss

    def training_epoch_end(self, outs):
        # self.log("train_acc_epoch", self.train_accuracy.compute())
        pass

    def validation_step(self, batch, batch_nb):
        img, target, fixation, text, output_type = batch
        # target = target[:,0:1,:,:]
        # print(img.size())
        # print(target.size())
        # print(fixation.size())
        out , out_bin= self(img, text,output_type) 
        # print("out size", out.size())
        # multi_loss = isinstance(out, tuple)
        # # print("multi_loss is tuple?", multi_loss)
        # if multi_loss:
        #     val_loss = self.criterion(*out, target)
        # else:
        if target:
            val_loss = self.criterion(out, target)
        if fixation:
            bin_loss = self.criterion(out, fixation)
        
        
        out = out.detach().cpu()
        out_bin = out_bin.detach().cpu()
        target = target.detach().cpu()
        fixation = fixation.detach().cpu()
        cc_value = salmetrics.CC(out, target)
        auc_value = salmetrics.auc(out, fixation)
        nss_value = salmetrics.nss(out, fixation)
        # print("cc", cc_value)
        # print("auc", auc_value)
        # print("nss", nss_value)
        # self.log("CC", cc_value)
        # self.log("AUC", auc_value)
        # self.log("nss", nss_value)
        self.batch_eval_value["cc"].append(cc_value)
        self.batch_eval_value["auc"].append(auc_value)
        self.batch_eval_value["nss"].append(nss_value)
        self.batch_eval_value["BCE_Loss"].append(val_loss)
        self.store_out.append((out, target, out_bin, fixation))

        # final_output = out[0] if multi_loss else out
        # valid_pred, valid_gt = self._filter_invalid(final_output, target)

        # self.val_iou.update(target, final_output)
        # pixAcc, iou = self.val_iou.get()
        # self.log("val_loss_step", val_loss)
        # self.log("pix_acc_step", pixAcc)
        # self.log(
        #     "val_acc_step",
        #     self.val_accuracy(valid_pred, valid_gt),
        # )
        # self.log("val_iou", iou)
        # self.log("BCE_Loss", val_loss)

    def validation_epoch_end(self, outs):
        # pixAcc, iou = self.val_iou.get()
        # self.log("val_acc_epoch", self.val_accuracy.compute())
        # self.log("val_iou_epoch", iou)
        # self.log("pix_acc_epoch", pixAcc)
        
        # calculate average 
        cc_avg = sum(self.batch_eval_value["cc"]) / len(self.batch_eval_value["cc"])
        auc_avg = sum(self.batch_eval_value["auc"]) / len(self.batch_eval_value["auc"])
        nss_avg = sum(self.batch_eval_value["nss"]) / len(self.batch_eval_value["nss"])
        val_loss_avg = sum(self.batch_eval_value["BCE_Loss"]) / len(self.batch_eval_value["BCE_Loss"])
        self.log("CC_epoch", cc_avg)
        self.log("AUC_epoch", auc_avg)
        self.log("nss_epoch", nss_avg)
        self.log("BCE_Loss", val_loss_avg)

        print("cc_avg", cc_avg)
        print("auc_avg", auc_avg)
        print("nss_avg", nss_avg)
        print("BCE_Loss_avg", val_loss_avg)

        # clear batch_eval_value
        self.batch_eval_value = {"cc":[], "auc":[], "nss":[], "BCE_Loss": []}

        # save outputs in self.store_out to image
        for i in range(0, len(self.store_out), 50):

            out, target, bin_out, fixation = self.store_out[i]
            print("bin_out_size: ",bin_out.size())
            print("fixation_size: ", fixation.size())
            
            res = self.to_pil_image(out[0])
            target_res = self.to_pil_image(target[0])
            bin_out = self.to_pil_image(bin_out[0])
            fixation = self.to_pil_image(fixation[0])
            res.save(f'./vis/res/val_out{i}.png')
            target_res.save(f'./vis/tar_res/val_tar{i}.png')
            bin_out.save(f'./vis/bin_res/val_bin_out{i}.png')
            fixation.save(f'./vis/tar_fixa/val_tar_fixa{i}.png')

        self.store_out = []
            




    def _filter_invalid(self, pred, target):
        valid = target != self.other_kwargs["ignore_index"]
        _, mx = torch.max(pred, dim=1)
        return mx[valid], target[valid]

    def configure_optimizers(self):
        params_list = [
            {"params": self.net.pretrained.parameters(), "lr": self.base_lr},
        ]
        if hasattr(self.net, "scratch"):
            print("Found output scratch")
            params_list.append(
                {"params": self.net.scratch.parameters(), "lr": self.base_lr * 10}
            )
        if hasattr(self.net, "auxlayer"):
            print("Found auxlayer")
            params_list.append(
                {"params": self.net.auxlayer.parameters(), "lr": self.base_lr * 10}
            )
        if hasattr(self.net, "scale_inv_conv"):
            print(self.net.scale_inv_conv)
            print("Found scaleinv layers")
            params_list.append(
                {
                    "params": self.net.scale_inv_conv.parameters(),
                    "lr": self.base_lr * 10,
                }
            )
            params_list.append(
                {"params": self.net.scale2_conv.parameters(), "lr": self.base_lr * 10}
            )
            params_list.append(
                {"params": self.net.scale3_conv.parameters(), "lr": self.base_lr * 10}
            )
            params_list.append(
                {"params": self.net.scale4_conv.parameters(), "lr": self.base_lr * 10}
            )

        if self.other_kwargs["midasproto"]:
            print("Using midas optimization protocol")
            
            opt = torch.optim.Adam(
                params_list,
                lr=self.base_lr,
                betas=(0.9, 0.999),
                weight_decay=self.other_kwargs["weight_decay"],
            )
            sch = torch.optim.lr_scheduler.LambdaLR(
                opt, lambda x: pow(1.0 - x / self.epochs, 0.9)
            )

        else:
            opt = torch.optim.SGD(
                params_list,
                lr=self.base_lr,
                momentum=0.9,
                weight_decay=self.other_kwargs["weight_decay"],
            )
            sch = torch.optim.lr_scheduler.LambdaLR(
                opt, lambda x: pow(1.0 - x / self.epochs, 0.9)
            )
        return [opt], [sch]

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=16,
            worker_init_fn=lambda x: random.seed(time.time() + x),
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.valset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=16,
        )

    def get_trainset(self, dset, augment=False, **kwargs):
        if self.other_kwargs["mytraintype"]:
            print(self.other_kwargs["mytraintype"])
            traintype = self.other_kwargs["mytraintype"] 
        else:
            traintype = None

        if self.other_kwargs["mysetup"] == 0:
            # print(kwargs)
            if augment == True:
                mode = "train_x"
            else:
                mode = "train"

            print(mode)
            dset = get_dataset(
                dset,
                root=self.data_path,
                split="train",
                mode=mode,
                transform=self.train_transform,
                **kwargs
            )

            self.num_classes = dset.num_class
            self.train_accuracy = pl.metrics.Accuracy()

            return dset
        else:
            print("load mysetup")
            mode = "train"
            print(mode)
            # Example usage:

            # Instantiate the dataset with the desired output_type
            output_type = traintype  # Replace with the type you want to filter
            dset = SaliencyDataset(
                csv_file='./datasets/saliency/meta_data.csv',
                source_image_dir='./datasets/saliency/image/',
                target_image_dir='./datasets/saliency/map/',
                target_fixation_dir='./datasets/saliency/fixation/',
                output_type=output_type,
                image_size=(520,520),
                transform=None, #use built-in
                split="train"
            )
            self.num_classes = 2
            # Get a sample from the dataset
            sample = dset[0]
            # print(sample)
            # print(len(dset))
            return dset




    def get_valset(self, dset, augment=False, **kwargs):
        self.val_accuracy = pl.metrics.Accuracy()
        self.val_iou = SegmentationMetric(self.num_classes)

        if augment == True:
            mode = "val_x"
        else:
            mode = "val"

        print(mode)
        if self.other_kwargs["mytraintype"]:
            print(self.other_kwargs["mytraintype"])
            traintype = self.other_kwargs["mytraintype"] 
        else:
            traintype = None
        # return get_dataset(
        #     dset,
        #     root=self.data_path,
        #     split="val",
        #     mode=mode,
        #     transform=self.val_transform,
        #     **kwargs
        # )
        dset2 = SaliencyDataset(
                csv_file='./datasets/saliency/meta_data.csv',
                source_image_dir='./datasets/saliency/image/',
                target_image_dir='./datasets/saliency/map/',
                target_fixation_dir='./datasets/saliency/fixation/',
                output_type=traintype,
                image_size=(520,520),
                transform=None, #use built-in
                split = "val"
            )
        return dset2


    def get_criterion(self, **kwargs):
        return SegmentationLosses(
            se_loss=kwargs["se_loss"], 
            aux=kwargs["aux"], 
            nclass=self.num_classes, 
            se_weight=kwargs["se_weight"], 
            aux_weight=kwargs["aux_weight"], 
            ignore_index=kwargs["ignore_index"], 
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--data_path", type=str, help="path where dataset is stored"
        )
        parser.add_argument(
            "--dataset",
            choices=get_available_datasets(),
            default="ade20k",
            help="dataset to train on",
        )
        parser.add_argument(
            "--batch_size", type=int, default=16, help="size of the batches"
        )
        parser.add_argument(
            "--base_lr", type=float, default=0.004, help="learning rate"
        )
        parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
        parser.add_argument(
            "--weight_decay", type=float, default=1e-4, help="weight_decay"
        )
        parser.add_argument(
            "--aux", action="store_true", default=False, help="Auxilary Loss"
        )
        parser.add_argument(
            "--aux-weight",
            type=float,
            default=0.2,
            help="Auxilary loss weight (default: 0.2)",
        )
        parser.add_argument(
            "--se-loss",
            action="store_true",
            default=False,
            help="Semantic Encoding Loss SE-loss",
        )
        parser.add_argument(
            "--se-weight", type=float, default=0.2, help="SE-loss weight (default: 0.2)"
        )

        parser.add_argument(
            "--midasproto", action="store_true", default=False, help="midasprotocol"
        )

        parser.add_argument(
            "--ignore_index",
            type=int,
            default=-1,
            help="numeric value of ignore label in gt",
        )
        parser.add_argument(
            "--augment",
            action="store_true",
            default=False,
            help="Use extended augmentations",
        )

        return parser
