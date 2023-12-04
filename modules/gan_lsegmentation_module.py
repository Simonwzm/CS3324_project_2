import time
import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from saliency_dataset import SaliencyDataset
import metrics.metrics as salmetrics
from torchvision.transforms import ToPILImage
from .models.gan_lseg_net import GAN_Lseg, Discriminator

from argparse import ArgumentParser

import pytorch_lightning as pl

# add mixed precision
import torch.cuda.amp as amp
import numpy as np

# from encoding.utils import SegmentationMetric

class GAN_LSegmentationModule(pl.LightningModule):
    def __init__(self, data_path, dataset, batch_size, base_lr, max_epochs, **kwargs):
        super().__init__()

        self.automatic_optimization = False
        self.base_size = 512
        self.crop_size = 512

        self.G = GAN_Lseg(
            labels=None,
            backbone=kwargs["backbone"],
            features=kwargs["num_features"],
            crop_size=self.crop_size,
            arch_option=kwargs["arch_option"],
            block_depth=kwargs["block_depth"],
            activation=kwargs["activation"],
        )
        self.D = Discriminator()

        self.G.input_layer.pretrained.model.patch_embed.img_size = (
            self.crop_size,
            self.crop_size,
        )

        # self._up_kwargs = up_kwargs
 
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
        # return self.net(x, text, train_type)

        # need new gan training method
        pass
 

    def training_step(self, batch, batch_idx, optimizer_idx):
        img, target,fixation, text, train_type = batch
        real_labels = torch.ones(self.batch_size, 1).to(img.device)
        fake_labels = torch.zeros(self.batch_size, 1).to(img.device)


        opt_gen, opt_dis = self.optimizers()


        if batch_idx % 4 == 0:
            print("train Discriminator")
            inputs_real = torch.cat((img, target), 1)
            gen_image = self.G(img,text,train_type)
            # print(gen_image)
            inputs_fake = torch.cat((img, gen_image), 1)
            outputs_real = self.D(inputs_real)
            outputs_fake = self.D(inputs_fake)
            # print(outputs_fake, fake_labels)
            # print(outputs_real,real_labels)
            # exit()


            d_loss_real = self.criterion(outputs_real, real_labels)
            d_loss_fake = self.criterion(outputs_fake, fake_labels)
            d_loss = d_loss_real + d_loss_fake
            self.log("d_loss", d_loss)
            opt_dis.zero_grad()
            self.manual_backward(d_loss)
            opt_dis.step()
        else:
            # print("img,", img)
            # print("text", text)
            
            print("train Generator")
            fake_fix = self.G(img, text, train_type)
            # print(fake_fix)
            fake_inputs = torch.cat((img, fake_fix), 1)
            fake_outputs = self.D(fake_inputs)

            g_lossG = self.criterion(fake_fix, target)
            g_lossD = self.criterion(fake_outputs, real_labels)

            alpha = 0.05
            g_loss = alpha * g_lossG + g_lossD
            self.log("g_loss", g_loss)
            opt_gen.zero_grad()
            self.manual_backward(g_loss)
            opt_gen.step()

        # accumulated_grad_batches = batch_nb % 2 == 0
        # def closure_gen():
        #     print("train Generator")
        #     fake_fix = self.G(img, text, train_type)
        #     fake_inputs = torch.cat((img, fake_fix), 1)
        #     fake_outputs = self.D(fake_inputs)

        #     g_lossG = self.criterion(fake_fix, target)
        #     g_lossD = self.criterion(fake_outputs, real_labels)

        #     alpha = 0.05
        #     g_loss = alpha * g_lossG + g_lossD
        #     self.log("g_loss", g_loss)
        #     self.manual_backward(g_loss)
        #     if accumulated_grad_batches:
        #         opt_gen.zero_grad()
        # with opt_gen.toggle_model(sync_grad=accumulated_grad_batches):
        #     opt_gen.step(closure=closure_gen)   
        
        # def closure_dis():
        #     print("train Discriminator")
        #     inputs_real = torch.cat((img, target), 1)
        #     inputs_fake = torch.cat((img, self.G(img, text, train_type)), 1)
        #     outputs_real = self.D(inputs_real)
        #     outputs_fake = self.D(inputs_fake)

        #     d_loss_real = self.criterion(outputs_real, real_labels)
        #     d_loss_fake = self.criterion(outputs_fake, fake_labels)
        #     d_loss = d_loss_real + d_loss_fake
        #     self.log("d_loss", d_loss)
        #     self.manual_backward(d_loss)
        #     if accumulated_grad_batches:
        #         opt_dis.zero_grad()
        # with opt_dis.toggle_model(sync_grad=accumulated_grad_batches):
        #     opt_dis.step(closure_dis)



            







        # print(fixation.size())
        # with amp.autocast(enabled=self.enabled):
        #     out, out_bin = self(img, text, train_type)
        #     print(out_bin.size())
        #     if target:
        #         prob_loss = self.criterion(out, target)
        #     if fixation:
        #         bin_loss = self.criterion(out_bin, fixation)
        #         # use BCE loss between out
        #     loss = self.scaler.scale(prob_loss) + self.scaler.scale(bin_loss)
        # self.log("train_loss", prob_loss)
        # self.log("bin_trainloss", bin_loss)
        # self.log("all_loss", loss)
        # return loss

    def training_epoch_end(self, outs):
        pass

    def validation_step(self, batch, batch_idx):
        img, target, fixation, text, output_type = batch

        out = self.G(img, text, output_type)
        # bin_loss = self.criterion(out, target)



        val_loss = self.criterion(out, target)
        # bin_loss = self.criterion(out, fixation)


        # logging
        out = out.detach().cpu()
        target = target.detach().cpu()
        fixation = fixation.detach().cpu()


        cc_value = salmetrics.CC(out, target)
        auc_value = salmetrics.auc(out, fixation)
        nss_value = salmetrics.nss(out, fixation)
        self.batch_eval_value["cc"].append(cc_value)
        self.batch_eval_value["auc"].append(auc_value)
        self.batch_eval_value["nss"].append(nss_value)
        self.batch_eval_value["BCE_Loss"].append(val_loss)
        self.store_out.append((out, target,fixation))


    def validation_epoch_end(self, outs):
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

            out, target, fixation = self.store_out[i]
            
            res = self.to_pil_image(out[0])
            target_res = self.to_pil_image(target[0])
            # bin_out = self.to_pil_image(bin_out[0])
            fixation = self.to_pil_image(fixation[0])
            res.save(f'./vis/res/val_out{i}.png')
            target_res.save(f'./vis/tar_res/val_tar{i}.png')
            # bin_out.save(f'./vis/bin_res/val_bin_out{i}.png')
            fixation.save(f'./vis/tar_fixa/val_tar_fixa{i}.png')

        self.store_out = []


    def configure_optimizers(self):
        params_list = [
            {"params": self.G.input_layer.pretrained.parameters(), "lr": self.base_lr},
        ]
        if hasattr(self.G.input_layer, "scratch"):
            print("Found output scratch")
            params_list.append(
                {"params": self.G.input_layer.scratch.parameters(), "lr": self.base_lr * 10}
            )
        if hasattr(self.G.input_layer, "auxlayer"):
            print("Found auxlayer")
            params_list.append(
                {"params": self.G.input_layer.auxlayer.parameters(), "lr": self.base_lr * 10}
            )
        if hasattr(self.G.input_layer, "scale_inv_conv"):
            print(self.G.input_layer.scale_inv_conv)
            print("Found scaleinv layers")
            params_list.append(
                {
                    "params": self.G.input_layer.scale_inv_conv.parameters(),
                    "lr": self.base_lr * 10,
                }
            )
            params_list.append(
                {"params": self.G.input_layer.scale2_conv.parameters(), "lr": self.base_lr * 10}
            )
            params_list.append(
                {"params": self.G.input_layer.scale3_conv.parameters(), "lr": self.base_lr * 10}
            )
            params_list.append(
                {"params": self.G.input_layer.scale4_conv.parameters(), "lr": self.base_lr * 10}
            )

        # if self.other_kwargs["midasproto"]:
        #     print("Using midas optimization protocol")
            
        #     opt = torch.optim.Adam(
        #         params_list,
        #         lr=self.base_lr,
        #         betas=(0.9, 0.999),
        #         weight_decay=self.other_kwargs["weight_decay"],
        #     )
        #     sch = torch.optim.lr_scheduler.LambdaLR(
        #         opt, lambda x: pow(1.0 - x / self.epochs, 0.9)
        #     )

        # else:
            # opt = torch.optim.SGD(
            #     params_list,
            #     lr=self.base_lr,
            #     momentum=0.9,
            #     weight_decay=self.other_kwargs["weight_decay"],
            # )
            # sch = torch.optim.lr_scheduler.LambdaLR(
            #     opt, lambda x: pow(1.0 - x / self.epochs, 0.9)
            # )
        # Optimizer for Generator (G)
        opt_G = torch.optim.Adam(
            self.G.parameters(),
            lr=self.base_lr,  # Adjust as needed
            betas=(0.5, 0.999)  # Example values, adjust as needed
        )

        opt_D = torch.optim.SGD(
        self.D.parameters(),
        lr=self.base_lr,  # Adjust as needed
        momentum=0.9,
        weight_decay=self.other_kwargs["weight_decay"]
        )

    # Optionally define schedulers
        sch_G = torch.optim.lr_scheduler.LambdaLR(
            opt_G, lambda x: pow(1.0 - x / self.epochs, 0.9)
        )
        sch_D = torch.optim.lr_scheduler.LambdaLR(
            opt_D, lambda x: pow(1.0 - x / self.epochs, 0.9)
        )

        return [opt_G, opt_D], [sch_G, sch_D]
        # return [opt], [sch]

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

        if self.other_kwargs["mysetup"] == 1:
            print("load mysetup")
            mode = "train"
            print(mode)
            output_type = traintype  # Replace with the type you want to filter
            dset = SaliencyDataset(
                csv_file='./datasets/saliency/meta_data.csv',
                source_image_dir='./datasets/saliency/image/',
                target_image_dir='./datasets/saliency/map/',
                target_fixation_dir='./datasets/saliency/fixation/',
                output_type=output_type,
                image_size=(512,512),
                transform=None, #use built-in
                split="train"
            )
            self.num_classes = 2
            sample = dset[0]
            return dset




    def get_valset(self, dset, augment=False, **kwargs):
        # self.val_accuracy = pl.metrics.Accuracy()
        # self.val_iou = SegmentationMetric(self.num_classes)

        mode = "val"

        print(mode)
        if self.other_kwargs["mytraintype"]:
            print(self.other_kwargs["mytraintype"])
            traintype = self.other_kwargs["mytraintype"] 
        else:
            traintype = None
        dset2 = SaliencyDataset(
                csv_file='./datasets/saliency/meta_data.csv',
                source_image_dir='./datasets/saliency/image/',
                target_image_dir='./datasets/saliency/map/',
                target_fixation_dir='./datasets/saliency/fixation/',
                output_type=traintype,
                image_size=(512,512),
                transform=None, #use built-in
                split = "val"
            )
        return dset2



    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--data_path", type=str, help="path where dataset is stored"
        )
        parser.add_argument(
            "--dataset",
            # choices=get_available_datasets(),
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

