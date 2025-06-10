import torch as ch
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist

ch.backends.cudnn.benchmark = True
ch.autograd.profiler.emit_nvtx(False)
ch.autograd.profiler.profile(False)

from torchvision import models
import torchmetrics
import numpy as np
from tqdm import tqdm

import os
import time
import json
from uuid import uuid4
from typing import List
from pathlib import Path
from argparse import ArgumentParser

from fastargs import get_current_config
from fastargs.decorators import param
from fastargs import Param, Section
from fastargs.validation import And, OneOf

from ffcv.pipeline.operation import Operation
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import (
    ToTensor,
    ToDevice,
    Squeeze,
    NormalizeImage,
    RandomHorizontalFlip,
    ToTorchImage,
)
from ffcv.fields.rgb_image import (
    CenterCropRGBImageDecoder,
    RandomResizedCropRGBImageDecoder,
)
from ffcv.fields.basics import IntDecoder

import pandas as pd
import utils

Section("model", "model details").params(
    arch=Param(And(str, OneOf(models.__dir__() + ["linear"])), default="resnet50"),
    pretrained=Param(int, "is pretrained? (1/0)", default=0),
)

Section("resolution", "resolution scheduling").params(
    min_res=Param(int, "the minimum (starting) resolution", default=160),
    max_res=Param(int, "the maximum (starting) resolution", default=192),
    end_ramp=Param(int, "when to stop interpolating resolution", default=76),
    start_ramp=Param(int, "when to start interpolating resolution", default=65),
)

Section("data", "data related stuff").params(
    train_dataset=Param(str, ".dat file to use for training", required=True, default="train.beton"),
    val_dataset=Param(str, ".dat file to use for validation", required=True, default="val.beton"),
    num_workers=Param(int, "The number of workers", required=True, default=32),
    in_memory=Param(int, "does the dataset fit in memory? (1/0)", required=True, default=1),
)

Section("lr", "lr scheduling").params(
    step_ratio=Param(float, "learning rate step ratio", default=0.1),
    step_length=Param(int, "learning rate step length", default=30),
    lr_schedule_type=Param(OneOf(["step", "cyclic"]), default="cyclic"),
    lr=Param(float, "learning rate", default=0.1),
    lr_peak_epoch=Param(int, "Epoch at which LR peaks", default=2),
)

Section("logging", "how to log stuff").params(
    folder=Param(str, "log location", required=True, default="./runs/"),
    log_level=Param(int, "0 if only at end 1 otherwise", default=1),
    save_model=Param(int, "save the final model or not", default=1),
)

Section("validation", "Validation parameters stuff").params(
    batch_size=Param(int, "The batch size for validation", default=256),
    resolution=Param(int, "final resized validation image size", default=256),
    lr_tta=Param(int, "should do lr flipping/avging at test time", default=1),
)

Section("training", "training hyper param stuff").params(
    seed=Param(int, "seed", default=0),
    eval_only=Param(int, "eval only?", default=0),
    batch_size=Param(int, "The batch size", default=64),
    optimizer=Param(And(str, OneOf(["sgd"])), "The optimizer", default="sgd"),
    momentum=Param(float, "SGD momentum", default=0.9),
    weight_decay=Param(float, "weight decay", default=0.00001),
    epochs=Param(int, "number of epochs", default=88),
    label_smoothing=Param(float, "label smoothing parameter", default=0.1),
    distributed=Param(int, "is distributed?", default=0),
    use_blurpool=Param(int, "use blurpool?", default=1),
    crop_scale=Param(float, "Ratio of crop in %", default=8),
    dropout1d=Param(float, "Ratio of pixel dropout in %", default=0),
    dropout2d=Param(float, "Ratio of channel dropout in %", default=0),
    loss_type=Param(And(str, OneOf(["normal", "focal", "pw", "tce", "GGF", "apstar", "CLAM"])), "type of loss", default="normal")
)

Section("dist", "distributed training options").params(
    world_size=Param(int, "number gpus", default=1),
    address=Param(str, "address", default="localhost"),
    port=Param(str, "port", default="12355"),
)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_CROP_RATIO = 224 / 256

@param("lr.lr")
@param("lr.step_ratio")
@param("lr.step_length")
@param("training.epochs")
def get_step_lr(epoch, lr, step_ratio, step_length, epochs):
    if epoch >= epochs:
        return 0

    num_steps = epoch // step_length
    return step_ratio ** num_steps * lr


@param("lr.lr")
@param("training.epochs")
@param("lr.lr_peak_epoch")
def get_cyclic_lr(epoch, lr, epochs, lr_peak_epoch):
    xs = [0, lr_peak_epoch, epochs]
    ys = [1e-4 * lr, lr, 0]
    return np.interp([epoch], xs, ys)[0]


def Focal_Loss(probs, targets, gamma):
    selected_probs = probs[range(len(probs)), targets] + 1e-8 # avoid torch.log(0)
    # FL(p) = -log(p) * (1-p) ** \gamma 
    loss = -ch.log(selected_probs) * (1-selected_probs)**gamma

    return ch.mean(loss)

def Pw_Loss(probs, targets, theta, gamma):
    selected_probs = probs[range(len(probs)), targets] + 1e-8 # avoid torch.log(0)
    # FL(p) = -log(p) * (\theta + (1-p) ** \gamma) 
    loss = -ch.log(selected_probs) * (theta + (1-selected_probs)**gamma)

    return ch.mean(loss)

class BlurPoolConv2d(ch.nn.Module):
    def __init__(self, conv):
        super().__init__()
        default_filter = ch.tensor([[[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]]) / 16.0
        filt = default_filter.repeat(conv.in_channels, 1, 1, 1)
        self.conv = conv
        self.register_buffer("blur_filter", filt)

    def forward(self, x):
        blurred = F.conv2d(
            x,
            self.blur_filter,
            stride=1,
            padding=(1, 1),
            groups=self.conv.in_channels,
            bias=None,
        )
        return self.conv.forward(blurred)


class ImageNetTrainer:
    @param("training.distributed")
    def __init__(self, gpu, distributed):
        self.all_params = get_current_config()
        self.gpu = gpu
        self.num_classes = 1000

        self.uid = str(uuid4())
        self.loss_type = self.all_params['training.loss_type']
        if self.loss_type in ['tce', 'GGF', 'apstar', 'CLAM']:
            self.class_weights = {}
            for i_class in range(self.num_classes):
                self.class_weights[i_class] = 1.0
        if self.loss_type == 'focal':
            self.gamma = 2.0
        if self.loss_type == 'pw':
            self.gamma, self.theta = 2.5, 0.8
        if self.loss_type == 'GGF':
            self.discount, self.min_weight = 0.998, 0.2
        if self.loss_type == 'apstar':
            self.K, self.K_min, self.apstar_max_loss, self.alpha = 1, 1, 10, 0.5
        if self.loss_type == 'CLAM':
            self.CLAM_start_epoch = 30
        
        self.exp_type = '{}_crop_scale{}'.format(self.all_params['training.loss_type'], self.all_params['training.crop_scale'])
        utils.set_seed_everywhere(self.all_params['training.seed'])

        if distributed:
            self.setup_distributed()

        self.train_loader = self.create_train_loader()
        self.val_loader = self.create_val_loader()
        self.model, self.scaler = self.create_model_and_scaler()
        print(self.model, self.scaler)
        self.create_optimizer()
        self.initialize_logger()

    @param("dist.address")
    @param("dist.port")
    @param("dist.world_size")
    def setup_distributed(self, address, port, world_size):
        os.environ["MASTER_ADDR"] = address
        os.environ["MASTER_PORT"] = port

        dist.init_process_group("nccl", rank=self.gpu, world_size=world_size)
        ch.cuda.set_device(self.gpu)

    def cleanup_distributed(self):
        dist.destroy_process_group()

    @param("lr.lr_schedule_type")
    def get_lr(self, epoch, lr_schedule_type):
        lr_schedules = {"cyclic": get_cyclic_lr, "step": get_step_lr}

        return lr_schedules[lr_schedule_type](epoch)

    # resolution tools
    @param("resolution.min_res")
    @param("resolution.max_res")
    @param("resolution.end_ramp")
    @param("resolution.start_ramp")
    def get_resolution(self, epoch, min_res, max_res, end_ramp, start_ramp):
        assert min_res <= max_res

        if epoch <= start_ramp:
            return min_res

        if epoch >= end_ramp:
            return max_res

        # otherwise, linearly interpolate to the nearest multiple of 32
        interp = np.interp([epoch], [start_ramp, end_ramp], [min_res, max_res])
        final_res = int(np.round(interp[0] / 32)) * 32
        return final_res

    @param("training.momentum")
    @param("training.optimizer")
    @param("training.weight_decay")
    @param("training.label_smoothing")
    @param("model.arch")
    def create_optimizer(
        self, momentum, optimizer, weight_decay, label_smoothing, arch
    ):
        assert optimizer == "sgd"

        # Only do weight decay on non-batchnorm parameters
        all_params = list(self.model.named_parameters())
        print([len(v.shape) for _, v in all_params])
        if arch == "linear":
            other_params = [v for k, v in all_params if len(v.shape) > 1]
            bn_params = [v for k, v in all_params if len(v.shape) <= 1]
        else:
            other_params = [v for k, v in all_params if not ("bn" in k)]
            bn_params = [v for k, v in all_params if ("bn" in k)]
        param_groups = [
            {"params": bn_params, "weight_decay": 0.0},
            {"params": other_params, "weight_decay": weight_decay},
        ]

        self.optimizer = ch.optim.SGD(param_groups, lr=1, momentum=momentum)
        self.loss = ch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.sample_loss = ch.nn.CrossEntropyLoss(label_smoothing=label_smoothing, reduction='none')

    @param("data.train_dataset")
    @param("data.num_workers")
    @param("training.batch_size")
    @param("training.distributed")
    @param("data.in_memory")
    @param("training.crop_scale")
    @param("model.arch")
    def create_train_loader(
        self,
        train_dataset,
        num_workers,
        batch_size,
        distributed,
        in_memory,
        crop_scale,
        arch
    ):
        print('train_dataset',train_dataset)
        this_device = f"cuda:{self.gpu}"
        train_path = Path(train_dataset)
        assert train_path.is_file()

        if arch == "linear":
            res = 256
        else:
            res = self.get_resolution(epoch=0)

        if crop_scale == 100:
            self.decoder = CenterCropRGBImageDecoder((res, res), ratio=224 / 256)
        else:
            self.decoder = RandomResizedCropRGBImageDecoder(
                (res, res), scale=(crop_scale / 100, 1.0)
            )

        image_pipeline: List[Operation] = [
            self.decoder,
            RandomHorizontalFlip(),
            ToTensor(),
            ToDevice(ch.device(this_device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16),
        ]

        label_pipeline: List[Operation] = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(ch.device(this_device), non_blocking=True),
        ]

        order = OrderOption.RANDOM if distributed else OrderOption.QUASI_RANDOM
        loader = Loader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            order=order,
            os_cache=in_memory,
            drop_last=True,
            pipelines={"image": image_pipeline, "label": label_pipeline},
            distributed=distributed
        )

        return loader

    @param("data.val_dataset")
    @param("data.num_workers")
    @param("validation.batch_size")
    @param("validation.resolution")
    @param("training.distributed")
    def create_val_loader(
        self, val_dataset, num_workers, batch_size, resolution, distributed
    ):
        print('val_dataset',val_dataset)
        this_device = f"cuda:{self.gpu}"
        val_path = Path(val_dataset)
        assert val_path.is_file()
        res_tuple = (resolution, resolution)
        cropper = CenterCropRGBImageDecoder(res_tuple, ratio=DEFAULT_CROP_RATIO)
        image_pipeline = [
            cropper,
            ToTensor(),
            ToDevice(ch.device(this_device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16),
        ]

        label_pipeline = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(ch.device(this_device), non_blocking=True),
        ]

        loader = Loader(
            val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            order=OrderOption.SEQUENTIAL,
            drop_last=False,
            pipelines={"image": image_pipeline, "label": label_pipeline},
            distributed=distributed
        )
        return loader

    @param("training.epochs")
    @param("logging.log_level")
    @param("logging.save_model")
    @param("model.arch")
    def train(self, epochs, log_level, save_model, arch):
        for epoch in range(epochs):
            self.epoch_start_time = time.time() 
            if arch != "linear":
                res = self.get_resolution(epoch)
                print('epoch {} res {}'.format(epoch, res))
                self.decoder.output_size = (res, res)
            train_loss = self.train_loop(epoch)

            print('epoch {} epoch_time {} total_time {}'.format(epoch, time.time()-self.epoch_start_time, time.time()-self.start_time))

            if log_level > 0:
                extra_dict = {"train_loss": train_loss, "epoch": epoch}
                self.eval_and_log(extra_dict=extra_dict)

        self.eval_and_log(extra_dict={"epoch": epoch})
        if self.gpu == 0 and save_model:
            ch.save(self.model.state_dict(), self.log_folder / "final_weights.pt")

    def eval_and_log(self, extra_dict={}):
        start_val = time.time()
        stats = self.val_loop()
        
        # update eval_df and save
        self.eval_df.loc[len(self.eval_df)] = np.concatenate([[extra_dict['epoch']], stats['per_class'], [np.mean(stats['per_class'])]])
        self.eval_df.to_csv(os.path.join(self.log_folder,'eval.csv'))

        val_time = time.time() - start_val
        if self.gpu == 0:
            bucket = {
                "current_lr": self.optimizer.param_groups[0]["lr"],
                "top_1": stats["top_1"],
                "top_5": stats["top_5"],
                "per_class": stats["per_class"],
                "val_time": val_time,
            }
            self.log(dict(bucket, **extra_dict))

        return stats

    @param("model.arch")
    @param("model.pretrained")
    @param("training.distributed")
    @param("training.use_blurpool")
    @param("training.dropout2d")
    @param("training.dropout1d")
    def create_model_and_scaler(
        self, arch, pretrained, distributed, use_blurpool, dropout2d, dropout1d
    ):
        scaler = GradScaler()
        if arch == "linear":
            model = ch.nn.Sequential(ch.nn.Flatten(), ch.nn.Linear(3 * 256 * 256, 1000))
        else:
            if "convnext" in arch:
                model = getattr(models, arch)(
                    pretrained=pretrained, stochastic_depth_prob=0.0
                )
            elif "vit" in arch:
                model = getattr(models, arch)(pretrained=pretrained)
            else:
                model = getattr(models, arch)(pretrained=pretrained)
                if use_blurpool:

                    def apply_blurpool(mod: ch.nn.Module):
                        for (name, child) in mod.named_children():
                            if isinstance(child, ch.nn.Conv2d) and (
                                np.max(child.stride) > 1 and child.in_channels >= 16
                            ):
                                setattr(mod, name, BlurPoolConv2d(child))
                            else:
                                apply_blurpool(child)

                    apply_blurpool(model)
        if dropout2d > 0:

            def apply_dropout(mod: ch.nn.Module):
                for (name, child) in mod.named_children():
                    if isinstance(child, ch.nn.ReLU):
                        print("applying dropout2d to", name, child)
                        setattr(
                            mod,
                            name,
                            ch.nn.Sequential(child, ch.nn.Dropout2d(dropout2d / 100)),
                        )
                    else:
                        apply_dropout(child)

            apply_dropout(model)
        if dropout1d > 0:
            def apply_dropout(mod: ch.nn.Module):
                for (name, child) in mod.named_children():
                    if isinstance(child, ch.nn.ReLU):
                        print("applying dropout1d to", name, child)
                        setattr(
                            mod,
                            name,
                            ch.nn.Sequential(child, ch.nn.Dropout(dropout1d / 100)),
                        )
                    else:
                        apply_dropout(child)

            apply_dropout(model)

        model = model.to(memory_format=ch.channels_last)
        model = model.to(self.gpu)

        if distributed:
            model = ch.nn.parallel.DistributedDataParallel(model, device_ids=[self.gpu])

        return model, scaler

    @param("logging.log_level")
    def train_loop(self, epoch, log_level):
        if self.loss_type in ['CLAM', 'tce', 'apstar']:
            # save weights
            self.weights_df.loc[len(self.weights_df.index)] = np.concatenate([[epoch], [self.class_weights[_] for _ in self.class_weights]])
            self.weights_df.to_csv(os.path.join(self.log_folder,'weights.csv'))

        model = self.model
        model.train()
        losses = []

        lr_start, lr_end = self.get_lr(epoch), self.get_lr(epoch + 1)
        iters = len(self.train_loader)
        lrs = np.interp(np.arange(iters), [0, iters], [lr_start, lr_end])

        iterator = tqdm(self.train_loader)
        for ix, (images, targets) in enumerate(iterator):
            ### Training start
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lrs[ix]
            
            self.optimizer.zero_grad(set_to_none=True)
            with autocast():
                output = self.model(images)
                if (self.loss_type == 'CLAM' and epoch >= self.CLAM_start_epoch) or self.loss_type in ['tce', 'GGF', 'apstar']:
                    # CLAM loss or tce loss or GGF loss or apstar loss
                    loss = self.sample_loss(output, targets)
                    
                    targets_numpy = targets.cpu().numpy()
                    batch_class_weights = np.array([self.class_weights[target] for target in targets_numpy])
                    batch_class_weights = ch.tensor(batch_class_weights).reshape(1,-1).to(self.gpu)

                    loss_train = ch.mean(batch_class_weights * loss)
                
                elif self.loss_type in ['focal', 'pw']:
                    # focal loss or pw loss
                    # probs: batch_size * num_classes
                    softmax=nn.Softmax(dim=1)    
                    probs = softmax(output)
                    
                    if self.loss_type == 'focal':
                        loss_train = Focal_Loss(probs, targets, self.gamma)
                    if self.loss_type == 'pw':
                        loss_train = Pw_Loss(probs, targets, self.theta, self.gamma)
                    
                else:
                    # normal_loss
                    loss_train = self.loss(output, targets)
            if ix % 100 == 0: print('iter {} lr {} loss_train {}'.format(ix, lrs[ix], loss_train))
            
            self.scaler.scale(loss_train).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            ### Training end

            ### Logging start
            if log_level > 0:
                losses.append(loss_train.detach())

                group_lrs = []
                for _, group in enumerate(self.optimizer.param_groups):
                    group_lrs.append(f'{group["lr"]:.3f}')

                names = ["ep", "iter", "shape", "lrs"]
                values = [epoch, ix, tuple(images.shape), group_lrs]
                if log_level > 1:
                    names += ["loss"]
                    values += [f"{loss_train.item():.3f}"]

                msg = ", ".join(f"{n}={v}" for n, v in zip(names, values))
                iterator.set_description(msg)
            ### Logging end
        
        if (self.loss_type == 'CLAM' and epoch >= self.CLAM_start_epoch) or self.loss_type in ['tce', 'apstar', 'GGF']:
            self.update_class_weights()

    def update_class_weights(self):
        with ch.no_grad():
            model = self.model
            model.eval()

            total_loss_per_class = dict()
            for _ in range(self.num_classes):
                total_loss_per_class[_] = []

            label_accuracies = np.zeros(self.num_classes)
            label_nums = np.zeros(self.num_classes)
            correct, total = 0, 0

            iterator = tqdm(self.train_loader)
            for ix, (images, targets) in enumerate(iterator):
                iter_start_time = time.time()
                targets_numpy = targets.cpu().numpy()
                
                with autocast():
                    output = self.model(images)
                    loss = self.sample_loss(output, targets)
                    loss_numpy = loss.detach().cpu().numpy()
                    for i_sample in range(len(targets_numpy)):
                        total_loss_per_class[targets_numpy[i_sample]].append(loss_numpy[i_sample])

                # train_acc per class
                _, predicted = ch.max(output.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum()
                correct_prediction_flag = predicted == targets
                for j in range(len(correct_prediction_flag.cpu().numpy())):
                    label = targets[j].cpu().numpy()
                    label_nums[label] += 1
                    if correct_prediction_flag[j]:
                        label_accuracies[label] += 1

            # train loss per class
            for label in total_loss_per_class:
                total_loss_per_class[label] = np.mean(total_loss_per_class[label])
            print('trainloss_per_class', total_loss_per_class)

            sorted_total_loss_per_class = sorted(total_loss_per_class.items(), key=lambda x:x[1], reverse=True)
            class_idx_order = [_[0] for _ in sorted_total_loss_per_class]
            print('class_idx_order', class_idx_order)

            if self.loss_type == 'CLAM':
                # weight by trainacc
                # update self.class_weights by training accuracy
                print('label_accuracies/label_nums', label_accuracies/label_nums)
                diff_weights = np.exp(-label_accuracies / label_nums) / np.sum(np.exp(- label_accuracies / label_nums))
                diff_weights = diff_weights / np.mean(diff_weights)
                print('diff_weights', diff_weights)

                for class_id in range(self.num_classes):
                    tmp_weight = self.class_weights[class_id]
                    tmp_weight = np.clip(tmp_weight * diff_weights[class_id], 0.5, 2.0)
                    self.class_weights[class_id] = tmp_weight

                # normalize the weights
                mean_weight = sum(self.class_weights.values()) / len(self.class_weights)
                for _ in self.class_weights:
                    self.class_weights[_] /= mean_weight

            if self.loss_type == 'GGF':
                weights = np.array([max(self.discount**_,self.min_weight) for _ in range(len(total_loss_per_class))])
                weights = weights / np.mean(weights)
                for _ in range(len(total_loss_per_class)):
                    self.class_weights[sorted_total_loss_per_class[_][0]] = weights[_]
                
            if self.loss_type == 'tce':
                weights = np.exp(list(total_loss_per_class.values())) / np.sum(np.exp(list(total_loss_per_class.values())))
                weights = weights / np.mean(weights)
                for label in total_loss_per_class:
                    self.class_weights[label] = 0.5*self.class_weights[label] + 0.5*weights[label]

            if self.loss_type == 'apstar':
                for _ in range(len(total_loss_per_class)):
                    label = sorted_total_loss_per_class[_][0]
                    num_worst_classes = int(self.num_classes * 0.1)
                    if _ < num_worst_classes:
                        # increase the weights for the worst classes
                        # the value of the identity vector is divided by num_worst classes, then multiplied by num_classes (sum of weights is $n$ in our case)
                        identity_vector_value = self.num_classes / num_worst_classes
                        self.class_weights[label] = self.class_weights[label] * self.alpha + identity_vector_value / self.K * (1 - self.alpha)
                    else:
                        self.class_weights[label] = self.class_weights[label] * self.alpha

                # clip to avoid extreme weights 
                # unnecessary in group fairness with limited number of groups
                # but crucial in class fairness
                for _ in range(self.num_classes):
                    self.class_weights[_] = np.clip(self.class_weights[_], 0.5, 2.0)
                
                # normalize the weights
                for _ in range(self.num_classes):
                    self.class_weights[_] = self.class_weights[_] * num_classes / sum(self.class_weights.values())
                
                # if max loss decreases, set K to K_min
                if sorted_total_loss_per_class[0][1] < self.apstar_max_loss:
                    print('worst_class_loss', sorted_total_loss_per_class[0][1])
                    self.apstar_max_loss = sorted_total_loss_per_class[0][1]
                    self.K = self.K_min
                else:
                    self.K += 1
                
            print('self.class_weights', self.class_weights)

    @param("validation.lr_tta")
    def val_loop(self, lr_tta):
        model = self.model
        model.eval()

        with ch.no_grad():
            with autocast():
                for images, targets in tqdm(self.val_loader):
                    output = model(images)
                    if lr_tta:
                        output += self.model(ch.flip(images, dims=[3]))

                    for k in ["top_1", "top_5", "per_class"]:
                        self.val_meters[k](output, targets)

                    loss_val = self.loss(output, targets)
                    self.val_meters["loss"](loss_val)

        stats = {
            k: m.compute().item() if k != "per_class" else m.compute().tolist()
            for k, m in self.val_meters.items()
        }
        [meter.reset() for meter in self.val_meters.values()]
        return stats

    @param("logging.folder")
    def initialize_logger(self, folder):
        self.val_meters = {
            "top_1": torchmetrics.Accuracy(task='multiclass', num_classes=1000).to(self.gpu),
            "top_5": torchmetrics.Accuracy(task='multiclass', num_classes=1000, top_k=5).to(self.gpu),
            "per_class": torchmetrics.Accuracy(task='multiclass', num_classes=1000, top_k=1, average=None).to(self.gpu),
            "loss": MeanScalarMetric().to(self.gpu),
        }

        if self.gpu == 0:
            #folder = (Path(folder) / str(self.uid)).absolute()
            folder = Path(folder) / str(self.all_params['training.loss_type']) / self.exp_type
            if not os.path.exists(folder):
                os.makedirs(folder)
            
            self.log_folder = folder
            self.eval_df = self.create_eval_csv(num_classes=self.num_classes)
            self.eval_df.to_csv(os.path.join(self.log_folder,'eval.csv'))
            self.weights_df = self.create_weights_csv(num_classes=self.num_classes)
            self.start_time = time.time()

            print(f"=> Logging in {self.log_folder}")
            params = {
                ".".join(k): self.all_params[k] for k in self.all_params.entries.keys()
            }

            with open(folder / "params.json", "w+") as handle:
                json.dump(params, handle)

    def create_eval_csv(self, num_classes=1000):
        cols = list(range(num_classes))
        cols.insert(0, 'epoch')
        cols.append('average')
        
        eval_df = pd.DataFrame(columns=cols)

        return eval_df

    def create_weights_csv(self, num_classes=1000):
        cols = list(range(num_classes))
        cols.insert(0, 'epoch')
        
        weights_df = pd.DataFrame(columns=cols)

        return weights_df

    def log(self, content):
        print(f"=> Log: {content}")
        if self.gpu != 0:
            return
        cur_time = time.time()
        bucket = {
            "timestamp": cur_time,
            "relative_time": cur_time - self.start_time,
            **content,
        }
        with open(self.log_folder / "log", "a+") as fd:
            fd.write(json.dumps(bucket) + "\n")
            fd.flush()

    @classmethod
    @param("training.distributed")
    @param("dist.world_size")
    def launch_from_args(cls, distributed, world_size):
        if distributed:
            ch.multiprocessing.spawn(cls._exec_wrapper, nprocs=world_size, join=True)
        else:
            cls.exec(0)

    @classmethod
    def _exec_wrapper(cls, *args, **kwargs):
        make_config(quiet=True)
        cls.exec(*args, **kwargs)

    @classmethod
    @param("training.distributed")
    @param("training.eval_only")
    def exec(cls, gpu, distributed, eval_only):
        trainer = cls(gpu=gpu)
        if eval_only:
            trainer.eval_and_log()
        else:
            trainer.train()

        if distributed:
            trainer.cleanup_distributed()


# Utils
class MeanScalarMetric(torchmetrics.Metric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.add_state("sum", default=ch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=ch.tensor(0), dist_reduce_fx="sum")

    def update(self, sample: ch.Tensor):
        self.sum += sample.sum()
        self.count += sample.numel()

    def compute(self):
        return self.sum.float() / self.count


# Running
def make_config(quiet=False):
    config = get_current_config()
    parser = ArgumentParser(description="Fast imagenet training")
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode="stderr")
    if not quiet:
        config.summary()


if __name__ == "__main__":
    make_config()
    ImageNetTrainer.launch_from_args()
