import os
import time
import util
import torch
import argparse
import dataset
import transform
import config as cfg
import torch.distributed as dist

import solver

from sampler import groupSampler
from sampler import distributedGroupSampler
from fcos import FCOS
from dataloader import build_dataloader

pretrained_path={
    50:"./pretrained/resnet50_caffe.pth",
    101:"./pretrained/resnet101_caffe.pth"
}

def train(is_dist,start_epoch,local_rank):
    transforms=transform.build_transforms()
    coco_dataset = dataset.COCODataset(is_train=True, transforms=transforms)
    if(is_dist):
        sampler = distributedGroupSampler(coco_dataset)
    else:
        sampler = groupSampler(coco_dataset)
    dataloader = build_dataloader(coco_dataset, sampler)

    batch_time_meter = util.AverageMeter()
    iou_qflloss_meter = util.AverageMeter()
    reg_loss_meter = util.AverageMeter()
    cen_loss_meter = util.AverageMeter()
    cls_loss_meter=util.AverageMeter()
    losses_meter = util.AverageMeter()

    model = FCOS(is_train=True)
    if (start_epoch == 1):
        model.resNet.load_pretrained(pretrained_path[cfg.resnet_depth])
    else:
        util.load_model(model, start_epoch - 1)
    model=model.cuda()

    if is_dist:
        model=torch.nn.parallel.DistributedDataParallel(model,device_ids=[local_rank,],output_device=local_rank,broadcast_buffers=False)
    optimizer=solver.build_optimizer(model)
    scheduler=solver.scheduler(optimizer)

    model.train()
    logs = []

    for epoch in range(1, cfg.max_epochs + 1):
        if is_dist:
            dataloader.sampler.set_epoch(epoch-1)
        scheduler.lr_decay(epoch)

        end_time = time.time()
        for iteration, datas in enumerate(dataloader, 1):
            scheduler.constant_warmup(epoch, iteration - 1)
            images = datas["images"] #torch.Size([2, 3, 800, 1088])
            bboxes = datas["bboxes"] # 14 84
            labels = datas["labels"] #[tensor([4]), tensor([6, 7, 4, 4, 4, 4, 4, 4])]

            images = images.cuda()
            bboxes = [bbox.cuda() for bbox in bboxes]
            labels = [label.cuda() for label in labels]

            loss_dict = model([images, bboxes, labels])  # ([2, 3, 608, 928])  【3 4】 【1，4】tensor([ 0,  0, 31], device='cuda:0') 0
            iou_qflloss = loss_dict["iou_qflloss"]
            reg_loss = loss_dict["reg_loss"]
            cen_loss = loss_dict["cen_loss"]
            cls_loss = loss_dict["cls_loss"] ########



            losses = iou_qflloss + reg_loss + cen_loss+cls_loss #####

            optimizer.zero_grad()
            losses.backward()
            grad_norm=util.clip_grads(model.parameters())
            optimizer.step()

            batch_time_meter.update(time.time()-end_time)
            end_time = time.time()

            iou_qflloss_meter.update(iou_qflloss.item())
            reg_loss_meter.update(reg_loss.item())
            cen_loss_meter.update(cen_loss.item())
            cls_loss_meter.update(cls_loss.item())  ##########

            losses_meter.update(losses.item())

            if(iteration % 2 == 0):
                if(local_rank == 0):

                    res = "\t".join([
                        "Epoch: [%d/%d]" % (epoch,cfg.max_epochs),
                        "Iter: [%d/%d]" % (iteration, len(dataloader)),
                        "Time: %.3f (%.3f)" % (batch_time_meter.val, batch_time_meter.avg),
                        "iou_qflloss: %.4f (%.4f)" % (iou_qflloss_meter.val, iou_qflloss_meter.avg),
                        "Reg_loss: %.4f (%.4f)" % (reg_loss_meter.val, reg_loss_meter.avg),
                        "Cen_loss: %.4f (%.4f)" % (cen_loss_meter.val, cen_loss_meter.avg),
                        "cls_loss: %.4f (%.4f)" % (cls_loss_meter.val, cls_loss_meter.avg),#####

                        "Loss: %.4f (%.4f)" % (losses_meter.val, losses_meter.avg),
                        "lr: %.6f" % (optimizer.param_groups[0]["lr"]),
                        "grad_norm: %.4f" % (grad_norm.item(),)
                    ])
                    print(res)
                    logs.append(res)
                batch_time_meter.reset()
                iou_qflloss_meter.reset()
                reg_loss_meter.reset()
                cen_loss_meter.reset()
                cls_loss_meter.reset()  #######

                losses_meter.reset()

        if (local_rank == 0):
            util.save_model(model, epoch)
        if (is_dist):
            util.synchronize()

    if (local_rank == 0):
        with open("logs.txt", "w") as f:
            for i in logs:
                f.write(i + "\n")

def main():
    parser=argparse.ArgumentParser(description="FCOS")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--start_epoch", type=int, default=1)
    parser.add_argument("--dist",action="store_true")

    args=parser.parse_args()
    if(args.dist):
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        util.synchronize()

    util.init_seeds(0)
    train(args.dist,args.start_epoch,args.local_rank)

if __name__=="__main__":
    main()