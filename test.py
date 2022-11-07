import os
import gc
import time
import util
import torch
import argparse
import sampler
import dataset
import transform
import config as cfg

from tqdm import tqdm
from fcos import FCOS
from dataloader import build_dataloader

def test(epochs_tested):
    is_train=False
    transforms = transform.build_transforms(is_train=is_train)
    coco_dataset = dataset.COCODataset(is_train=is_train, transforms=transforms)
    dataloader = build_dataloader(coco_dataset, sampler=None, is_train=is_train)

    assert isinstance(epochs_tested, (list, set)), "during test, archive_name must be a list or set!"
    model = FCOS(is_train=is_train)

    for epoch in epochs_tested:
        util.load_model(model, epoch)
        model.cuda()
        model.eval()

        final_results = []

        with torch.no_grad():
            for data in tqdm(dataloader):
                img = data["images"]                 #torch.Size([1, 3, 1152, 800])
                ori_img_shape = data["ori_img_shape"] #tensor([[500, 353]])
                fin_img_shape = data["fin_img_shape"] #tensor([[1133,  800]])
                index = data["indexs"] #0

                img = img.cuda()
                ori_img_shape = ori_img_shape.cuda()
                fin_img_shape = fin_img_shape.cuda()

                cls_pred, reg_pred, label_pred = model([img, ori_img_shape, fin_img_shape])

                cls_pred = cls_pred[0].cpu() #100
                reg_pred = reg_pred[0].cpu() #100 4
                label_pred = label_pred[0].cpu() #100
                index = index[0] #

                img_info = dataloader.dataset.img_infos[index]
                imgid = img_info["id"] #20180000001

                reg_pred = util.xyxy2xywh(reg_pred)  #xyxy torch.Size([100, 4])  转为 xywh

                label_pred = label_pred.tolist() #list 100
                cls_pred = cls_pred.tolist() #list  分数
                # print(len(reg_pred))
                # for k in range(len(reg_pred)):
                #    print(k,k)
                #    print( label_pred[k])
                final_results.extend(
                    [
                        {
                            "image_id": imgid,
                            "category_id": dataloader.dataset.label2catid[label_pred[k]],
                            "bbox": reg_pred[k].tolist(),
                            "score": cls_pred[k],
                        }
                        for k in range(len(reg_pred))
                    ]
                )


        output_path = os.path.join(cfg.output_path, "fcos_"+str(epoch)+".json")
        util.evaluate_coco(dataloader.dataset.coco, final_results, output_path, "bbox")

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    epochs_tested=[3,12]
    util.mkdir(cfg.output_path)
    test(epochs_tested)

if __name__ == "__main__":
    main()