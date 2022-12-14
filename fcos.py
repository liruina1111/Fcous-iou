import torch
import math
import config as cfg
import torch.nn.functional as F
from resnet import resNet
from fpn import PyramidFeatures
from loss import FCOSLoss
from inference import FCOSInference

class scale(torch.nn.Module):
    def __init__(self):
        super(scale, self).__init__()    # 1.0  =  cfg.scale_init_value
        self.scale=torch.nn.Parameter(torch.FloatTensor([cfg.scale_init_value])) #Parameter containing:tensor([1.], requires_grad=True)

    def forward(self,input):
        return input*self.scale
# class iou(torch.nn.Module):
#     def __init__(self):
#         super(iou, self).__init__()
#
#         self.conv1 = torch.nn.Conv2d(cfg.fpn_channels, cfg.fpn_channels, kernel_size=3, padding=1)
#         self.conv2 = torch.nn.Conv2d(cfg.fpn_channels, cfg.fpn_channels, kernel_size=3, padding=1)
#         self.conv3 = torch.nn.Conv2d(cfg.fpn_channels, cfg.fpn_channels, kernel_size=3, padding=1)
#         self.conv4 = torch.nn.Conv2d(cfg.fpn_channels, cfg.fpn_channels, kernel_size=3, padding=1)
#
#
#         self.iou = torch.nn.Conv2d(cfg.fpn_channels, 1, kernel_size=3, padding=1)
#
#         for m in self.modules():
#             if(isinstance(m, torch.nn.Conv2d)):
#                 torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
#                 torch.nn.init.constant_(m.bias, 0)



class classification(torch.nn.Module):
    def __init__(self):
        super(classification, self).__init__()

        self.conv1 = torch.nn.Conv2d(cfg.fpn_channels, cfg.fpn_channels, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(cfg.fpn_channels, cfg.fpn_channels, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(cfg.fpn_channels, cfg.fpn_channels, kernel_size=3, padding=1)
        self.conv4 = torch.nn.Conv2d(cfg.fpn_channels, cfg.fpn_channels, kernel_size=3, padding=1)

        self.output = torch.nn.Conv2d(cfg.fpn_channels, cfg.num_classes, kernel_size=3, padding=1) #分类
        self.centerness = torch.nn.Conv2d(cfg.fpn_channels, 1, kernel_size=3, padding=1)
        ### 定义iou分支
        self.iou= torch.nn.Conv2d(cfg.fpn_channels, 1, kernel_size=3, padding=1)
        for m in self.modules():
            if(isinstance(m, torch.nn.Conv2d)):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
                torch.nn.init.constant_(m.bias, 0)

        bias_value = -math.log((1-cfg.class_prior_prob)/cfg.class_prior_prob) #-4.59511985013459分类分支，分类预测的初始得分
        torch.nn.init.constant_(self.output.bias, bias_value)

    def forward(self, x):
        cls_preds=[]
        cen_preds=[]
        iou_preds = []
        for x_per_level in x:
            x_per_level = self.conv1(x_per_level)
            x_per_level.relu_()

            x_per_level = self.conv2(x_per_level)
            x_per_level.relu_()

            x_per_level = self.conv3(x_per_level)
            x_per_level.relu_()

            x_per_level = self.conv4(x_per_level)
            x_per_level.relu_()

            logits = self.output(x_per_level) #torch.Size([2, 80, 28, 28])
            centerness = self.centerness(x_per_level)
            cls_preds.append(logits)
            cen_preds.append(centerness)

###### 加入iou分支
            iou = self.iou(x_per_level)
            iou_preds.append(iou)
        return cls_preds, cen_preds,iou_preds

class regression(torch.nn.Module):
    def __init__(self):
        super(regression, self).__init__()

        self.conv1 = torch.nn.Conv2d(cfg.fpn_channels, cfg.fpn_channels, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(cfg.fpn_channels, cfg.fpn_channels, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(cfg.fpn_channels, cfg.fpn_channels, kernel_size=3, padding=1)
        self.conv4 = torch.nn.Conv2d(cfg.fpn_channels, cfg.fpn_channels, kernel_size=3, padding=1)

        self.output = torch.nn.Conv2d(cfg.fpn_channels, 4, kernel_size=3,padding=1)
        self.scales = torch.nn.ModuleList([scale() for _ in range(5)])

        for m in self.modules():
            if(isinstance(m, torch.nn.Conv2d)):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
                torch.nn.init.constant_(m.bias,0)

    def forward(self, x):
        reg_preds=[]
        for i, x_per_level in enumerate(x):
            x_per_level = self.conv1(x_per_level)
            x_per_level.relu_()

            x_per_level = self.conv2(x_per_level)
            x_per_level.relu_()

            x_per_level = self.conv3(x_per_level)
            x_per_level.relu_()

            x_per_level = self.conv4(x_per_level)
            x_per_level.relu_()

            x_per_level=self.output(x_per_level)
            x_per_level=self.scales[i](x_per_level)
            x_per_level=torch.exp(x_per_level)
            reg_preds.append(x_per_level)

        return reg_preds

class FCOS(torch.nn.Module):
    def __init__(self, is_train=True):
        super(FCOS, self).__init__()
        self.is_train = is_train

        self.resNet = resNet()
        self.fpn = PyramidFeatures()

        self.regression = regression()
        self.classification = classification()
        # self.iou = iou()
        self.loss = FCOSLoss()
        self.inference = FCOSInference()

    def forward(self, input):
        if self.is_train:
            img_batch, gt_bboxes, gt_labels = input
        else:
            img_batch, ori_img_shape, fin_img_shape = input

        c3, c4, c5=self.resNet(img_batch)
        features = self.fpn([c3, c4, c5])

        reg_preds = self.regression(features) #torch.Size([1, 4, 144, 100]) torch.Size([1, 4, 72, 50]) torch.Size([1, 4, 36, 25])。。

        cls_preds, cen_preds,iou_preds = self.classification(features) #
        # iou_preds = self.iou(features)

        if self.is_train:
            losses = self.loss(cls_preds, reg_preds, cen_preds, iou_preds,gt_bboxes, gt_labels)
            return losses
        else:
            result = self.inference(cls_preds, reg_preds, cen_preds, iou_preds,ori_img_shape, fin_img_shape)
            return result
if __name__ == '__main__':
    x3 = torch.rand(2, 256, 152, 100)
    x4 = torch.rand(2, 256, 76, 50)
    x5 = torch.rand(2, 256, 38, 25)
    x6 = torch.rand(2, 256, 19, 13)
    x7 = torch.rand(2, 256, 10, 7)
    x=[x3,x4,x5,x6,x7]
    model=classification()
    out=model(x)
