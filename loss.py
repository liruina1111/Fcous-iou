import torch
import util
import config as cfg
import torch.nn.functional as F

INF = 100000000

class SigmoidFocalLoss(torch.nn.Module):
    def __init__(self):
        super(SigmoidFocalLoss, self).__init__()
        self.gamma = cfg.focal_loss_gamma
        self.alpha = cfg.focal_loss_alpha
    def forward(self, logits, targets):
        """
        计算分类分支的loss, 即focal loss.
        :param logits:  ([36268, 200])    神经网络分类分支的输出. type为tensor, shape为(cumsum_5(N*ni),80),其中N是batch size, ni为第i层feature map的样本数量
        :param targets: 36268            表示分类分之的targer labels, type为tensor, shape为(cumsum_5(N*ni),), 其中N是batch size, 正样本的label介于[0,79], 负样本的label为-1
        :return loss: 所有anchor point的loss之和.
        """
        device = targets.device
        dtype = targets.dtype

        num_classes = logits.shape[1]
        class_range = torch.arange(0, num_classes, dtype=dtype, device=device).unsqueeze(0)  #torch.Size([1, 20])  0，1，2，3，4，5，6，，，79

        t = targets.unsqueeze(1)
        p = torch.sigmoid(logits)

        term1 = (1-p)**self.gamma*torch.log(p)
        term2 = p**self.gamma*torch.log(1-p)
        #  t == class_range 正阳本*（1-p）^g*log(p)）*0.25     (t != class_range）抚养本* （p）^2*log(1-p)）*0.75使用了focal loss之后，很多easy negative的权重被降的很低，而hard positive的数量比hard negative数量多
        loss = -(t == class_range).float()*term1*self.alpha - ((t != class_range)*(t >= 0)).float()*term2*(1-self.alpha) #torch.Size([36268, 20])

        return loss.sum()

class IouLoss(torch.nn.Module):
    def __init__(self):
        super(IouLoss, self).__init__()
        self.eps = 1e-6

    def forward(self, bbox_pred_pos, regress_target_pos, weight_pos):
        """
        计算为回归任务计算iou loss
        :param bbox_pred_pos: 回归任务预测正样本框的坐标，格式为ltrb, shape为[N, 4] torch.Size([109, 4])
        :param regress_target_pos: 正样本框的gt坐标，格式为ltrb, shape为[N,    4 torch.Size([109, 4])
        :param weight_pos: 默认采用centerness为每个正样本的回归loss加权。          109
        :return:
        """
        iou = self.compute_iou_ltrb(bbox_pred_pos, regress_target_pos) #101  torch.Size([101, 4]) torch.Size([101, 4])
        avg_factor = weight_pos.sum()

        loss = -iou.log()
        loss = (loss*weight_pos).sum() / avg_factor

        return loss

    def compute_iou_ltrb(self, fbbox, lbbox):
        """
        给定ltrb格式的框，计算其iou，需要保证fbbox和lbbox是一一对应的，即中心点一一对应
        :param fbbox: shape为lrtb torch.Size([101, 4])
        :param lbbox: shape为lrtb torch.Size([101, 4])
        :return:
        """
        f_left = fbbox[:, 0]
        f_top = fbbox[:, 1]
        f_right = fbbox[:, 2]
        f_bottem = fbbox[:, 3]

        l_left = lbbox[:, 0]
        l_top = lbbox[:, 1]
        l_right = lbbox[:, 2]
        l_bottem = lbbox[:, 3]

        f_area = (f_left + f_right) * (f_top + f_bottem)
        l_area = (l_left + l_right) * (l_top + l_bottem)

        w_intersect = torch.min(f_left, l_left) + torch.min(f_right, l_right)
        h_intersect = torch.min(f_top, l_top) + torch.min(f_bottem, l_bottem)

        area_intersect = w_intersect * h_intersect       #交集得面积
        area_union = f_area + l_area - area_intersect    #并集得面积
        area_union.clamp(min=self.eps)
        ious = area_intersect / area_union

        return ious

class CenternessLoss(torch.nn.Module):
    def __init__(self):
        super(CenternessLoss, self).__init__()

    def forward(self, centerness_pred, centerness_target):  #109 109
        loss = F.binary_cross_entropy_with_logits(centerness_pred, centerness_target,reduction='none')
        return loss.mean()

#####  计算iou分支损失
class QFLLoss(torch.nn.Module):
    def __init__(self):
        super(QFLLoss, self).__init__()
        self.weight=cfg.qfl_loss_weight
        self.beta=cfg.qfl_loss_beta

    def forward(self,preds,targets,avg_factor=None):
        loss=F.binary_cross_entropy_with_logits(preds,targets,reduction='none')

        preds_sigmoid=preds.sigmoid()
        scale_factor=(preds_sigmoid-targets).abs().pow(self.beta)
        loss=loss*scale_factor
        loss=loss.sum()*self.weight

        if(not avg_factor is None):
           return loss/avg_factor

        return loss
class FCOSLoss(torch.nn.Module):
    def __init__(self):
        super(FCOSLoss, self).__init__()
        self.cls_loss_func = SigmoidFocalLoss()
        self.reg_loss_func = IouLoss()
        self.cen_loss_func = CenternessLoss()
        self.qfl_loss_func = QFLLoss()     ###  定义iou分支的loss
        self.eps = 1e-6

    def compute_targets(self, scales, gt_bboxes, gt_labels):
        """
        计算分类、回归和中心度的target
        :param scales: feature map的尺度，[torch.Size([100, 136]), torch.Size([50, 68]), torch.Size([25, 34]), torch.Size([13, 17]), torch.Size([7, 9])]
        :param gt_bboxes: type为list       ([2, 4])    tore([1, 4])
        :param gt_labels: type为list     tensor([3, 4],   【1='cuda:0')]
        :return:
        """
        #print(gt_labels)
        dtype = gt_bboxes[0].dtype
        device = gt_bboxes[0].device
        num=0
        points = util.compute_points(scales, device, dtype) # 需要获取预设anchor point  (e([3600, 2])(([900, 2]))([850, 2])([225, 2]))t([64, 2])([16, 2])
        num_points_per_level = [point.shape[0] for point in points] #每一层样本点的数量)[13600, 3400, 850, 221, 63]
        num_levels = len(points)

        cls_labels=[]
        reg_targets=[]
        ranges_all_level=[]
        for i, range_per_level in enumerate(cfg.regress_ranges):
            ranges_all_level.append(torch.tensor(range_per_level, dtype=dtype, device=device).expand_as(points[i]))  #总共有([13600, 个【-1，64】  有·[3400, 2]个[64, 128]     [850, 2]个[128, 256]

        #ranges_all_level为每个样本点的边界距离约束
        ranges_all_level = torch.cat(ranges_all_level) #
        points = torch.cat(points) #
        points = points[:, None, :]

        # 对每张图片分别计算ground truth    gt_bboxes2个list 2张图片
        for i, gt_bbox in enumerate(gt_bboxes): #
            gt_label = gt_labels[i]
            areas = util.compute_areas(gt_bbox)
            areas = areas[None].expand(points.shape[0], gt_bbox.shape[0])
            areas = areas.contiguous()

            gt_bbox = gt_bbox[None, :, :]

            #使用广播技巧，来计算每个样本点的四个边界距离 points=tze(([4805, 1, 2])  gt_bbox=([1, 5, 4])
            distance_left = points[..., 0] - gt_bbox[..., 0]
            distance_top = points[..., 1] - gt_bbox[..., 1]
            distance_right = gt_bbox[..., 2] - points[..., 0]
            distance_bottem = gt_bbox[..., 3] - points[..., 1]

            distance = torch.stack([distance_left, distance_top, distance_right, distance_bottem], dim=-1) #(torch.Size([3525, 11, 4]))
            min_distance, _ = distance.min(dim=-1)
            max_distance, _ = distance.max(dim=-1)

            #正样本标准1：位于gt bbox内部
            invalid1 = min_distance > 0

            #正样本标准2：到边框的最大距离不超过预设范围 ranges_all_level=t([18134, 2]) 就是（-1 64）.。。
            min_distance_range = ranges_all_level[:, [0]]
            max_distance_range = ranges_all_level[:, [1]]
            invalid2 = (max_distance > min_distance_range) & (max_distance < max_distance_range) #((([4805, 5])

            areas[invalid1==0] = INF
            areas[invalid2==0] = INF

            min_area, min_area_inds = areas.min(dim=1)
            cls_label = gt_label[min_area_inds]
            cls_label[min_area == INF] = cfg.num_classes#

            reg_target = distance[range(points.shape[0]), min_area_inds]
            cls_labels.append(cls_label)
            reg_targets.append(reg_target)

        #注意到此时class_labels和regress_targets都是图片优先的，需要转化为level优先
        cls_labels=[cls_label.split(num_points_per_level,dim=0) for cls_label in cls_labels]
        reg_targets=[reg_target.split(num_points_per_level,dim=0) for reg_target in reg_targets]

        cls_labels_level_first=[]
        reg_targets_level_first=[]
        for i in range(num_levels):
            cls_labels_level_first.append(torch.cat([cls_label[i] for cls_label in cls_labels]))
            reg_targets_level_first.append(torch.cat([reg_target[i] for reg_target in reg_targets]))

        return cls_labels_level_first, reg_targets_level_first

    def compute_centerness_targets(self, reg_targets):
        """
        计算centerness分支的target
        :param regress_targets: shape为[N,4], 格式为ltrb  # ([583, 4])
        :return:
        """
        left_right = reg_targets[:,[0,2]]  #距离左边的距离 距离右边的距离([583, 2])
        top_bottem = reg_targets[:,[1,3]]  #距离上边的距离 距离下边的juli torch.Size([583, 2])
        centerness_targets = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0])*\
                             (top_bottem.min(dim=-1)[0] / top_bottem.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)

    def forward(self, cls_preds, reg_preds, cen_preds, iou_preds,gt_bboxes, gt_labels):
        """
        计算loss
        :param cls_preds: 分类分支的输出 （[2, 20, 152, 100])  ([2, 20, 50, 68])  ([2, 20, 25, 34])  ([2, 80, 13, 17])    ([2, 80, 7，9)
        :param reg_preds: 回归分支的输出  ([2, 4, 152, 100])  ([2, 4, 50, 68])  ([2, 4, 25, 34])  ([2, 4, 13, 17])    ([2, 4, 7, 9])
        :param cen_preds: 中心分支的输出  ([2, 1, 152, 100]) ([2, 1, 50, 68])  ([2, 1, 25, 34])  ([2, 1, 13, 17])    ([2, 1, 7  9】
        :param gt_bboxes: 2list ([1, 4]) ([2, 4])
        :param gt_labels: tensor[2] ([0, 1],
        :return:
        """
        scales=[cls_pred.shape[2:] for cls_pred in cls_preds] #[[torch.Size([152, 100]), torch.Size([76, 50]), torch.Size([38, 25]), torch.Size([19, 13]), torch.Size([10, 7])], 9])]

        cls_labels_level_first, reg_targets_level_first= self.compute_targets(scales, gt_bboxes, gt_labels)

        cls_labels_all = torch.cat(cls_labels_level_first)
        reg_targets_all = torch.cat(reg_targets_level_first)


        cls_preds = [cls_pred.permute(0,2,3,1).reshape(-1, cfg.num_classes) for cls_pred in cls_preds]
        reg_preds = [reg_pred.permute(0,2,3,1).reshape(-1,4) for reg_pred in reg_preds]
        cen_preds = [cen_pred.permute(0,2,3,1).reshape(-1) for cen_pred in cen_preds]
        iou_preds = [iou_pred.permute(0, 2, 3, 1).reshape(-1) for iou_pred in iou_preds]

        cls_preds_all = torch.cat(cls_preds)  #([12170, 80])
        reg_preds_all = torch.cat(reg_preds)  #([12170, 4])
        cen_preds_all = torch.cat(cen_preds)  # 12170
        iou_preds_all = torch.cat(iou_preds)  # 12170

        pos_inds = torch.nonzero((cls_labels_all >= 0) & (cls_labels_all != cfg.num_classes), as_tuple=False).reshape(-1) #522
        num_pos = len(pos_inds)               #正样本的数量 336
        avg_factor = num_pos + cfg.samples_per_gpu


        cls_loss = self.cls_loss_func(cls_preds_all, cls_labels_all) / avg_factor


        bbox_preds_pos = reg_preds_all[pos_inds]     #torch.Size([366, 4])
        reg_targets_pos = reg_targets_all[pos_inds]  #torch.Size([366, 4])
        cen_preds_pos = cen_preds_all[pos_inds]      #366
###### 添加代码 ：  创建iou分支的target,  正样本target：正样本预测框于真实矩形框的iou取值，    负样本traget：设置为0

        iou_target_all=torch.zeros_like(iou_preds_all, device=reg_preds_all[0].device)  #12170   创建iou预测分支一样大小 全0
        pos_iou = IouLoss.compute_iou_ltrb(self, reg_preds_all[pos_inds].detach(), reg_targets_all[pos_inds]) # 计算对于正样本 366
        #pos_iou = torch.pow(pos_iou,2)
        iou_target_all[pos_inds]=pos_iou    ## 12170    pos_inds正样本索引
######
        if(num_pos > 0):
            cen_targets_pos = self.compute_centerness_targets(reg_targets_pos)
            reg_loss = self.reg_loss_func(bbox_preds_pos, reg_targets_pos, cen_targets_pos)
            cen_loss = self.cen_loss_func(cen_preds_pos,cen_targets_pos)
            iou_qflloss = self.qfl_loss_func(iou_preds_all, iou_target_all, avg_factor=num_pos)  #iou分支loss


        else:
            reg_loss = cls_loss.new_tensor(0, requires_grad=True)
            cen_loss = cls_loss.new_tensor(0, requires_grad=True)
            iou_qflloss = cls_loss.new_tensor(0, requires_grad=True)
        return dict(iou_qflloss=iou_qflloss, reg_loss=reg_loss, cen_loss=cen_loss,cls_loss= cls_loss)

