import torch
import util
import config as cfg

class FCOSInference():
    def __init__(self):
        self.num_pred_before_nms = cfg.num_pred_before_nms #每一层在nms之前，保留的候选预测框的数量
        self.pos_threshold = cfg.pos_threshold #nms之前，认为是正样本的概率值，默认为0.05
        self.nms_threshold = cfg.nms_threshold #nms过程中的阈值

    def compute_results(self, cls_preds, reg_preds, cen_preds, iou_preds,points, ori_img_shape, fin_img_shape):
        """
        对一张图片计算最终的结果，预测结果的初步筛选和nms操作
        :param cls_preds: 当前图片的分类预测, 类型为  list([1, 20, 144, 100]) ([1, 20, 72, 50])[1, 20, 36, 25])([1, 80, 18, 13])([1, 80, 9, 7])
        :param reg_preds: 当前图片的回归预测, 类型为list, 格式为ltrb, 分别表示到左,上,右,下边界的距离
        :param cen_preds: 当前图片的中心预测, 类型为list
        :param points: 根据feature map的size计算出的预设的anchor pointtorch.Size([14400, 2])torch.Size([3600, 2])torch.Size([900, 2])
        :param ori_img_shape: 输入图片的原始大小，即没有放缩之前的大小
        :param fin_img_shape: 输入图片放缩之后的大小
        :return:
        """
        result_reg=[]
        result_cls=[]
        result_cen=[]
        result_iou=[]
        #对每个特征层进行循环
        for cls_pred, reg_pred, cen_pred, iou_pred ,point in zip(cls_preds, reg_preds, cen_preds, iou_preds,points):
            cls_pred = cls_pred.permute(1, 2, 0).reshape(-1, cfg.num_classes).sigmoid()
            cen_pred = cen_pred.permute(1, 2, 0).reshape(-1).sigmoid()
            ## 获取iou预测分支
            iou_pred = iou_pred.permute(1, 2, 0).reshape(-1).sigmoid()

            reg_pred = reg_pred.permute(1, 2, 0).reshape(-1, 4)

            if(cls_pred.shape[0] > self.num_pred_before_nms):
                max_scores, _ = (cls_pred * cen_pred[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(self.num_pred_before_nms)  #

                point = point[topk_inds, :]
                reg_pred = reg_pred[topk_inds, :] #torch.Size([1000, 4])
                cls_pred = cls_pred[topk_inds, :] #torch.Size([1000, 80])
                cen_pred = cen_pred[topk_inds]    #1000
                iou_pred = iou_pred[topk_inds]    #1000

            bboxes = util.ltrb2xyxy(point, reg_pred, fin_img_shape)

            result_reg.append(bboxes)   #
            result_cls.append(cls_pred) #
            result_cen.append(cen_pred) #
            result_iou.append(iou_pred)

        result_reg = torch.cat(result_reg)
        result_cls = torch.cat(result_cls)
        result_cen = torch.cat(result_cen)
        result_iou = torch.cat(result_iou)
        #由于在训练即inference时，对原始图片做了放缩处理，因此这里需要将预测框放缩回去
        ori_img_shape = ori_img_shape.float()
        fin_img_shape = fin_img_shape.float()
        w_factor = ori_img_shape[1] / fin_img_shape[1]
        h_factor = ori_img_shape[0] / fin_img_shape[0]
        factor = result_cls.new([w_factor, h_factor, w_factor, h_factor]) #4
        result_reg = result_reg * factor[None,:]

        result_cls, result_reg, result_label = util.ml_nms(result_cls, result_reg, result_cen,result_iou, self.pos_threshold, self.nms_threshold)
        return result_cls, result_reg, result_label


    def __call__(self, cls_preds, reg_preds, cen_preds, iou_preds,ori_img_shapes, fin_img_shapes):
        """
        inference函数，只考虑batchsize为1的情况
        :param cls_preds: 分类分支的预测, 类型为list ([1, 20, 144, 100]) ([1, 20, 72, 50])[1, 20, 36, 25])([1, 80, 18, 13])([1, 80, 9, 7])
        :param reg_preds: 回归分支的预测, 类型为list
        :param cen_preds: 中心分支的预测, 类型为list
        :param ori_img_shapes: 输入图片的原始大小，即没有放缩之前的大小tensor([[500, 353]], device='cuda:0')
        :param fin_img_shapes: 输入图片放缩之后的大小             tensor([[1133,  800]], device='cuda:0')
        :return:
        """
        device=cls_preds[0].device
        dtype=cls_preds[0].dtype

        batch_size=cls_preds[0].size(0) # 1张图 测试
        feature_map_shapes=[cls_pred.shape[2:] for cls_pred in cls_preds] #[([144, 100]),([72, 50]), ([36, 25]), ([18, 13]), ([9, 7])]
        points=util.compute_points(feature_map_shapes, device, dtype) #点的坐标 te([14400, 2])   ([3600, 2])    ....            获取预设的anchor point

        result_cls=[]
        result_reg=[]
        result_label=[]


        for i in range(batch_size): #1
            cls_preds_cur_img = [cls_pred[i] for cls_pred in cls_preds] #e([20, 144, 100])([20, 72, 50])e([20, 36, 25])([20, 18, 13])([20, 9, 7])
            reg_preds_cur_img = [reg_pred[i] for reg_pred in reg_preds] #e([4, 144, 100])([4, 72, 50])e([4, 36, 25])
            cen_preds_cur_img = [cen_pred[i] for cen_pred in cen_preds] #torch.Size([1, 144, 100])。。。
            iou_preds_cur_img = [iou_preds[i] for iou_preds in iou_preds]
            result_cls_cur_img, result_reg_cur_img, result_label_cur_img = self.compute_results(
                 cls_preds_cur_img, reg_preds_cur_img, cen_preds_cur_img, iou_preds_cur_img,points, ori_img_shapes[i], fin_img_shapes[i]
            )

            result_cls.append(result_cls_cur_img) #100
            result_reg.append(result_reg_cur_img) #100 4
            result_label.append(result_label_cur_img) #100

        return result_cls, result_reg, result_label
