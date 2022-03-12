import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms

from utils.pytorch_util import calculate_iou, convert_box_from_yxhw_to_xyxy
from metric import categorical_accuracy


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_bn=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Sequential()
        self.leaky_relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.leaky_relu(x)

        return x


class TinyDarknet2(nn.Module):
    def __init__(self, num_classes, device=torch.device('cuda:0')):
        super().__init__()
        self.num_classes = num_classes
        self.device = device
        self.scales = [13, 26]
        self.anchors = torch.Tensor([[0.19, 0.19], [0.41, 0.32], [0.77, 0.83],
                                     [0.03, 0.02], [0.06, 0.05], [0.14, 0.09]]).to(self.device)

        self.convs1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.LeakyReLU(inplace=True)
        )
        self.convs2_1 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.LeakyReLU(inplace=True)
        )
        self.convs2_2 = nn.Sequential(
            nn.MaxPool2d(2, 1),
            nn.Conv2d(512, 1024, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(1024, 256, 1, 1, 0),
            nn.LeakyReLU(inplace=True)
        )
        self.scale1 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(512, 3*(5+self.num_classes), 1, 1, 0)
        )
        self.scale2_1 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.LeakyReLU(inplace=True)
        )
        self.scale2_2 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 3*(5+self.num_classes), 1, 1, 0)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _activate(self, x, scale_idx):
        B, _, H, W = x.shape
        anchors = self.anchors[3*scale_idx:3*(scale_idx+1)].clone().reshape(-1).unsqueeze(0).repeat(H, W, 1).reshape(-1, 2)
        anchors[..., 0] *= H
        anchors[..., 1] *= W
        x = x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, 5+self.num_classes)
        x = torch.cat([
            torch.sigmoid(x[..., 0:2]),
            torch.exp(x[..., 2:4]) * anchors,
            torch.sigmoid(x[..., 4:])
        ], dim=-1)

        x = x.reshape(B, H, W, -1, 5+self.num_classes)

        return x

    def _generate_anchor_box(self, anchor_box_sizes, out_size):
        """
        Make anchor box the same shape as output's.
        :param anchor_box_sizes: tensor, [# anchor box, (height, width)]
        :param out_size: tuple or list or int, (height, width) or size,

        :return tensor, [height, width, (cy, cx, h, w) * num bounding box]
        """

        if isinstance(out_size, int):
            out_size = (out_size, out_size)

        out = torch.zeros(out_size[0], out_size[1], 4 * len(anchor_box_sizes)).cuda()
        cy_ones = torch.ones(1, out_size[1], dtype=torch.int)
        cx_ones = torch.ones(out_size[0], 1, dtype=torch.int)
        cy_tensor = torch.zeros(1, out_size[1], dtype=torch.int)
        cx_tensor = torch.zeros(out_size[0], 1, dtype=torch.int)
        for i in range(1, out_size[0]):
            cy_tensor = torch.cat([cy_tensor, cy_ones * i], dim=0)
            cx_tensor = torch.cat([cx_tensor, cx_ones * i], dim=1)

        ctr_tensor = torch.cat([cy_tensor.unsqueeze(2), cx_tensor.unsqueeze(2)], dim=2)

        for i in range(len(anchor_box_sizes)):
            out[:, :, 4 * i:4 * i + 2] = ctr_tensor.type(torch.float32)
            out[:, :, 4 * i + 2] = anchor_box_sizes[i, 0] * out_size[0]
            out[:, :, 4 * i + 3] = anchor_box_sizes[i, 1] * out_size[1]

        return out

    def _generate_anchor_boxes(self):
        anchor_box_list = []
        for i in range(len(self.scales)):
            anchor_box = self._generate_anchor_box(self.anchors[3*i:3*(i+1)], self.scales[i])
            anchor_box_list.append(anchor_box)

        return anchor_box_list

    def forward(self, x):
        x1 = self.convs1(x)
        x2 = self.convs2_1(x1)
        x2 = F.pad(x2, (1, 0, 1, 0), mode='replicate')
        x2 = self.convs2_2(x2)

        s1 = self.scale1(x2)
        s1 = self._activate(s1, 0)

        s2 = self.scale2_1(x2)
        s2 = F.interpolate(s2, scale_factor=2, mode='nearest')
        s2 = torch.cat([x1, s2], dim=1)
        s2 = self.scale2_2(s2)
        s2 = self._activate(s2, 1)

        return s1, s2

    def loss(self, predict_list, target_list):

        def giou_loss(predict, target, reduction='mean'):
            num_objs = len(predict)
            iou = calculate_iou(predict, target, box_format='yxhw')

            area_pred = predict[..., 2] * predict[..., 3]
            area_tar = target[..., 2] * target[..., 3]

            pred = convert_box_from_yxhw_to_xyxy(predict)
            tar = convert_box_from_yxhw_to_xyxy(target)

            x1_inter = torch.maximum(pred[..., 0], tar[..., 0])
            y1_inter = torch.maximum(pred[..., 1], tar[..., 1])
            x2_inter = torch.minimum(pred[..., 2], tar[..., 2])
            y2_inter = torch.minimum(pred[..., 3], tar[..., 3])
            area_inter = nn.ReLU()(x2_inter - x1_inter) * nn.ReLU()(y2_inter - y1_inter)
            area_union = area_pred + area_tar - area_inter

            x1_global = torch.minimum(pred[..., 0], tar[..., 0])
            y1_global = torch.minimum(pred[..., 1], tar[..., 1])
            x2_global = torch.maximum(pred[..., 2], tar[..., 2])
            y2_global = torch.maximum(pred[..., 3], tar[..., 3])
            area_global = nn.ReLU()(x2_global - x1_global) * nn.ReLU()(y2_global - y1_global)

            area_remain = area_global - area_union
            giou = iou - (area_remain / area_global)
            loss = 1 - giou

            if reduction == 'mean':
                return loss.mean()
            if reduction == 'sum':
                return loss.sum()
            return loss

        def ciou_loss(predict, target, reduction='mean'):
            iou = calculate_iou(predict, target, box_format='yxhw')

            pred_center = predict[..., 0:2]
            tar_center = target[..., 0:2]
            pred_hw = predict[..., 2:4]
            tar_hw = target[..., 2:4]

            pred = convert_box_from_yxhw_to_xyxy(predict)
            tar = convert_box_from_yxhw_to_xyxy(target)

            x1_global = torch.minimum(pred[..., 0], tar[..., 0])
            y1_global = torch.minimum(pred[..., 1], tar[..., 1])
            x2_global = torch.maximum(pred[..., 2], tar[..., 2])
            y2_global = torch.maximum(pred[..., 3], tar[..., 3])

            euclidean_center_distance = torch.sqrt(torch.sum(torch.square(pred_center - tar_center), dim=-1))
            diagonal_length = torch.sqrt(torch.square(x2_global - x1_global) + torch.square(y2_global - y1_global))
            aspect_ratio_consistency = 4 / math.pi ** 2 * \
                                       torch.square(torch.arctan_(tar_hw[..., 1] / tar_hw[..., 0]) - \
                                                    torch.arctan_(pred_hw[..., 1] / pred_hw[..., 0]))
            trade_off_param = aspect_ratio_consistency / (1 - iou + aspect_ratio_consistency)

            loss = 1 - iou + euclidean_center_distance / torch.square(diagonal_length) + trade_off_param * aspect_ratio_consistency

            if reduction == 'mean':
                return loss.mean()
            if reduction == 'sum':
                return loss.sum()
            return loss

        def multi_mse_loss(predict, target, reduction='mean'):
            squared = torch.square(predict - target)
            loss = torch.sum(squared, dim=-1)

            if reduction == 'mean':
                return loss.mean()
            if reduction == 'sum':
                return loss.sum()
            return loss

        def multi_bce_loss(predict, target, reduction='mean'):
            def bce(predict, target, reduction):
                ce = -(target * torch.log(predict + 1e-16) + (1 - target) * torch.log(1 - predict + 1e-16))

                if reduction == 'mean':
                    return ce.mean()
                if reduction == 'sum':
                    return ce.sum()
                return ce

            loss = sum([bce(predict[..., i], target[..., i], reduction) for i in range(predict.shape[-1])])

            return loss

        lambda_box = 50
        lambda_obj = 1
        lambda_noobj = 10
        lambda_cls = 1

        pred_result_list = []
        tar_result_list = []
        anchor_result_list = []

        for i in range(len(self.scales)):
            pred = predict_list[i]
            tar = target_list[i]

            B, H, W = pred.shape[:3]
            pred = pred.reshape(-1, 5+self.num_classes)
            tar = tar.reshape(-1, 5+self.num_classes)

            anchors = self.anchors[3*i:3*(i+1)].clone().reshape(-1).unsqueeze(0).repeat(B, H, W, 1).reshape(-1, 2)
            anchors[..., 0] *= H
            anchors[..., 1] *= W

            pred_result_list.append(pred)
            tar_result_list.append(tar)
            anchor_result_list.append(anchors)

        pred = torch.cat(pred_result_list, dim=0)
        tar = torch.cat(tar_result_list, dim=0)
        anchor = torch.cat(anchor_result_list, dim=0)

        objs_mask = tar[..., 4] == 1
        no_objs_mask = tar[..., 4] == 0

        pred_bbox = pred[objs_mask][..., 0:4]
        # tar_bbox = torch.cat([
        #     tar[objs_mask][..., 0:2],
        #     torch.exp(tar[objs_mask][..., 2:4]) * anchor[objs_mask]
        # ], dim=-1)
        tar_bbox = tar[objs_mask][..., 0:4]
        with torch.no_grad():
            ious = calculate_iou(pred_bbox, tar_bbox, 'yxhw')

        loss_box = lambda_box * giou_loss(pred_bbox, tar_bbox, reduction='mean')
        # loss_box = lambda_box * F.mse_loss(pred_bbox, tar_bbox, reduction='mean')
        loss_obj = lambda_obj * F.mse_loss(pred[objs_mask][..., 4], ious * tar[objs_mask][..., 4], reduction='mean')
        loss_noobj = lambda_noobj * F.binary_cross_entropy(pred[no_objs_mask][..., 4], tar[no_objs_mask][..., 4], reduction='mean')
        loss_cls = lambda_cls * F.binary_cross_entropy(pred[objs_mask][..., 5:], tar[objs_mask][..., 5:], reduction='mean')

        loss = loss_box + loss_obj + loss_noobj + loss_cls
        iou = torch.mean(ious)

        with torch.no_grad():
            acc_cls = categorical_accuracy(pred[objs_mask][..., 5:], tar[objs_mask][..., 5:])
        # acc_cls = 0

        if not self.training:
            loss = loss.detach().cpu().item()

        return loss, \
               loss_box.detach().cpu().item(), \
               loss_obj.detach().cpu().item(), \
               loss_noobj.detach().cpu().item(), \
               loss_cls.detach().cpu().item(), \
               iou.detach().cpu().item(), \
               acc_cls

    def final_result(self, predict_list, conf_thresh, nms_thresh):
        box_list = []
        conf_list = []
        cls_list = []

        box_per_cls_list = []
        conf_per_cls_list = []

        for i in range(len(self.scales)):
            pred = predict_list[i].clone().reshape(-1, 5 + self.num_classes)
            anchor_boxes = self._generate_anchor_box(self.anchors[3 * i:3 * (i + 1)], self.scales[i])
            anchor_boxes = anchor_boxes.reshape(-1, 4)

            pred[..., :2] = pred[..., :2] + anchor_boxes[..., :2]

            conf_idx = pred[..., 4] >= conf_thresh
            pred = pred[conf_idx]
            pred_box = pred[..., :4]
            pred_conf = pred[..., 4]
            pred_cls = pred[..., 5:]

            pred_box = convert_box_from_yxhw_to_xyxy(pred_box) / self.scales[i]

            box_list.append(pred_box)
            conf_list.append(pred_conf)
            cls_list.append(pred_cls)

        pred_box = torch.cat(box_list, dim=0)
        pred_conf = torch.cat(conf_list, dim=0)
        pred_cls = torch.cat(cls_list, dim=0)
        if len(pred_cls) > 0:
            pred_cls = torch.argmax(pred_cls, dim=-1)

        for i in range(self.num_classes):
            box_cls = []
            conf_cls = []

            if len(pred_cls) > 0:
                cls_idx = pred_cls == i
                box_cls = pred_box[cls_idx]
                conf_cls = pred_conf[cls_idx]

                nms_idx = nms(box_cls, conf_cls, nms_thresh)
                box_cls = box_cls[nms_idx]
                conf_cls = conf_cls[nms_idx]

            box_per_cls_list.append(box_cls)
            conf_per_cls_list.append(conf_cls)

        return box_per_cls_list, conf_per_cls_list



if __name__ == '__main__':
    from torchsummary import summary
    model = TinyDarknet2(1).cuda()
    summary(model, (3, 416, 416))
