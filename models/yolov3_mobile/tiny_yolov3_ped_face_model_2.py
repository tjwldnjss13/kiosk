import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import nms

from .conv import Conv
from .tiny_darknet import TinyDarknet
from utils.pytorch_util import calculate_iou, convert_box_from_yxhw_to_xyxy
from metric import categorical_accuracy


class TinyYOLOV3PedFace2(nn.Module):
    def __init__(self, feat_size, num_classes, use_batch_norm):
        super().__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.backbone = TinyDarknet(use_batch_norm)

        self.feat_size = feat_size
        self.num_classes = num_classes
        self.train_safe = False

        self.num_box_predict = 3
        self.scales = [13, 26, 52]
        self.anchors = torch.Tensor([[0.22, 0.28], [0.48, 0.38], [0.78, 0.9],
                                     [0.15, 0.07], [0.11, 0.15], [0.29, 0.14],
                                     [0.03, 0.02], [0.07, 0.04], [0.06, 0.08]]).to(self.device)
        self.anchor_boxes = self._generate_anchor_boxes()

        self.stage1_conv = Conv(512, 160, 3, 1, 1, use_batch_norm)
        self.stage1_skip = Conv(160, 128, 1, 1, 0, use_batch_norm)
        self.stage1_detector = Conv(160, self.num_box_predict*(5+self.num_classes), 1, 1, 0, False, False)

        self.stage2_conv = Conv(128+512, 128, 3, 1, 1, use_batch_norm)
        self.stage2_skip = Conv(128, 96, 1, 1, 0, use_batch_norm)
        self.stage2_detector = Conv(128, self.num_box_predict*(5+self.num_classes), 1, 1, 0, False, False)

        self.stage3_conv = Conv(96+256, 64, 3, 1, 1, use_batch_norm)
        self.stage3_detector = Conv(64, self.num_box_predict*(5+self.num_classes), 1, 1, 0, False, False)

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

    def _activate_detector(self, x):
        B, H, W = x.shape[:3]
        x = x.reshape(B, -1, 5+self.num_classes)
        x = torch.cat(
            [torch.sigmoid(x[..., :2]),
             x[..., 2:4],
             torch.sigmoid(x[..., 4:])],
            dim=-1)
        x = x.reshape(B, H, W, -1)

        return x

    def forward(self, x):
        x1, x2, x3 = self.backbone(x)

        s1 = self.stage1_conv(x3)
        s1_detector = self.stage1_detector(s1)

        s1_skip = self.stage1_skip(s1)
        s1_skip = F.interpolate(s1_skip, scale_factor=2, mode='bicubic', align_corners=True)

        s2 = torch.cat([s1_skip, x2], dim=1)
        s2 = self.stage2_conv(s2)
        s2_detector = self.stage2_detector(s2)

        s2_skip = self.stage2_skip(s2)
        s2_skip = F.interpolate(s2_skip, scale_factor=2, mode='bicubic', align_corners=True)

        s3 = torch.cat([s2_skip, x1], dim=1)
        s3 = self.stage3_conv(s3)
        s3_detector = self.stage3_detector(s3)

        s1_detector = s1_detector.permute(0, 2, 3, 1)
        s2_detector = s2_detector.permute(0, 2, 3, 1)
        s3_detector = s3_detector.permute(0, 2, 3, 1)

        s1_detector = self._activate_detector(s1_detector)
        s2_detector = self._activate_detector(s2_detector)
        s3_detector = self._activate_detector(s3_detector)

        return s1_detector, s2_detector, s3_detector

    def loss(self, predict_list, target_list):
        def binary_cross_entropy_loss(predict, target, reduction='mean'):
            assert predict.shape == target.shape

            loss = -(target * torch.log(predict + 1e-16) + (1 - target) * torch.log(1 - predict + 1e-16))

            if reduction == 'mean':
                return torch.mean(loss)
            if reduction == 'sum':
                return torch.sum(loss)
            return loss

        def mse_root_loss(predict, target, reduction='mean'):
            return F.mse_loss(torch.sqrt(predict), torch.sqrt(target), reduction=reduction)

        def root_l1_loss(predict, target, reduction='mean'):
            pred = torch.sqrt(predict)
            tar = torch.sqrt(target)

            return F.l1_loss(pred, tar, reduction=reduction)

        def sigmoid_inverse(x):
            return torch.log(x / (1 - x + 1e-16))

        lambda_box = 1
        lambda_obj = 1
        lambda_noobj = 20
        lambda_cls = 10

        pred_result_list = []
        tar_result_list = []
        anchor_result_list = []

        for i in range(len(self.scales)):
            predict = predict_list[i]
            target = target_list[i]

            B, H, W = predict.shape[:3]

            pred = predict.reshape(-1, 5+self.num_classes)
            tar = target.reshape(-1, 5+self.num_classes)

            anchors = self.anchors[3*i:3*(i+1)].clone().reshape(-1).unsqueeze(0).repeat(B, H, W, 1).reshape(-1, 2)
            anchors[..., 0] *= H
            anchors[..., 1] *= W

            # pred_result = torch.zeros(pred.shape).to(self.device)
            # tar_result = torch.zeros(tar.shape).to(self.device)
            #
            # pred_result[..., :2] = pred[..., :2]
            # pred_result[..., 2:4] = torch.exp(pred[..., 2:4]) * anchors
            # pred_result[..., 4:] = pred[..., 4:]
            #
            # tar_result[..., :2] = tar[..., :2]
            # tar_result[..., 2:4] = torch.exp(tar[..., 2:4]) * anchors
            # tar_result[..., 4:] = tar[..., 4:]

            pred_result_list.append(pred)
            tar_result_list.append(tar)
            anchor_result_list.append(anchors)

        pred = torch.cat(pred_result_list, dim=0)
        tar = torch.cat(tar_result_list, dim=0)
        anchor = torch.cat(anchor_result_list, dim=0)

        objs_mask = tar[..., 4] == 1
        no_objs_mask = tar[..., 4] == 0

        with torch.no_grad():
            pred_bbox = torch.cat([
                pred[objs_mask][..., :2],
                torch.exp(pred[objs_mask][..., 2:4]) * anchor[objs_mask]
            ], dim=-1)
            tar_bbox = torch.cat([
                tar[objs_mask][..., :2],
                torch.exp(tar[objs_mask][..., 2:4]) * anchor[objs_mask]
            ], dim=-1)
            ious = calculate_iou(pred_bbox, tar_bbox, 'yxhw')

        # pred_bbox[..., 2:4] = torch.pow(torch.sigmoid(pred[objs_mask][..., 2:4]) * 2, 3)
        # tar_bbox[..., 2:4] = torch.pow(torch.sigmoid(tar[objs_mask][..., 2:4]) * 2, 3)

        # Box loss
        pred_bbox = torch.cat([
            pred[objs_mask][..., :2],
            torch.exp(pred[objs_mask][..., 2:4]) * anchor[objs_mask]
        ], dim=-1)
        tar_bbox = torch.cat([
            tar[objs_mask][..., :2],
            torch.exp(tar[objs_mask][..., 2:4]) * anchor[objs_mask]
        ], dim=-1)
        # loss_box = (F.l1_loss(pred_bbox[..., :2], tar_bbox[..., :2], reduction='mean') + root_l1_loss(pred_bbox[..., 2:4], tar_bbox[..., 2:4], reduction='mean')) / 2
        loss_box = F.l1_loss(pred_bbox[..., 0], tar_bbox[..., 0]) + \
                   F.l1_loss(pred_bbox[..., 1], tar_bbox[..., 1]) + \
                   root_l1_loss(pred_bbox[..., 2], tar_bbox[..., 2]) + \
                   root_l1_loss(pred_bbox[..., 3], tar_bbox[..., 3])

        # Object loss
        # loss_obj = lambda_obj * cross_entropy_loss(pred[objs_mask][..., 4], ious, reduction='sum') / B
        loss_obj = F.l1_loss(pred[objs_mask][..., 4], ious, reduction='mean')

        # No object loss
        loss_noobj = lambda_noobj * F.binary_cross_entropy(pred[no_objs_mask][..., 4], tar[no_objs_mask][..., 4], reduction='mean')

        # Class loss
        loss_cls = lambda_cls * sum([F.binary_cross_entropy(pred[objs_mask][..., 5+i], tar[objs_mask][..., 5+i]) for i in range(self.num_classes)])

        loss = loss_box + loss_obj + loss_noobj + loss_cls

        iou = torch.mean(ious)

        with torch.no_grad():
            acc_cls = categorical_accuracy(pred[..., 5:], tar[..., 5:])

        if not self.training:
            loss = loss.detach().cpu().item()

        return loss, \
               loss_box.detach().cpu().item(), \
               loss_obj.detach().cpu().item(), \
               loss_noobj.detach().cpu().item(), \
               loss_cls.detach().cpu().item(), \
               iou.detach().cpu().item(), \
               acc_cls

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

    def final_result(self, predict_list, confidence_threshold, nms_threshold):
        assert len(predict_list) == len(self.scales)

        box_list = []
        conf_list = []
        cls_list = []

        box_per_cls_list = []
        conf_per_cls_list = []

        for i in range(len(self.scales)):
            pred = predict_list[i].clone().reshape(-1, 5+self.num_classes)
            anchor_boxes = self._generate_anchor_box(self.anchors[3*i:3*(i+1)], self.scales[i])
            anchor_boxes = anchor_boxes.reshape(-1, 4)

            pred[..., :2] = torch.sigmoid(pred[..., :2]) + anchor_boxes[..., :2]
            pred[..., 2:4] = torch.exp_(pred[..., 2:4]) * anchor_boxes[..., 2:4]

            conf_idx = pred[..., 4] >= confidence_threshold
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

                nms_idx = nms(box_cls, conf_cls, nms_threshold)
                box_cls = box_cls[nms_idx]
                conf_cls = conf_cls[nms_idx]

            box_per_cls_list.append(box_cls)
            conf_per_cls_list.append(conf_cls)

        return box_per_cls_list, conf_per_cls_list


if __name__ == '__main__':
    from torchsummary import summary
    model = TinyYOLOV3PedFace2((416, 416), 2, False).cuda()
    summary(model, (3, 416, 416))





















