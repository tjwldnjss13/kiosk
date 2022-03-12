import os
import time
import random
import argparse
import datetime
import numpy as np
import cv2 as cv
import torchvision.transforms as T
import torchvision.ops as ops

from sys import platform

from deep_sort import DeepSort
# from models.yolov2_ped_face_model_lite import YOLOV2PedFaceLite
from models.yolov3_mobile.tiny_yolov3_ped_face_model_2 import TinyYOLOV3PedFace2
from models.agnet_mobilenetv2_super_sub import AGNetSuperSub
from utils.pytorch_util import *

vis_data_dict = {'id': '0',
                 'kiosk_no': -1,
                 'gender': '0',
                 'age': '0',
                 'approach_time': '0',
                 'stay_time': -1}

sub_data_dict = {'face_ids_saved': [],
                 'person_ids_saved': [],
                 'start_time': -1,
                 'end_time': -1,
                 'face_detecting': False,
                 'person_detecting': False,
                 'face_detected': False,
                 'person_detected': False,
                 'first_appear_time': 2000,
                 'legal_leave_time': 3000,
                 'record_flag': False,
                 'num_ped': 0,
                 'record_file': '0',
                 'prev_face_id': 0,
                 'cur_person_id': 0,
                 'same_face_flag': False}

bool_str = ['', 'not']

num_ped = 0


def get_current_microsecond():
    return int(time.time() * 1000)


def get_current_date_time_str():
    now = datetime.datetime.now()
    year = now.year
    month = now.month
    day = now.day
    hour = now.hour
    min = now.minute
    sec = now.second
    microsec = now.microsecond // 1000

    time_str = f'{year}-{month:02d}-{day:02d} {hour:02d}:{min:02d}:{sec:02d}.{microsec:03d}'

    return time_str


def get_current_date_time_str_title():
    now = datetime.datetime.now()
    year = now.year
    month = now.month
    day = now.day
    hour = now.hour
    min = now.minute
    sec = now.second
    microsec = now.microsecond // 1000

    time_str = f'{year}{month:02d}{day:02d}-{hour:02d}{min:02d}{sec:02d}{microsec:03d}'

    return time_str


def bbox_rel(bbox_left, bbox_top, bbox_w, bbox_h):
    """" Calculates the relative bounding box from absolute pixel values. """
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


# def draw_face_box(img, bbox, identity, offset=(0,0), age=None, gender=None):
#     x1, y1, x2, y2 = [int(i) for i in bbox]
#     x1 += offset[0]
#     x2 += offset[0]
#     y1 += offset[1]
#     y2 += offset[1]
#
#     color = compute_color_for_labels(identity)
#
#     label = '{}{:d}:{}'.format("", identity, 'face')
#     t_size = cv.getTextSize(label, cv.FONT_HERSHEY_PLAIN, 2, 2)[0]
#     img = cv.rectangle(img.copy(), (x1, y1), (x2, y2), color, 3)
#     img = cv.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
#     img = cv.putText(img, label, (x1, y1 + t_size[1] + 4), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
#
#     t_size = cv.getTextSize(str(age), cv.FONT_HERSHEY_PLAIN, 2, 2)[0]
#     img = cv.rectangle(img, (x1, y2), (x1 + t_size[0] + 3, y2 + t_size[1] + 4), color, -1)
#     img = cv.putText(img, age, (x1, y2 + t_size[1] + 4), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
#
#     t_size = cv.getTextSize(gender, cv.FONT_HERSHEY_PLAIN, 2, 2)[0]
#     img = cv.rectangle(img, (x1, y2 + t_size[1] + 4), (x1 + t_size[0] + 3, y2 + 2 * t_size[1] + 4), color, -1)
#     img = cv.putText(img, gender, (x1, y2 + 2 * t_size[1] + 4), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
#
#     return img


def draw_box(img, bbox, identity, class_label, offset=(0,0), age=None, gender=None):
    names = ['person', 'face']

    x1, y1, x2, y2 = [int(i) for i in bbox]
    x1 += offset[0]
    x2 += offset[0]
    y1 += offset[1]
    y2 += offset[1]

    color_person = (0, 0, 0)
    color_face = compute_color_for_labels(identity)
    colors = [color_person, color_face]

    label = '{}{:d}:{}'.format("", identity, names[class_label])
    t_size = cv.getTextSize(label, cv.FONT_HERSHEY_PLAIN, 2, 2)[0]
    img = cv.rectangle(img.copy(), (x1, y1), (x2, y2), colors[class_label], 3)
    img = cv.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), colors[class_label], -1)
    img = cv.putText(img, label, (x1, y1 + t_size[1] + 4), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

    if age:
        t_size = cv.getTextSize(str(age), cv.FONT_HERSHEY_PLAIN, 2, 2)[0]
        img = cv.rectangle(img, (x1, y2), (x1 + t_size[0] + 3, y2 + t_size[1] + 4), colors[class_label], -1)
        img = cv.putText(img, age, (x1, y2 + t_size[1] + 4), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

    if gender:
        t_size = cv.getTextSize(gender, cv.FONT_HERSHEY_PLAIN, 2, 2)[0]
        img = cv.rectangle(img, (x1, y2 + t_size[1] + 4), (x1 + t_size[0] + 3, y2 + 2 * t_size[1] + 4), colors[class_label], -1)
        img = cv.putText(img, gender, (x1, y2 + 2 * t_size[1] + 4), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

    return img


def make_new_record_file(prev_date_str):
    now = datetime.datetime.now()
    date_str = f'{now.date().year}{now.date().month:02d}{now.date().day:02d}'
    fn_record = date_str + '_record.csv'
    fn_num_ped = date_str + '_num_ped.csv'
    sub_data_dict['record_file'] = f'records/{fn_record}'
    sub_data_dict['num_ped_file'] = f'records/{fn_num_ped}'

    if prev_date_str != date_str or prev_date_str is None:
        if os.path.exists(f'records/{fn_record}'):
            with open(f'records/{fn_record}', 'r') as f_record:
                lines = f_record.readlines()[1:]
                for line in lines:
                    id_ = int(line.strip().split(',')[0])
                    sub_data_dict['face_ids_saved'].append(id_)
        else:
            f_record = open(f'records/{fn_record}', 'w+')
            f_ped = open(f'records/{fn_num_ped}', 'w+')
            s = ''
            for i, k in enumerate(vis_data_dict.keys()):
                s += k
                if i <= len(vis_data_dict.keys()) - 2:
                    s += ', '
                else:
                    s += '\n'
            f_record.write(s)
            f_record.close()
            f_ped.close()

    return date_str


def update_record(age_results, gender_results, gender_labels):
    vis_data_dict['stay_time'] = sub_data_dict['end_time'] - sub_data_dict['start_time']
    if vis_data_dict['stay_time'] >= sub_data_dict['first_appear_time']:
        # age_idx = torch.Tensor(age_results).mode(-1).values.type(torch.long).item()
        # age = age_labels[age_idx]
        age = int(sum(age_results) // len(age_results))

        gender_idx = torch.Tensor(gender_results).mode(-1).values.type(torch.long).item()
        gender = gender_labels[gender_idx]

        id = vis_data_dict['id']
        kiosk_id = vis_data_dict['kiosk_no']
        approach_time = vis_data_dict['approach_time']
        stay_time = vis_data_dict['stay_time']

        sub_data_dict['face_ids_saved'].append(id)
        sub_data_dict['person_ids_saved'].append(sub_data_dict['cur_person_id'])

        record_str = f'{str(id)},{kiosk_id},{gender},{age},{approach_time},{stay_time}\n'
        f_record = open(sub_data_dict['record_file'], 'a+')
        f_record.write(record_str)
        f_record.close()
        print('Recorded.')
    else:
        print('Left.')

    sub_data_dict['face_detecting'] = False
    sub_data_dict['person_detecting'] = False
    sub_data_dict['face_detected'] = False
    sub_data_dict['person_detected'] = False
    sub_data_dict['same_face_flag'] = False


def get_age_result(age_super, age_subs, gender):
    age_super = age_super[0]
    age_subs = [age_sub[0] for age_sub in age_subs]
    gender = gender[0]

    age_super_idx = torch.argmax(age_super, dim=-1)
    age_sub_idx = [torch.argmax(i, dim=-1) for i in age_subs]

    age_score, _ = torch.max(age_subs[age_super_idx], dim=-1)
    gender_score, _ = torch.max(gender, dim=-1)

    age = (age_super_idx + 3) * 5 + age_sub_idx[age_super_idx]
    gender = torch.argmax(gender, dim=-1)

    return age.detach().cpu().item(), gender.detach().cpu().item()


def detect(args):
    yolo_ckpt_pth = args.yolo_ckpt
    agnet_ckpt_pth = args.agnet_ckpt
    view_result = args.view_result
    ped_conf_thresh = args.person_conf
    face_conf_thresh = args.face_conf
    iou_nms = args.iou_nms

    print('view_result:', view_result)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device:', torch.cuda.get_device_name(0), device)

    prev_date_str = None

    # Initialize model
    yolo = TinyYOLOV3PedFace2(
        feat_size=(416, 416),
        num_classes=2,
        use_batch_norm=False
    ).to(device)
    yolo_ckpt = torch.load(yolo_ckpt_pth)
    yolo.load_state_dict(yolo_ckpt['model_state_dict'])
    num_classes = 2
    print(f'Model {yolo.__class__.__name__} loaded.')

    agnet = AGNetSuperSub().to(device)
    agnet_ckpt = torch.load(agnet_ckpt_pth)
    agnet.load_state_dict(agnet_ckpt['model_state_dict'])
    print(f'Model {agnet.__class__.__name__} loaded.')

    deepsort_ped = DeepSort("deep_sort/deep/checkpoint/ckpt.t7")
    deepsort_face = DeepSort("deep_sort/deep/checkpoint/ckpt.t7")
    print('DeepSort loaded.')

    # Eval mode
    yolo.eval()
    agnet.eval()

    # Set Dataloader
    torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference

    # Get names and colors
    ages = [str(i) for i in range(15, 59)] + ['59+']
    genders = ['Male', 'Female']

    print('Preparing for camera...')
    now = get_current_date_time_str_title()
    cap = cv.VideoCapture(0)

    transform = T.ToTensor()

    cur_age_list = []
    cur_gender_list = []

    # Load frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        prev_date_str = make_new_record_file(prev_date_str)

        # Inference per frame
        img = cv.resize(frame, (416, 416), interpolation=cv.INTER_CUBIC)
        img_result = img

        img = transform(img).unsqueeze(0).to(device)
        h, w = img.shape[2:]

        pred = yolo(img)[0]
        pred = pred.view(-1, 5 + num_classes)
        cls = torch.argmax(pred[..., 5:], dim=-1)

        torch.cuda.empty_cache()

        idx_ped = cls == 0
        coords_yxhw_ped = pred[idx_ped][..., :4]
        conf_ped = pred[idx_ped][..., 4]
        cls_ped = cls[idx_ped]

        del idx_ped
        torch.cuda.empty_cache()

        idx_face = cls == 1
        coords_yxhw_face = pred[idx_face][..., :4]
        conf_face = pred[idx_face][..., 4]
        cls_face = cls[idx_face]

        del idx_face

        del pred, cls
        torch.cuda.empty_cache()

        coords_yxyx_ped = convert_box_from_yxhw_to_yxyx(coords_yxhw_ped)
        coords_yxyx_ped[..., 0:3:2] *= (h / 13)
        coords_yxyx_ped[..., 1:4:2] *= (w / 13)

        coords_yxyx_face = convert_box_from_yxhw_to_yxyx(coords_yxhw_face)
        coords_yxyx_face[..., 0:3:2] *= (h / 13)
        coords_yxyx_face[..., 1:4:2] *= (w / 13)

        valid_idx_ped = torch.where(conf_ped > ped_conf_thresh)[0]
        coords_yxyx_ped = coords_yxyx_ped[valid_idx_ped]
        conf_ped = conf_ped[valid_idx_ped]

        del valid_idx_ped
        torch.cuda.empty_cache()

        valid_idx_face = torch.where(conf_face > face_conf_thresh)[0]
        coords_yxyx_face = coords_yxyx_face[valid_idx_face]
        conf_face = conf_face[valid_idx_face]

        del valid_idx_face
        torch.cuda.empty_cache()

        coords_xyxy_ped = convert_box_from_yxyx_to_xyxy(coords_yxyx_ped)
        coords_xyxy_face = convert_box_from_yxyx_to_xyxy(coords_yxyx_face)

        del coords_yxyx_ped, coords_yxyx_face
        torch.cuda.empty_cache()

        # Apply NMS
        nms_idx_ped = ops.nms(coords_xyxy_ped, conf_ped, iou_nms)
        coords_ped = coords_xyxy_ped[nms_idx_ped]
        conf_ped = conf_ped[nms_idx_ped]
        cls_ped = cls_ped[nms_idx_ped]

        del nms_idx_ped, coords_xyxy_ped
        torch.cuda.empty_cache()

        # print('coords_ped after nms', coords_ped)

        size_idx_ped = (coords_ped[..., 2] - coords_ped[..., 0]) * (coords_ped[..., 3] - coords_ped[..., 1]) > 10000
        coords_ped = coords_ped[size_idx_ped]
        conf_ped = conf_ped[size_idx_ped]
        cls_ped = cls_ped[size_idx_ped]

        del size_idx_ped
        torch.cuda.empty_cache()

        # print('coords_ped with valid size', coords_ped)

        nms_idx_face = ops.nms(coords_xyxy_face, conf_face, iou_nms)
        coords_face = coords_xyxy_face[nms_idx_face]
        conf_face = conf_face[nms_idx_face]
        cls_face = cls_face[nms_idx_face]

        del nms_idx_face, coords_xyxy_face
        torch.cuda.empty_cache()

        nonzero_idx_face = (coords_face[..., 0] > 0) * (coords_face[..., 1] > 0) * (coords_face[..., 2] > 0) * (coords_face[..., 3] > 0)
        coords_face = coords_face[nonzero_idx_face]
        conf_face = conf_face[nonzero_idx_face]
        cls_face = cls_face[nonzero_idx_face]

        del nonzero_idx_face
        torch.cuda.empty_cache()

        size_idx_face = (coords_face[..., 2] - coords_face[..., 0] > 50) * (coords_face[..., 3] - coords_face[..., 1] > 50)
        coords_face = coords_face[size_idx_face]
        conf_face = conf_face[size_idx_face]
        cls_face = cls_face[size_idx_face]

        del size_idx_face
        torch.cuda.empty_cache()

        # Select the largest box
        if len(coords_ped) > 1:
            areas_ped = (coords_ped[..., 2] - coords_ped[..., 0]) * (coords_ped[..., 3] - coords_ped[..., 1])
            max_ped_idx = torch.argmax(areas_ped, dim=-1)
            coords_ped = coords_ped[max_ped_idx].unsqueeze(0)
            conf_ped = conf_ped[max_ped_idx].unsqueeze(0)
            cls_ped = cls_ped[max_ped_idx].unsqueeze(0)

            del areas_ped, max_ped_idx
            torch.cuda.empty_cache()

        if len(coords_face) > 1:
            areas_face = (coords_face[..., 2] - coords_face[..., 0]) * (coords_face[..., 3] - coords_face[..., 1])
            max_face_idx = torch.argmax(areas_face, dim=-1)
            coords_face = coords_face[max_face_idx].unsqueeze(0)
            conf_face = conf_face[max_face_idx].unsqueeze(0)
            cls_face = cls_face[max_face_idx].unsqueeze(0)

            del areas_face, max_face_idx
            torch.cuda.empty_cache()

        pred_ped = torch.cat([coords_ped, conf_ped.unsqueeze(-1), cls_ped.unsqueeze(-1)], dim=-1)
        pred_face = torch.cat([coords_face, conf_face.unsqueeze(-1), cls_face.unsqueeze(-1)], dim=-1)

        # pred = [torch.cat([pred_ped, pred_face], dim=0)]
        pred = [pred_ped, pred_face]

        del coords_ped, conf_ped, cls_ped, coords_face, conf_face, cls_face, pred_ped, pred_face
        torch.cuda.empty_cache()

        img_np = np.array(img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy())

        # Track objects
        sub_data_dict['person_detecting'] = len(pred[0]) > 0
        sub_data_dict['face_detecting'] = len(pred[1]) > 0

        for i, det in enumerate(pred):
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords((h, w), det[:, :4], img_np.shape).round()
                # print(f'{len(det)} persons, ', end='')

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class

                bbox_xywh_ped = []
                bbox_xywh_face = []
                confs_ped = []
                confs_face = []

                # Update per object
                for *xyxy, conf, cls in det:
                    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
                    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
                    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
                    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
                    x_c, y_c, bbox_w, bbox_h = bbox_rel(bbox_left, bbox_top, bbox_w, bbox_h)
                    obj = [x_c, y_c, bbox_w, bbox_h]

                    if i == 0:
                        bbox_xywh_ped.append(obj)
                        confs_ped.append(conf.item())
                        outputs = deepsort_ped.update((torch.Tensor(bbox_xywh_ped)), (torch.Tensor(confs_ped)), img_np)

                        if len(outputs) > 0:
                            bbox_xyxy_ped = outputs[:, :4][0]
                            id_ped = int(outputs[:, -1][0])

                            if not sub_data_dict['person_detected']:
                                if id_ped in sub_data_dict['person_ids_saved']:
                                    id_ped = max(sub_data_dict['person_ids_saved']) + 1
                                sub_data_dict['cur_person_id'] = id_ped
                                sub_data_dict['person_detected'] = True
                            else:
                                id_ped = sub_data_dict['cur_person_id']

                            # sub_data_dict['cur_person_id'] = id_ped

                            # img_result = draw_boxes(img_result.copy(), bbox_xyxy_ped, id_ped, 0)
                            img_result = draw_box(img_result.copy(), bbox_xyxy_ped, id_ped, 0)

                    elif i == 1:
                        bbox_xywh_face.append(obj)
                        confs_face.append(conf.item())
                        outputs = deepsort_face.update((torch.Tensor(bbox_xywh_face)), (torch.Tensor(confs_face)), img_np)

                        if len(outputs) > 0:
                            print(end='\r')
                            
                            if not sub_data_dict['face_detected']:
                                cur_age_list = []
                                cur_gender_list = []

                            bbox_xyxy_face = outputs[:, :4][0]
                            id_face = int(outputs[:, -1][0])

                            # Classify age, gender per face object
                            age_result_list = []
                            gender_result_list = []

                            face_feat = img_np[bbox_xyxy_face[1]:bbox_xyxy_face[3], bbox_xyxy_face[0]:bbox_xyxy_face[2]]
                            face_feat = cv.resize(face_feat, (224, 224), interpolation=cv.INTER_CUBIC)
                            face_tensor = T.ToTensor()(face_feat).unsqueeze(0).to(device)

                            pred_age_super, pred_age_sub, pred_gender = agnet(face_tensor)
                            pred_age, pred_gender = get_age_result(pred_age_super, pred_age_sub, pred_gender)

                            age_result_list.append(ages[pred_age-15])
                            gender_result_list.append(genders[pred_gender])

                            cur_age_list.append(pred_age)
                            cur_gender_list.append(pred_gender)

                            if not sub_data_dict['face_detected']:
                                if id_face in sub_data_dict['face_ids_saved']:
                                    id_face = max(sub_data_dict['face_ids_saved']) + 1
                                vis_data_dict['id'] = id_face
                                vis_data_dict['approach_time'] = get_current_date_time_str()
                                sub_data_dict['start_time'] = get_current_microsecond()
                                sub_data_dict['face_detected'] = True
                                sub_data_dict['prev_face_id'] = id_face
                            else:
                                id_face = sub_data_dict['prev_face_id']

                            # img_result = draw_face_box(img_result.copy(), bbox_xyxy_face, id_face, age=ages[pred_age-15], gender=genders[pred_gender])
                            img_result = draw_box(img_result.copy(), bbox_xyxy_face, id_face, 1, age=ages[pred_age-15], gender=genders[pred_gender])

                            sub_data_dict['end_time'] = get_current_microsecond()

                            t_cur = sub_data_dict['end_time'] - sub_data_dict['start_time']
                            print(f'{id_face}, {t_cur}ms ', end='')

            if sub_data_dict['face_detected'] and not sub_data_dict['face_detecting']:
                if sub_data_dict['person_detecting']:
                    sub_data_dict['same_face_flag'] = True
                else:
                    t_temp = get_current_microsecond()
                    if t_temp - sub_data_dict['end_time'] > sub_data_dict['legal_leave_time']:
                        update_record(cur_age_list, cur_gender_list, genders)

            torch.cuda.empty_cache()

        if view_result:
            img_result = cv.resize(img_result, (640, 480))
            cv.imshow('result', img_result)

        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
   
    parser.add_argument('--yolo_ckpt', required=False, type=str, default='./models/weights/TinyYOLOV3PedFace2_temp.ckpt')
    parser.add_argument('--agnet_ckpt', required=False, type=str, default='./models/weights/AGNetSuperSub.ckpt')
    parser.add_argument('--view_result', required=False, type=bool, default=True)
    parser.add_argument('--person_conf', required=False, type=float, default=.5)
    parser.add_argument('--face_conf', required=False, type=float, default=.2)
    parser.add_argument('--iou_nms', required=False, type=float, default=.3)

    args = parser.parse_args()

    with torch.no_grad():
        detect(args)
