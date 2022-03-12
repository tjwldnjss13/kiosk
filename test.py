import os
import time
import datetime
import argparse
import numpy as np
import torchvision.transforms as T
from sklearn.metrics import average_precision_score, precision_recall_curve

from torch.utils.data import DataLoader, random_split
from torchvision.ops import nms

from models.yolov2_ped_face_model_lite import YOLOV2PedFaceLite
from models.agnet_mobilenetv2_super_sub import *
from datasets.yolo_coco_dataset import *
from datasets.coco_dataset import *
from datasets.afad_dataset import *
from utils.pytorch_util import *
from metric import average_precision


def get_current_time_str():
    now = datetime.datetime.now()
    hour = now.hour
    min = now.minute
    sec = now.second
    microsec = now.microsecond // 1000

    time_str = f'{hour:02d}:{min:02d}:{sec:02d}:{microsec:03d}'

    return time_str


def get_ap(confusion_matrix):
    cm = confusion_matrix


def test_person(args):

    def get_output(predict, num_classes):
        pred = predict.reshape(-1, 5+num_classes)

        person_idx = torch.argmax(pred[..., 5:], dim=-1) == 0
        pred = pred[person_idx]

        valid_idx = pred[..., 4] > args.conf_thresh
        pred = pred[valid_idx]

        if len(pred) == 0:
            return None

        pred_bbox = convert_box_from_yxhw_to_yxyx(pred[..., :4])
        # pred_bbox = convert_box_from_yxyx_to_xyxy(pred_bbox)
        pred_scores = pred[..., 4]
        pred_cls = torch.argmax(pred[..., 5:], dim=-1)

        nms_idx = nms(pred_bbox, pred_scores, args.nms_thresh)
        pred_bbox = pred_bbox[nms_idx]
        pred_scores = pred_scores[nms_idx]
        pred_cls = pred_cls[nms_idx]

        return pred_bbox, pred_scores, pred_cls

    def get_results(predict, annotation):
        pred_bbox, pred_score, pred_cls = predict

        if pred_bbox is None:
            return np.array([]), np.array([]), np.array([])

        pred_bbox *= 32

        del predict
        torch.cuda.empty_cache()

        tar_bbox = annotation['bbox'].to(device)
        tar_cls = annotation['class'].to(device)
        tar_person_idx = tar_cls == 0
        tar_bbox = tar_bbox[tar_person_idx]

        if len(tar_bbox) == 0:
            return np.array([]), np.array([]), np.array([])

        ious = calculate_ious_grid(pred_bbox, tar_bbox)

        ious_val, ious_idx = torch.sort(ious, dim=-1, descending=True)

        truths = ious_val[:, 0] > args.iou_thresh

        cls_idx = pred_cls == 0
        # truth_list = np.append(truth_list, truths[cls_idx].numpy())
        # score_list = np.append(score_list, pred_score[cls_idx].detach().cpu().numpy())

        # print()
        # print(ious_val[0].detach().cpu().numpy())
        # print(truths[cls_idx].numpy())
        # print(pred_score[cls_idx].detach().cpu().numpy())
        # exit()

        # return truth_list, score_list, ious_val[0].detach().cpu().numpy()
        return ious_val[:, 0].detach().cpu().numpy(), truths[cls_idx].detach().cpu().numpy(), pred_score[cls_idx].detach().cpu().numpy()

    device = torch.device('cuda:0' if args.use_gpu else 'cpu')
    root_dset = args.coco_path
    year_dset = args.coco_year
    ckpt_pth = args.yolo_ckpt

    truth_list = np.array([])
    score_list = np.array([])
    aps = []
    num_gt_list = [0, 0]
    num_time_correct = 0

    class_strs = ['Person', 'Face']

    in_size = (416, 416)
    num_classes = 2
    # model = YOLOV2PedFace(in_size, num_classes).to(device)
    model = YOLOV2PedFaceLite(in_size, num_classes).to(device)
    model.load_state_dict(torch.load(ckpt_pth)['model_state_dict'])

    num_dset_split = 1
    transform = T.Compose([T.Resize(in_size), T.ToTensor()])

    with open('records/test_person.csv', 'w') as f_record:
        # f_record.write('IoU, Class predict, Class label, IoU correct, Class correct, Result, Process time, Time correct\n')
        f_record.write('IoU, IoU correct, Process time, Time correct, Time correct ratio, AP ratio\n')

    t_start = time.time()
    t_start_str = get_current_time_str()

    num_time_correct = 0
    num_total = 0
    num_total_iou = 0
    num_true_iou = 0
    num_tar = 0

    for split_idx in range(num_dset_split):
        # dset = YoloCOCODataset(root_dset, in_size, False, year_dset, transform, None, num_classes, True, True, num_dset_split, split_idx)
        dset = COCODataset(root=args.coco_path, year=args.coco_year, mode='val', img_size=416)
        loader = DataLoader(dset, collate_fn=custom_collate_fn, shuffle=False)

        for i, (img, _) in enumerate(loader):
            img = img[0].unsqueeze(0).to(device)
            pred = model(img)
            del img, pred
            torch.cuda.empty_cache()
            if i == 20:
                break

        for i, (img, ann) in enumerate(loader):
            # if i < len(dset) - 1:
            #     print(f'{split_idx+1}/{num_dset_split} - {i+1}/{len(dset)}', end='\r')
            # else:
            #     print(f'{split_idx + 1}/{num_dset_split} - {i + 1}/{len(dset)}')
            #     print()
            print(f'{split_idx + 1}/{num_dset_split} - {i + 1}/{len(dset)} ', end='')

            t_start_img = time.time()

            img = img[0].unsqueeze(0).to(device)
            ann = ann[0]
            num_tar += len(ann['bbox'])

            pred = model(img)

            outputs = get_output(pred, num_classes)

            t_end_img = time.time()
            t_img = t_end_img - t_start_img

            time_correct = t_img <= .2
            if time_correct:
                num_time_correct += 1
            num_total += 1

            if outputs is not None:
                # truth_list, score_list, ious = update_informations(outputs, ann, truth_list, score_list)
                ious, truths, scores = get_results(outputs, ann)
                with open('records/test_person.csv', 'a') as f_record:
                    for j in range(len(ious)):
                        num_total_iou += 1
                        if truths[j]:
                            num_true_iou += 1
                        truth_list = np.append(truth_list, np.array([truths[j]]))
                        score_list = np.append(score_list, np.array([scores[j]]))
                        # ap = average_precision_score(truth_list, score_list)
                        ap = average_precision(truth_list, score_list, num_tar)
                        if np.isnan(ap):
                            ap = 0
                        f_record.write(f'{ious[j]:.3f}, {truths[j]}, {t_img}, {time_correct}, {num_time_correct/num_total:.4f}, {ap}, {num_true_iou/num_total_iou:.3f}\n')
                # truth_list = np.append(truth_list, truths)
                # score_list = np.append(score_list, scores)

            # f_record.write(f'{0},person,{0},{False},{False},{True},{t_img},{time_correct},{num_time_correct/num_total:.4f}\n')


            '''
            img = img.squeeze().permute(1, 2, 0).detach().cpu().numpy()
            for b in bbox:
                b = b / 13 * 416
                y1, x1, y2, x2 = list(map(int, b))
                img = cv.rectangle(img.copy(), (x1, y1), (x2, y2), (0, 0, 255), 2)
            tar_bbox = ann['bbox']
            for b in tar_bbox:
                y1, x1, y2, x2 = list(map(int, b))
                img = cv.rectangle(img.copy(), (x1, y1), (x2, y2), (0, 255, 0), 2)
            plt.imshow(img)
            plt.show()
            exit()
            '''

            # del img, ann, pred, outputs
            # torch.cuda.empty_cache()

            # ap = average_precision_score(truth_list, score_list)
            ap = average_precision(truth_list, score_list, num_tar)
            print(f'(AP : {ap:.3f}) (# True IoU : {num_true_iou/num_total_iou:.3f})', end='\r')

            # plt.imshow(img[0].permute(1, 2, 0).detach().cpu().numpy())
            # plt.show()

        del dset, loader

    # Test metric.average_precision
    average_precision(truth_list, score_list, num_tar)
    # Test line end

    # ap = average_precision_score(truth_list, score_list)

    t_end = time.time()
    t_end_str = get_current_time_str()
    t = t_end - t_start

    with open('records/test_person.csv', 'a') as f_record:
        f_record.write('\nAverage Precision,Start,End,Total time(sec)\n')
        f_record.write(f'{ap},{t_start_str},{t_end_str},{t}\n')

    idx_sort = np.argsort(score_list, -1)[::-1]
    truth_list = truth_list[idx_sort]
    score_list = score_list[idx_sort]

    print()
    print(ap)

    precision, recall, _ = precision_recall_curve(truth_list, score_list)
    plt.plot(recall, precision)
    plt.show()



def test_age_gender(args):

    def get_output(age_super, age_subs, gender):
        age_super = age_super[0]
        age_subs = [age_sub[0] for age_sub in age_subs]
        gender = gender[0]

        age_super_idx = torch.argmax(age_super, dim=-1)
        age_sub_idx = [torch.argmax(i, dim=-1) for i in age_subs]

        age_score, _ = torch.max(age_subs[age_super_idx], dim=-1)
        gender_score, _ = torch.max(gender, dim=-1)

        age = (age_super_idx + 3) * 5 + age_sub_idx[age_super_idx]
        gender = torch.argmax(gender, dim=-1)

        age_score_list[age-15] = np.append(age_score_list[age-15], [age_score.detach().cpu().item()])
        gender_score_list[gender] = np.append(gender_score_list[gender], [gender_score.detach().cpu().item()])

        return age.detach().cpu().item(), gender.detach().cpu().item()

    def update_informations(age_predict, gender_predict, age_target, gender_target):
        age_truth = (age_target - 5 <= age_predict <= age_target + 5)
        gender_truth = gender_predict == gender_target

        age_truth_list[age_predict-15] = np.append(age_truth_list[age_predict-15], [age_truth])
        gender_truth_list[gender_predict] = np.append(gender_truth_list[gender_predict], [gender_truth])


    device = torch.device('cuda:0' if args.use_gpu else 'cpu')
    root_dset = args.afad_path
    ckpt_pth = args.agnet_ckpt

    gender_strs = ['Male', 'Female']

    model = AGNetSuperSub().to(device)
    model.load_state_dict(torch.load(ckpt_pth)['model_state_dict'])
    model.eval()

    dset = AFADDataset(root_dset, in_size=(224, 224), categorical=True)
    num_dset = 1000
    dset, _ = random_split(dset, [num_dset, len(dset) - num_dset])
    loader = DataLoader(dset, collate_fn=custom_collate_fn)

    f_record = open('records/test_age_gender.csv', 'w')
    f_record.write('Age predict, Age label, Gender predict, Gender label, Age correct, Gender correct, Process time, Time correct\n')

    t_start = time.time()
    t_start_str = get_current_time_str()

    age_cm = np.zeros((45, 45))
    gender_cm = np.zeros((2, 2))

    age_truth_list = [np.array([]) for _ in range(45)]
    age_score_list = [np.array([]) for _ in range(45)]

    gender_truth_list = [np.array([]) for _ in range(2)]
    gender_score_list = [np.array([]) for _ in range(2)]

    num_time_correct = 0

    for i, (img, label) in enumerate(loader):
        t_start_img = time.time()

        img = img[0].unsqueeze(0).to(device)
        label = label[0]
        age_tar, gender_tar = label['age'].item(), label['gender'].item()

        age_super_pred, age_subs_pred, gender_pred = model(img)
        age_pred, gender_pred = get_output(age_super_pred, age_subs_pred, gender_pred)
        update_informations(age_pred, gender_pred, age_tar, gender_tar)

        age_correct = age_truth_list[age_pred-15][-1]
        gender_correct = gender_truth_list[gender_pred][-1]

        t_end_img = time.time()
        t_img = t_end_img - t_start_img

        time_correct = t_img <= .6
        if time_correct:
            num_time_correct += 1

        f_record.write(f'{str(age_pred)}, {str(age_tar)}, {str(gender_strs[gender_pred])}, {str(gender_strs[gender_tar])}, {str(age_correct)}, {str(gender_correct)}, {t_img}, {time_correct}\n')

        age_pred_idx = age_pred - 15
        age_tar_idx = age_tar - 15
        if age_correct:
            age_cm[age_tar_idx, age_tar_idx] += 1
        else:
            age_cm[age_pred_idx, age_tar_idx] += 1

        gender_cm[gender_pred, gender_tar] += 1

        print(f'{i+1} / {len(dset)}', end='\r')

    age_ap_list = []
    gender_ap_list = []

    for i in range(45):
        if len(age_truth_list[i]) > 0:
            print(i, age_truth_list[i], age_score_list[i])
            ap = average_precision_score(age_truth_list[i], age_score_list[i])
            if ap != ap:
                ap = 0
            age_ap_list.append(ap)

    for i in range(2):
        gender_ap_list.append(average_precision_score(gender_truth_list[i], gender_score_list[i]))

    print(age_ap_list)

    age_map = sum(age_ap_list) / len(age_ap_list)
    gender_map = sum(gender_ap_list) / len(gender_ap_list)

    # age_precisions = np.zeros(45)
    # age_recalls = np.zeros(45)
    # for i in range(45):
    #     tp = age_cm[i, i]
    #     if tp > 0:
    #         fp = age_cm[i].sum() - tp
    #         fn = age_cm[:, i].sum() - tp
    #         age_precisions[i] = tp / (tp + fp)
    #         age_recalls[i] = tp / (tp + fn)

    # gender_precisions = np.array([gender_cm[i, i] / gender_cm[i].sum() for i in range(2)])

    # age_valid_idx = age_cm.sum(axis=-1) != 0
    # age_precision = age_precisions[age_valid_idx].mean()

    # gender_precision = gender_precisions.mean()

    t_end = time.time()
    t_end_str = get_current_time_str()
    t = t_end - t_start

    f_record.write('\nAge mean Average Precision, Gender mean Average Precision, Time pass ratio(%)\n')
    f_record.write(f'{age_map:.5f}, {gender_map:.5f}, {num_time_correct / num_dset * 100:.3f}\n')
    f_record.write('Start, End, Total time(sec)\n')
    f_record.write(f'{t_start_str}, {t_end_str}, {t}\n')

    f_record.close()

    if args.show_matrix:
        for c in age_cm:
            for i in c:
                print(f'{int(i):02d}', end=' ')
            print()


def test(args):
    mode = args.mode

    if mode == 'person':
        print('Mode: Person')
        test_person(args)

    elif mode == 'age_gender':
        print('Mode: Age & Gender')
        test_age_gender(args)


if __name__ == '__main__':
    yolo_ckpt = './models/weights/YOLOV2PedFaceLite_0.600iou.ckpt'
    yolo_ckpt = './models/weights/YOLOV2PedFaceLite_52epoch_0.000010lr_2.743loss(train)_3.068loss_2.263loss(box)_0.454loss(obj)_0.323loss(noobj)_0.029loss(cls)_0.569iou_0.991acc(cls).ckpt'

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, type=str)  # 'person' or 'age_gender'
    parser.add_argument('--device', required=False, type=str, default='cuda:0')
    parser.add_argument('--coco_path', required=False, type=str, default='./datasets/COCO/')
    parser.add_argument('--coco_year', required=False, type=str, default='2014')
    parser.add_argument('--afad_path', required=False, type=str, default='./datasets/AFAD-Full/Test/')
    parser.add_argument('--yolo_ckpt', required=False, type=str, default=yolo_ckpt)
    parser.add_argument('--agnet_ckpt', required=False, type=str, default='./models/weights/AGNetSuperSub_temp.ckpt')
    parser.add_argument('--conf_thresh', required=False, type=float, default=.3)
    parser.add_argument('--nms_thresh', required=False, type=float, default=.3)
    parser.add_argument('--iou_thresh', required=False, type=float, default=2/3)
    parser.add_argument('--show_matrix', required=False, type=bool, default=False)
    parser.add_argument('--use_gpu', required=False, type=bool, default=True)
    args = parser.parse_args()

    test(args)

