def mean_average_precision(
        pred_box_list,
        tar_box_list,
        pred_label_list,
        tar_label_list,
        iou_thresh,
        num_classes
):
    ious_list = []
    tp_list = []
    fp_list = []
    num_tar_list = []

    for i in range(len(pred_box_list)):
        print(f'{i+1}/{len(pred_box_list)}', end='\r')

        pred_boxes = pred_box_list[i]
        tar_boxes = tar_box_list[i]
        pred_label = pred_label_list[i]
        tar_label = tar_label_list[i]

        for c in range(num_classes):
            pred_idx = pred_label == c
            tar_idx = tar_label == c

            tar_covered = torch.zeros(len(tar_boxes[tar_idx]))
            tp = torch.zeros(len(pred_boxes[pred_idx]))
            fp = torch.zeros(len(pred_boxes[pred_idx]))

            if len(tar_boxes[tar_idx]) > 0 and len(pred_boxes[pred_idx]) > 0:
                ious_pred_tar = calculate_ious_grid(pred_boxes[pred_idx], tar_boxes[tar_idx], box_format='xyxy')
                ious_pred_tar, idx_pred_tar = torch.sort(ious_pred_tar, dim=-1, descending=True)
                ious_pred_tar = ious_pred_tar[..., 0]
                idx_pred_tar = idx_pred_tar[..., 0]
                # ious_pred_tar, idx_ord = torch.sort(ious_pred_tar, dim=0, descending=True)
                # idx_pred_tar = idx_pred_tar[idx_ord]

                ious_list.append(ious_pred_tar)

                for j, iou in enumerate(ious_pred_tar):
                    if iou > iou_thresh:
                        if tar_covered[idx_pred_tar[j]] == 0:
                            tp[j] = 1
                            tar_covered[idx_pred_tar[j]] = 1
                        else:
                            fp[j] = 1
                    else:
                        fp[j] = 1
            else:
                fp = torch.ones(len(pred_boxes[pred_idx]))

            tp_list.append(tp)
            fp_list.append(fp)
            num_tar_list.append(len(tar_boxes[tar_idx]))

    ious = torch.cat(ious_list)
    tp = torch.cat(tp_list)
    fp = torch.cat(fp_list)

    ious, idx_ious = torch.sort(ious, descending=True)
    tp = tp[idx_ious]
    fp = fp[idx_ious]
    print('tp :', tp)
    print('fp :', fp)

    tp_cumsum = torch.cumsum(tp, dim=0)
    fp_cumsum = torch.cumsum(fp, dim=0)

    print('tp_cumsum :', tp_cumsum)
    print('fp_cumsum :', fp_cumsum)

    precisions = torch.divide(tp_cumsum, (tp_cumsum + fp_cumsum + 1e-12))
    precisions = torch.cat([torch.tensor([1]), precisions])
    recalls = tp_cumsum / (sum(num_tar_list) + 1e-12)
    recalls = torch.cat([torch.tensor([0]), recalls])
    ap = torch.trapz(precisions, recalls)

    print('num_tar :', num_tar_list)

    print('precisions :', precisions)
    print('recalls :', recalls)
    print('ap :', ap)

    return ap