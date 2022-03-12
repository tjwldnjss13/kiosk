import torch


from utils.pytorch_util import calculate_ious_grid


def categorical_accuracy(predict, target):
    pred = torch.argmax(predict, dim=-1)
    tar = torch.argmax(target, dim=-1)

    correct = pred == tar
    correct = correct.reshape(-1)

    acc = correct.sum() / len(correct)

    return acc.detach().cpu().item()


def confidence_accuracy(predict, target, threshold):
    pred = predict >= threshold
    tar = target.bool()

    correct = pred == tar
    correct = correct.reshape(-1)

    acc = correct.sum() / len(correct)

    return acc.detach().cpu().item()


def average_precision(
        truth_list,
        score_list,
        num_target
):
    tp = torch.zeros(len(truth_list))
    fp = torch.zeros(len(truth_list))

    truths = torch.Tensor(truth_list)
    scores = torch.Tensor(score_list)

    scores, idx_sort = torch.sort(scores, descending=True)
    truths = truths[idx_sort]

    tp[truths == 1] = 1
    fp[truths == 0] = 1

    tp_cumsum = torch.cumsum(tp, dim=0)
    fp_cumsum = torch.cumsum(fp, dim=0)

    precisions = torch.divide(tp_cumsum, (tp_cumsum + fp_cumsum + 1e-12))
    precisions = torch.cat([torch.tensor([1]), precisions])

    recalls = torch.divide(tp_cumsum, (num_target + 1e-12))
    recalls = torch.cat([torch.tensor([0]), recalls])

    ap = torch.trapz(precisions, recalls)

    # print()
    # print('truths :', truths)
    # print('scores :', scores)
    # print('num_target :', num_target)
    # print('tp :', tp)
    # print('fp :', fp)
    # print('tp_cumsum :', tp_cumsum)
    # print('fp_cumsum :', fp_cumsum)
    # print('precisions :', precisions)
    # print('recalls :', recalls)
    # print('ap :', ap)
    # exit()

    return ap





if __name__ == '__main__':
    pred = torch.Tensor([.4, .6, .1, .7])
    tar = torch.Tensor([0, 1, 0, 1])

    acc = confidence_accuracy(pred, tar, .3)
    print(acc)
























