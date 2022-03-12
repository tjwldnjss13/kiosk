import os
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as T
import PIL.Image as Image

from pycocotools.coco import COCO

import cv2 as cv
import matplotlib.pyplot as plt


class COCODataset(data.Dataset):
    def __init__(
            self,
            root,
            year,
            mode,
            img_size=None
    ):
        super().__init__()
        self.root = root
        self.year = year
        self.mode = mode
        if isinstance(img_size, int):
            self.img_size = (img_size, img_size)
        elif isinstance(img_size, tuple) or isinstance(img_size, list) or img_size is None:
            self.img_size = img_size
        else:
            print('Wrong img_size format.')
            exit()

        self.anns = self._get_annotation()
        # self.anns = self.anns[:10]

    def _get_annotation(self):
        ann_list = []

        ann_pth = os.path.join(self.root, 'annotations', f'instances_{self.mode}{self.year}.json')
        coco = COCO(ann_pth)
        ids = list(coco.imgs.keys())

        for i, id in enumerate(ids):
            ann_dict = {}
            bbox_list = []
            category_id_list = []

            img_dict = coco.loadImgs(id)[0]
            img_fn = img_dict['file_name']
            h = img_dict['height']
            w = img_dict['width']
            del img_dict
            img_pth = os.path.join(self.root, 'images', f'{self.mode}{self.year}', img_fn)
            ann_ids = coco.getAnnIds(id)

            for ann_id in ann_ids:
                ann = coco.loadAnns(ann_id)[0]
                category_id = ann['category_id']
                iscrowd = ann['iscrowd']

                if category_id == 1 and not iscrowd:
                    category_id = 0
                    bbox = ann['bbox']

                    if self.img_size is not None:
                        h_new, w_new = self.img_size
                        x1 = bbox[0] * w_new / w
                        y1 = bbox[1] * h_new / h
                        x2 = x1 + bbox[2] * w_new / w
                        y2 = y1 + bbox[3] * h_new / h
                        bbox = [y1, x1, y2, x2]

                    bbox_list.append(bbox)
                    category_id_list.append(category_id)
                    del category_id, bbox
                del ann

            ann_dict['img_pth'] = img_pth
            ann_dict['bbox'] = np.asarray(bbox_list)
            ann_dict['category_id'] = np.asarray(category_id_list)

            ann_list.append(ann_dict)

        return ann_list

    def __getitem__(self, idx):
        ann = self.anns[idx]
        img = Image.open(ann['img_pth']).convert('RGB')
        img = T.Compose([T.Resize(self.img_size), T.ToTensor()])(img)

        # plt.imshow(img.permute(1, 2, 0).numpy())
        # plt.show()

        ann_ret = {}
        ann_ret['bbox'] = torch.Tensor(ann['bbox'])
        ann_ret['class'] = torch.Tensor(ann['category_id'])

        return img, ann_ret

    def __len__(self):
        return len(self.anns)



if __name__ == '__main__':
    root = os.path.abspath('../datasets/COCO')
    year = '2017'
    mode = 'val'
    img_size = 416

    dset = COCODataset(
        root=root,
        year=year,
        mode=mode,
        img_size=img_size
    )
    for i in range(len(dset)):
        dset[i]