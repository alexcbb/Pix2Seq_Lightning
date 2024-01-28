import cv2

import torch
from torch.utils.data import Dataset
from glob import glob
import json

class YCBDataset(Dataset):
    def __init__(
            self, 
            path_to_dataset, 
            transforms=None, 
            tokenizer=None
        ):
        scenes_train = glob(path_to_dataset + "/*/", recursive = True)

        train_pose_gt_files = []
        train_bbox_gt_files = []
        for scene in scenes_train:
            train_pose_gt_files.append(scene + "/scene_gt.json")   
            train_bbox_gt_files.append(scene + "/scene_gt_info.json")    

        self.labels = []
        for i in range(len(train_pose_gt_files)):
            f = open(train_pose_gt_files[i])
            gt_pose = json.load(f)
            f.close()
            g = open(train_bbox_gt_files[i])
            gt_bbox = json.load(g)
            g.close()
            for scene in gt_pose.keys():
                for obj in range(len(gt_pose[scene])):
                    bbox = gt_bbox[scene][obj]['bbox_obj']
                    bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]] # xmin, ymin, xmax, ymax
                    obj_class= gt_pose[scene][obj]['obj_id']
                    obj_trans = gt_pose[scene][obj]['cam_t_m2c']
                    obj_rot = gt_pose[scene][obj]['cam_R_m2c']
                    img_path = scenes_train[i] + 'rgb/' + str(scene).zfill(6) + '.png'
                    self.labels.append(
                        {
                        "bbox" : bbox, 
                        "obj_class": obj_class,
                        "obj_trans": obj_trans,
                        "obj_rot": obj_rot,
                        "img_path": img_path
                        }
                    )
        self.transforms = transforms
        self.tokenizer = tokenizer
        print(f"Load dataset at path {path_to_dataset} with size: {len(self.labels)}")

    def __getitem__(self, idx):
        sample = self.labels[idx]
        img_path = sample['img_path']

        img = cv2.imread(img_path)[..., ::-1]
        obj_class = sample['obj_class']
        bboxes = sample['bbox']

        if self.transforms is not None:
            transformed = self.transforms(**{
                'image': img
            })
            img = transformed['image']

        img = torch.FloatTensor(img).permute(2, 0, 1)

        # Converts the labels to a sequence for the autoregressive decoder
        if self.tokenizer is not None:
            seqs = self.tokenizer(obj_class, bboxes)
            seqs = torch.LongTensor(seqs)
            return img, seqs

        return img, obj_class, bboxes

    def __len__(self):
        return len(self.labels)

