import cv2
import numpy as np
import os
import random
import torch

import config.yolov3_config as cfg

from torch.utils.data import Dataset, DataLoader


class CoralReefDataset(Dataset):
    def __init__(self, anno_file_type, img_size=416):
        self.img_size = img_size
        self.classes = cfg.DATA["CLASSES"]
        self.num_classes = len(self.classes)
        self.class_to_id = dict(zip(self.classes, range(self.num_classes)))
        self.__annotations = self.__load_annotations(anno_file_type)

    def __len__(self):
        return len(self.__annotations)

    def __getitem__(self, item):


    def __load_annotations(self, anno_type):
        assert anno_type in ['train', 'val'], "You must choice one of the 'train' or 'test' for anno_type parameter"
        anno_path = f"{cfg.DATASET_PATH}/annotations/{anno_type}.json"


if __name__ == "__main__":
