import warnings

warnings.filterwarnings("ignore")

import ast
import os
import json
import pandas as pd
from shutil import copyfile
from tqdm.notebook import tqdm
tqdm.pandas()
from sklearn.model_selection import GroupKFold
from config import yolov3_config

TRAIN_PATH = yolov3_config.TRAIN_PATH
DATASET_PATH = yolov3_config.DATASET_PATH

def get_bbox(annots):
    bboxes = [list(annot.values()) for annot in annots]
    return bboxes

def get_path(row):
    row['image_path'] = f'{TRAIN_PATH}/train_images/video_{row.video_id}/{row.video_frame}.jpg'
    return row

if __name__ == "__main__":
    df = pd.read_csv(TRAIN_PATH + "/train.csv")
    df["num_bbox"] = df['annotations'].apply(lambda x: str.count(x, 'x'))
    df_train = df[df["num_box"] > 0]

    # literal_eval is security method to parse string to original type
    # 将train.csv中的annotations中的list字符串转换为真正的list
    df_train['annotations'] = df_train['annotations'].progress_apply(lambda x: ast.literal_eval(x))
    # 将annotations中的dict list中的元素拿出来，组成 [x, y, with, height]的list
    df_train['bboxes'] = df_train.annotations.progress_apply(get_bbox)

    # #Images resolution
    df_train["width"] = 1280
    df_train["height"] = 720

    df_train = df_train.progress_apply(get_path, axis=1)

    kf = GroupKFold(n_splits=5)
    df_train = df_train.reset_index(drop=True)
    df_train['fold'] = -1
    for fold, (train_idx, val_idx) in enumerate(
            kf.split(df_train, y=df_train.video_id.tolist(), groups=df_train.sequence)):
        df_train.loc[val_idx, 'fold'] = fold

    SELECTED_FOLD = 4

    for i in tqdm(range(len(df_train))):
        row = df_train.loc[i]
        if row.fold != SELECTED_FOLD:
            copyfile(f'{row.image_path}', f'{DATASET_PATH}/train2017/{row.image_id}.jpg')
        else:
            copyfile(f'{row.image_path}', f'{DATASET_PATH}/val2017/{row.image_id}.jpg')

    print(f'Number of training files: {len(os.listdir(f"{DATASET_PATH}/train2017/"))}')
    print(f'Number of validation files: {len(os.listdir(f"{DATASET_PATH}/val2017/"))}')
    print('Split Finish !!!')

    train_annot_json = dataset2coco(df_train[df_train.fold != SELECTED_FOLD], f"{DATASET_PATH}/train2017/")
    val_annot_json = dataset2coco(df_train[df_train.fold == SELECTED_FOLD], f"{DATASET_PATH}/val2017/")

    # Save converted annotations
    save_annot_json(train_annot_json, f"{DATASET_PATH}/annotations/train.json")
    save_annot_json(val_annot_json, f"{DATASET_PATH}/annotations/valid.json")

