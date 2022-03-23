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


def dataset2coco(df, dest_path):
    global annotion_id
    annotations_json = {
        "info": [],
        "licenses": [],
        "categories": [],
        "images": [],
        "annotations": []
    }

    info = {
        "year": "2021",
        "version": "1",
        "description": "COTS dataset - COCO format",
        "contributor": "",
        "url": "https://kaggle.com",
        "date_created": "2021-11-30T15:01:26+00:00"
    }
    annotations_json["info"].append(info)

    lic = {
        "id": 1,
        "url": "",
        "name": "Unknown"
    }
    annotations_json["licenses"].append(lic)

    classes = {"id": 0, "name": "starfish", "supercategory": "none"}
    annotations_json["categories"].append(classes)

    for ann_row in df.itertuples():
        images = {
            "id": ann_row[0],
            "license": 1,
            "file_name": ann_row.image_id + '.jpg',
            "height": ann_row.height,
            "width": ann_row.width,
            "date_captured": "2021-11-30T15:01:26+00:00"
        }

        annotations_json["images"].append(images)

        bbox_list = ann_row.bboxes

        for bbox in bbox_list:
            b_width, b_height = bbox[2], bbox[3]

            # if some boxes in COTS are outside the image height and width
            if (bbox[0] + bbox[2] > 1280):
                b_width = bbox[0] - 1280
            if (bbox[1] + bbox[3] > 720):
                b_height = bbox[1] - 720

            image_annotations = {
                "id": annotion_id,
                "image_id": ann_row[0],
                "category_id": 0,
                "bbox": [bbox[0], bbox[1], b_width, b_height],
                "area": bbox[2] * bbox[3],
                "segmentation": [],
                "iscrowd": 0
            }

            annotion_id += 1
            annotations_json["annotations"].append(image_annotations)

        print(f"Dataset COTS annotation to COCO json format completed! Files: {len(df)}")
        return annotations_json


def save_annot_json(json_annotation, filename):
    with open(filename, 'w') as f:
        output_json = json.dumps(json_annotation)
        f.write(output_json)


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
    # 将对应的图片的path对应付给df_train的row
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
