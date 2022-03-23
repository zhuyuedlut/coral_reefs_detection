import os

TRAIN_PATH = os.path.abspath('../../datasets/coral_reefs_detection/datasets/train_images')
DATASET_PATH = os.path.join(TRAIN_PATH, 'processed')

DATA = {
    "CLASSES": ["starfish"],
    "NUM": 1
}

# model
MODEL = {"ANCHORS": [[(1.25, 1.625), (2.0, 3.75), (4.125, 2.875)],  # Anchors for small obj
                     [(1.875, 3.8125), (3.875, 2.8125), (3.6875, 7.4375)],  # Anchors for medium obj
                     [(3.625, 2.8125), (4.875, 6.1875), (11.65625, 10.1875)]],  # Anchors for big obj
         "STRIDES": [8, 16, 32],
         "ANCHORS_PER_SCLAE": 3
         }

TRAIN = {
    "TRAIN_IMG_SIZE": 448,
    "AUGMENT": True,
    "BATCH_SIZE": 4,
    "MULTI_SCALE_TRAIN": True,
    "IOU_THRESHOLD_LOSS": 0.5,
    "EPOCHS": 50,
    "NUMBER_WORKERS": 4,
    "MOMENTUM": 0.9,
    "WEIGHT_DECAY": 0.0005,
    "LR_INIT": 1e-4,
    "LR_END": 1e-6,
    "WARMUP_EPOCHS": 2  # or None
}
