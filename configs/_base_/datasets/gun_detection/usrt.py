# dataset settings
train_dataset_type = "GunDataset"
val_dataset_type = "GunDatasetHOI"
dataset_root = "data/usrt/"

img_norm_cfg = dict(
    mean=[110.1658279, 103.20675178, 92.15120447], std=[36.85286008, 38.51197062, 45.77433122], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu = 2,
    workers_per_gpu = 2,
    train = dict(
        type = train_dataset_type,
        data_root = dataset_root,
        ann_file = "annotation_detection/annotations_train.json",
        img_prefix = "images/",
        pipeline = train_pipeline
    ),
    val = dict(
        type = val_dataset_type,
        data_root = dataset_root,
        ann_file = "annotation_detection/annotations_val.json",
        img_prefix = "images/",
        pipeline = test_pipeline
    ),
    test = dict(
        type = val_dataset_type,
        data_root = dataset_root,
        ann_file = "annotation_detection/annotations_test.json",
        img_prefix = "images/",
        pipeline = test_pipeline
    )
)