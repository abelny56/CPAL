dataset_type = 'mydata'  
data_root = 'data/PST'  
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
depth_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], data_name="depth")

img_scale = (480, 640)  
crop_size = (480, 640) 
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadDepthFromFile', depth_type='repeat1', depth_channels='grayscale'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=img_scale, ratio_range=(0.5, 2.0), apply_to_channels=['depth']),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='NormalizeData', **depth_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='WithDepthFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'], additional_meta_keys=['channels'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadDepthFromFile', depth_type='repeat', depth_channels='grayscale'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='NormalizeData', **depth_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'], additional_meta_keys=['channels'])
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images',
        depth_dir='images',
        ann_dir='annotations',
        pipeline=train_pipeline,
        split="splits/train.txt"),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images',
        depth_dir='images',
        ann_dir='annotations',
        pipeline=test_pipeline,
        split="splits/val.txt"),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images',
        depth_dir='images',
        ann_dir='annotations',
        pipeline=test_pipeline,
        split="splits/test.txt"))
