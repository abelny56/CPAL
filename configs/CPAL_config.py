_base_ = [
    './_base_/models/hamburger.py', './_base_/datasets/mydata.py',
    './_base_/default_runtime.py', './_base_/schedules/schedule_40k.py'
]
img_scale = (480, 640)
crop_size = (480, 640)
num_class = 5
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
depth_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], data_name="depth")
pretrained = 'https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_h_jointto22k_384.pth'
model = dict(
    backbone=dict(
        _delete_=True,
        type='InternImage',
        core_op='DCNv3',
        channels=320,  
        depths=[6, 6, 32, 6],
        groups=[10, 20, 40, 80],
        mlp_ratio=4.,
        drop_path_rate=0.5,
        norm_layer='LN',
        layer_scale=None,
        offset_scale=1.0,
        post_norm=False,
        dw_kernel_size=5, # for InternImage-H/G
        res_post_norm=True, # for InternImage-H/G
        level2_post_norm=True, # for InternImage-H/G
        level2_post_norm_block_ids=[5, 11, 17, 23, 29], # for InternImage-H/G
        center_feature_scale=True, # for InternImage-H/G
        with_cp=True,
        out_indices=(0, 1, 2, 3),
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    decode_head=dict(num_classes=num_class, in_channels=[320, 640, 1280, 2560]),
    test_cfg=dict(mode='whole'))
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadDepthFromFile', depth_type='repeat1', depth_channels='grayscale'), 
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='NormalizeData', **depth_norm_cfg),
            dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'], additional_meta_keys=['channels'])
        ])
]
optimizer = dict(
    _delete_=True, type='AdamW', lr=0.00002, betas=(0.9, 0.999), weight_decay=0.05,
    constructor='CustomLayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=50, layer_decay_rate=0.95,
                       depths=[6, 6, 32, 6], offset_lr_scale=1.0))
lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)
data=dict(samples_per_gpu=2,  
          val=dict(pipeline=test_pipeline),
          test=dict(pipeline=test_pipeline))
runner = dict(type='IterBasedRunner', max_iters=20000)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=0.1, norm_type=2))
checkpoint_config = dict(by_epoch=False, interval=5000, max_keep_ckpts=1)
evaluation = dict(interval=20000, metric='mIoU', save_best='mIoU') 
