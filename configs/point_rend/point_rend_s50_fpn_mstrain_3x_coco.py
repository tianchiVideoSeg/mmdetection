_base_ = '../resnest/mask_rcnn_s50_fpn_syncbn-backbone+head_mstrain_1x_coco.py'
model = dict(
    type='PointRend',
    roi_head=dict(
        type='PointRendRoIHead',
        mask_roi_extractor=dict(
            type='GenericRoIExtractor',
            aggregation='concat',
            roi_layer=dict(
                _delete_=True, type='SimpleRoIAlign', output_size=14),
            out_channels=256,
            featmap_strides=[4]),
        mask_head=dict(
            _delete_=True,
            type='CoarseMaskHead',
            num_fcs=2,
            in_channels=256,
            conv_out_channels=256,
            fc_out_channels=1024,
            num_classes=80,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)),
        point_head=dict(
            type='MaskPointHead',
            num_fcs=3,
            in_channels=256,
            fc_channels=256,
            num_classes=80,
            coarse_pred_each_layer=True,
            loss_point=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),
    # model training and testing settings
    train_cfg=dict(
        rcnn=dict(
            mask_size=7,
            num_points=14 * 14,
            oversample_ratio=3,
            importance_sample_ratio=0.75)),
    test_cfg=dict(
        rcnn=dict(
            subdivision_steps=5,
            subdivision_num_points=28 * 28,
            scale_factor=2)))
# dataset settings
dataset_type = 'CocoDataset'
data_root = '/home/baikai/Desktop/AliComp/datasets/PreRoundData/'
# use ResNeSt img_norm
img_norm_cfg = dict(
    mean=[123.68, 116.779, 103.939], std=[58.393, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize',
         img_scale=(640, 384),
         multiscale_mode='value',
         keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        #img_scale=(1333, 800),
        img_scale=(640, 384),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'coco_annos_train.json',
        img_prefix=data_root + 'JPEGImages/',
        pipeline=train_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'coco_annos_val.json',
        img_prefix=data_root + 'JPEGImages/',
        pipeline=test_pipeline))
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# learning policy
lr_config = dict(step=[28, 34])
runner = dict(type='EpochBasedRunner', max_epochs=36)
load_from = 'checkpoints/mask_rcnn_s50_fpn_syncbn-backbone+head_mstrain_1x_coco_20200926_125503-8a2c3d47.pth'
work_dir = 'train_ws/point_rend_s50_fpn_mstrain_1x_coco'
