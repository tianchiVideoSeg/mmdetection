_base_ = [
    '../_base_/models/mask_rcnn_se_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(
    pretrained='checkpoints/se_resnext101_64x4d-f9926f93.pth',
    backbone=dict(block='SEResNeXtBottleneck', layers=[3, 4, 23, 3], groups=64))
