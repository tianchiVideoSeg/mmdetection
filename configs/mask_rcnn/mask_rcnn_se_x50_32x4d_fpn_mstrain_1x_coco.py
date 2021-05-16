_base_ = [
    '../_base_/models/mask_rcnn_se_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(
    pretrained='checkpoints/se_resnext50_32x4d-a260b3a4.pth',
    backbone=dict(block='SEResNeXtBottleneck', groups=32))
