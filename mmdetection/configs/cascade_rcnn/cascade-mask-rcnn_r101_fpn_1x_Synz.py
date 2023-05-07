_base_ = './cascade-mask-rcnn_r50_fpn_1x_Synz.py'
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))
