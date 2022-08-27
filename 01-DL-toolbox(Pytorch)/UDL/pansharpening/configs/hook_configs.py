# checkpoint saving
# checkpoint_config = dict(interval=1)
checkpoint_config = dict(type='ModelCheckpoint', indicator='loss')
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable

# dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = "D:/ProjectSets/NDA/UDL/UDL/results/pansharpening/wv3/FusionNet/Test/model_2022-04-02-12-02-55/275.pth.tar"
resume_from = "D:/ProjectSets/NDA/UDL/UDL/results/pansharpening/wv3/FusionNet/Test/model_2022-04-02-12-02-55/275.pth.tar"
workflow = [('train', 1)]

# optimizer
optimizer = dict(type='Adam', lr=3e-4)
optimizer_config = dict(grad_clip=None)
lr_config = None
# learning policy
runner = dict(type='EpochBasedRunner', max_epochs=275)
