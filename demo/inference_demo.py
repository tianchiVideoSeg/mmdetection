from mmdet.apis import init_detector, inference_detector, show_result_pyplot, show_result_ins
import torch
import mmcv
import ttach as tta

transforms = tta.Compose(
    [
        tta.HorizontalFlip(),
        tta.Scale(scales=[1, 1.25, 1.5]),
    ]
)


config_file = '../configs/solov2/solov2_x101_dcn_fpn_8gpu_3x.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file = '../solov2_x101_dcn_fpn_8gpu_3x/epoch_37.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image
image = mmcv.imread('/home/baikai/Desktop/AliComp/datasets/PreRoundData/JPEGImages/638543/00053.jpg')
image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)

masks = []
labels = []
for transformer in transforms: # custom transforms or e.g. tta.aliases.d4_transform() 
    
    # augment image
    augmented_image = transformer.augment_image(image)
    augmented_image = augmented_image.squeeze(0).permute(1, 2, 0).numpy()
    
    # pass to model
    model_output = inference_detector(model, augmented_image)
    
    # reverse augmentation for mask and label
    deaug_mask = transformer.deaugment_mask(model_output[0])
    deaug_label = transformer.deaugment_label(model_output[1])
    
    # save results
    labels.append(deaug_mask)
    masks.append(deaug_label)
    
# reduce results as you want, e.g mean/max/min
result = []
result[0] = mean(masks)
result[1] = mean(labels)

show_result_ins(image, result, model.CLASSES, score_thr=0.25, out_file="demo_out.jpg")
