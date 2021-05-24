from mmdet.apis import init_detector, inference_detector, show_result_ins, show_result_pyplot
import torch
import mmcv

if __name__ == '__main__':
    config_file = '../configs/kirito/point_rend_instaboost_x50_32x4d_fpn_mstrain_4x.py'
    # download the checkpoint from model zoo and put it in `checkpoints/`
    checkpoint_file = '../train_ws/point_rend_instaboost_x50_mstrain_4x/epoch_48.pth'

    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    # test a single image
    image = mmcv.imread('00083.jpg')
    result = inference_detector(model, image)

    #show_result_ins(image, result, model.CLASSES, score_thr=0.25, out_file="demo_out.jpg")
    show_result_pyplot(model, image, result, score_thr=0.3)
