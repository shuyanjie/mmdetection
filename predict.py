from mmdet.apis import DetInferencer

# 初始化模型
# ssd300-config/ssd300_coco.py\ ssd300-train/epoch_12.pth
# faster_rcnn-config/faster-rcnn_r50_fpn_1x_coco.py\ faster_rcnn-train/epoch_12.pth
# cascade_rcnn-config/cascade-rcnn_r50_fpn_1x_coco.py\ cascade_rcnn-train/epoch_12.pth
inferencer = DetInferencer(model='cascade_rcnn-config/cascade-rcnn_r50_fpn_1x_coco.py', weights='cascade_rcnn-train/epoch_12.pth', device='cuda:0')

# 推理示例图片
inferencer('./data/coco/val2017/image_4425.png',out_dir='outputs/', no_save_pred=False)