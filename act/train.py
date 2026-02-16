from ultralytics import YOLOv10
import os
import torch
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'



if __name__ == '__main__':
    # 建立模型（加载预训练权重）
    model = YOLOv10("../ultralytics/cfg/models/v10/yolov10m_DA.yaml")
    # model = YOLOv10("../ultralytics/cfg/models/v10/yolov10m_DA_RTTS.yaml")
    model.load("yolov10m.pt")

    # # 断点续训
    # model = YOLOv10("runs/detect/train/weights/last.pt")

    # train
    model.train(
                data='../ultralytics/cfg/datasets/VOC.yaml',
                # data='../ultralytics/cfg/datasets/VOC_RTTS.yaml',
                # data='../ultralytics/cfg/datasets/VOC_Rain.yaml',
                # cfg='../ultralytics/cfg/default_RTTS.yaml',
                save=True,
                name="train",
                imgsz=640,
                epochs=150,
                batch=8,
                workers=8,
                device='0',
                # 优化器默认为auto
                optimizer='SGD',
                resume=True
                )

    # # 追加训练
    # model = YOLOv10("../act/runs/detect/train4/weights/last.pt")
    # # 训练模型
    # model.train(data="../ultralytics/cfg/datasets/VOC_RTTS.yaml",
    #             epochs=200,
    #             device=0,
    #             name="train4_next",
    #             imgsz=640,
    #             batch=8,
    #             workers=8,
    #             # 优化器默认为auto
    #             optimizer='SGD',
    #             pretrained=True
    #             )
