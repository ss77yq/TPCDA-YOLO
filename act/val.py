from ultralytics import YOLOv10
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'


if __name__ == '__main__':
    # 加载模型
    # model = YOLOv10('yolov10n.pt')  # 加载官方模型
    model = YOLOv10('/runs/detect/train_Foggy_54.6%/weights/best.pt')  # 加载自定义模型

    # 官方验证
    model.val(
              data='../ultralytics/cfg/datasets/VOC.yaml',
              cfg='../ultralytics/cfg/default.yaml',
              # # RTTS
              # data='../ultralytics/cfg/datasets/VOC_RTTS.yaml',
              # cfg='../ultralytics/cfg/default_RTTS.yaml',
              name='val_Foggy_54.6%',
              batch=1)

    # # 验证模型
    # metrics = model.val()  # 无需参数，数据集和设置记忆
    # metrics.box.map    # map50-95你
    # metrics.box.map50  # map50
    # metrics.box.map75  # map75
    # metrics.box.maps   # 包含每个类别的map50-95列表

