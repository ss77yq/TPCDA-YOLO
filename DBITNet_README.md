# QTNet训练代码

## 1.训练

**Step 1:** Training the QTNet and Preparing the data of translation image

```python
# train the model of normal_to_adverse
python DBITNet_train.py --mode normal_to_adverse --input_dir E:/Datasets/heavy_weather/Cityscapes_Nomal_Foggy/Normal_train/
                      --gt_dir E:/Datasets/heavy_weather/Cityscapes_Nomal_Foggy/Foggy_train/
python DBITNet_train.py --mode normal_to_adverse --input_dir F:/Datasets/heavy_weather/Cityscapes_Nomal_Foggy/Normal_train/ --gt_dir F:/Datasets/heavy_weather/Cityscapes_Nomal_Foggy/Rain_train/
# train the model of adverse_to_normal
python DBITNet_train.py --mode adverse_to_normal --input_dir E:/Datasets/heavy_weather/Cityscapes_Nomal_Foggy/Foggy_train/ \
                      --gt_dir E:/Datasets/heavy_weather/Cityscapes_Nomal_Foggy/Normal_train/
python DBITNet_train.py --mode adverse_to_normal --input_dir F:/Datasets/heavy_weather/Cityscapes_Nomal_Foggy/Rain_train/  --gt_dir F:/Datasets/heavy_weather/Cityscapes_Nomal_Foggy/Normal_train/
```

*生成图像后，将原始图像标签复制后，便可和生成图像搭配使用*





## 2. 推理

**Step 2**: use QTNet to  translate image

```python
# generate the normal translation image 
python DBITNet_infer.py --mode normal_to_adverse --input_dir E:/Datasets/heavy_weather/Cityscapes_Nomal_Foggy/Normal_train/  
                        --weight ./QTNet_run/QTNet_weights/normal_to_foggy_fin_OBSNet/_47.pth
# generate the adverse translation image 
# Foggy-Cityscapes
python DBITNet_infer.py --mode adverse_to_normal --input_dir E:/Datasets/heavy_weather/Cityscapes_Nomal_Foggy/Foggy_train/ \
                      --weight ./QTNet_run/QTNet_weights/foggy_to_normal_fin_OBSNet/_49.pth
python DBITNet_infer.py --mode adverse_to_normal --input_dir F:/Datasets/heavy_weather/Cityscapes_Nomal_Foggy/Rain_train/  --weight ./QTNet_run/QTNet_weights/foggy_to_normal_Rain/_30.pth
# RTTS
python DBITNet_infer.py --mode adverse_to_normal --input_dir E:/Datasets/heavy_weather/UnannotatedHazyImages/   \
--weight ./QTNet_run/QTNet_weights/foggy_to_normal_fin_OBSNet/_49.pth
```



## 3.合并

**Step 3**: 将原始图像和合成图像进行合并后训练

> Normal_train：原始正常图像（2965）+ 合成正常图像（2965）
>
> Foggy_train：原始有雾图像（2965）+ 合成有雾图像（2965）
> 
> RTTS: 无标签图像（4809）+ 合成无标签图像（4809）

```python
# move the translation image
mv ./datasets/Normal_to_Foggy/images/Foggy_feak/* ./datasets/Normal_to_Foggy/images/Foggy_train/
mv ./datasets/Normal_to_Foggy/images/Normal_feak/* ./datasets/Normal_to_Foggy/images/Normal_train/
# RTTS
mv E:/Datasets/heavy_weather/UnannotatedHazyImages_feak/*  E:/Datasets/heavy_weather/UnannotatedHazyImages/

```
![img.png](img.png)














