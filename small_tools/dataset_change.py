# 写一段程序，读取一个文件夹，将其中类似这种图片名称的（source_***_fake_B），进行删除
import os

folder_path = "D:/桌面/Foggy_train"

# 获取文件夹中的所有文件
files = os.listdir(folder_path)

# 遍历文件夹中的文件
for file in files:
    if "source_" in file and file.endswith("_fake_B.png"):
        file_path = os.path.join(folder_path, file)
        os.remove(file_path)
        print(f"Deleted: {file}")