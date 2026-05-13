import os
import shutil

# 指定要遍历的文件夹路径
folder_path = "/home/derrick/catkin_ws/src/code_llm/workspace/comparative/cap/shaping"

# 遍历文件夹中的文件
for file_name in os.listdir(folder_path):
    # 检查文件是否为Python文件
    if file_name.endswith(".py"):
        # 构造文件的完整路径
        file_path = os.path.join(folder_path, file_name)

        # 获取文件名（不包括扩展名）并创建同名文件夹
        folder_name = os.path.splitext(file_name)[0]
        new_folder_path = os.path.join(folder_path, folder_name)
        os.makedirs(new_folder_path, exist_ok=True)

        # 构造新文件的路径并将原文件复制并重命名为 main.py
        new_file_path = os.path.join(new_folder_path, "main.py")
        shutil.move(file_path, new_file_path)

        print(f"Processed {file_name} -> {new_folder_path}/main.py")
