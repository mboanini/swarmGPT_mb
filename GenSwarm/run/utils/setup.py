"""
Copyright (c) 2024 WindyLab of Westlake University, China
All rights reserved.

This software is provided "as is" without warranty of any kind, either
express or implied, including but not limited to the warranties of
merchantability, fitness for a particular purpose, or non-infringement.
In no event shall the authors or copyright holders be liable for any
claim, damages, or other liability, whether in an action of contract,
tort, or otherwise, arising from, out of, or in connection with the
software or the use or other dealings in the software.
"""

import os
import shutil


def setup_metagpt(directory):
    # 确保目录存在
    if not os.path.exists(directory):
        print(f"目录 {directory} 不存在")
        return
    # 如果目录下没有py文件，遍历所以的文件夹找到有py文件的，将其内部的所以py文件复制到directory下
    has_py_files = any(file.endswith(".py") for file in os.listdir(directory))
    if not has_py_files:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".py"):
                    src_file = os.path.join(root, file)
                    dest_file = os.path.join(directory, file)
                    shutil.copy(src_file, dest_file)
                    print(f"复制文件 {src_file} 到 {dest_file}")
    # 将apis.py文件复制到directory下
    source_file = os.path.join(
        "modules/deployment/execution_scripts", "apis_all.py"
    )
    if os.path.exists(source_file):
        shutil.copy(source_file, directory)
        # 将apis_all.py 重命名为api.py
        dest_file = os.path.join(directory, "api.py")
        os.rename(os.path.join(directory, "apis_all.py"), dest_file)
    else:
        print(f"文件 {source_file} 不存在")
        return
    source_file = os.path.join("modules/deployment/execution_scripts", "run_meta.py")
    if os.path.exists(source_file):
        shutil.copy(source_file, directory)
        dest_file = os.path.join(directory, "run.py")

        os.rename(os.path.join(directory, "run_meta.py"), dest_file)

    else:
        print(f"文件 {source_file} 不存在")
        return

    # 遍历文件夹下所有python文件，开头加上from apis import *
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py") and file != "api.py":
                file_path = os.path.join(root, file)
                with open(file_path, "r+", encoding="utf-8") as f:
                    content = f.read()
                    if not content.startswith("from api import *"):
                        f.seek(0, 0)
                        f.write("from api import *\n" + content)


def setup_cap(directory):
    # 确保目录存在
    if not os.path.exists(directory):
        print(f"目录 {directory} 不存在")
        return
    # 如果目录下没有py文件，遍历所以的文件夹找到有py文件的，将其内部的所以py文件复制到directory下
    has_py_files = any(file.endswith(".py") for file in os.listdir(directory))
    if not has_py_files:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".py"):
                    src_file = os.path.join(root, file)
                    dest_file = os.path.join(directory, file)
                    shutil.copy(src_file, dest_file)
                    print(f"复制文件 {src_file} 到 {dest_file}")
    # 将apis.py文件复制到directory下
    source_file = os.path.join(
        "modules/deployment/execution_scripts", "apis_all.py"
    )
    if os.path.exists(source_file):
        shutil.copy(source_file, directory)
        # 将apis_all.py 重命名为api.py
        dest_file = os.path.join(directory, "api.py")
        os.rename(os.path.join(directory, "apis_all.py"), dest_file)
    else:
        print(f"文件 {source_file} 不存在")
        return
    source_file = os.path.join("modules/deployment/execution_scripts", "run_meta.py")
    if os.path.exists(source_file):
        shutil.copy(source_file, directory)
        dest_file = os.path.join(directory, "run.py")

        os.rename(os.path.join(directory, "run_meta.py"), dest_file)

    else:
        print(f"文件 {source_file} 不存在")
        return

    # 遍历文件夹下所有python文件，开头加上from apis import *
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py") and file != "api.py":
                file_path = os.path.join(root, file)
                with open(file_path, "r+", encoding="utf-8") as f:
                    content = f.read()
                    if not content.startswith("from api import *"):
                        f.seek(0, 0)
                        f.write("from api import *\n" + content)


def setup_llm2swarm(directory):
    # 确保目录存在
    if not os.path.exists(directory):
        print(f"目录 {directory} 不存在")
        return
    # 如果目录下没有py文件，遍历所以的文件夹找到有py文件的，将其内部的所以py文件复制到directory下
    has_py_files = any(file.endswith(".py") for file in os.listdir(directory))
    if not has_py_files:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".py"):
                    src_file = os.path.join(root, file)
                    dest_file = os.path.join(directory, file)
                    shutil.copy(src_file, dest_file)
                    print(f"复制文件 {src_file} 到 {dest_file}")
    # 将apis.py文件复制到directory下
    source_file = os.path.join(
        "modules/deployment/execution_scripts", "apis_all.py"
    )
    if os.path.exists(source_file):
        shutil.copy(source_file, directory)
        # 将apis_all.py 重命名为api.py
        dest_file = os.path.join(directory, "api.py")
        os.rename(os.path.join(directory, "apis_all.py"), dest_file)
    else:
        print(f"文件 {source_file} 不存在")
        return
    source_file = os.path.join("modules/deployment/execution_scripts", "run_llm2swarm.py")
    if os.path.exists(source_file):
        shutil.copy(source_file, directory)
        dest_file = os.path.join(directory, "run.py")

        os.rename(os.path.join(directory, "run_llm2swarm.py"), dest_file)

    else:
        print(f"文件 {source_file} 不存在")
        return

    source_file = os.path.join("modules/deployment/execution_scripts", "robot.py")
    if os.path.exists(source_file):
        shutil.copy(source_file, directory)

    source_file = os.path.join("modules/deployment/execution_scripts", "aux.py")
    if os.path.exists(source_file):
        shutil.copy(source_file, directory)
