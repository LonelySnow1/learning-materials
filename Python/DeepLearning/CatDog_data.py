import os, shutil


# 数据预处理
def mymovefile(srcfile, dstfile):
    if not os.path.isfile(srcfile):
        print("src not exist!")
    else:
        fpath, fname = os.path.split(dstfile)  # 分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)  # 创建路径
        shutil.move(srcfile, dstfile)  # 移动文件


test_rate = 0.2  # 训练集和测试集的比例为8:2。
img_num = 3000
test_num = int(img_num * test_rate)

import random

test_index = random.sample(range(0, img_num), test_num)
file_path = r"D:\下载\data"
tr = "train"
te = "test"
cat = "cats"
dog = "dogs"

# 将上述index中的文件都移动到/test/cats/和/test/dogs/下面去。
for i in range(len(test_index)):
    # 移动猫
    srcfile = os.path.join(file_path, tr, cat, str(test_index[i]) + ".jpg")
    dstfile = os.path.join(file_path, te, cat, str(test_index[i]) + ".jpg")
    mymovefile(srcfile, dstfile)
    # 移动狗
    srcfile = os.path.join(file_path, tr, dog, str(test_index[i]) + ".jpg")
    dstfile = os.path.join(file_path, te, dog, str(test_index[i]) + ".jpg")
    mymovefile(srcfile, dstfile)
