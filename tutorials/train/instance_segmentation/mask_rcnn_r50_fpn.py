# 环境变量配置，用于控制是否使用GPU
# 说明文档：https://paddlex.readthedocs.io/zh_CN/develop/appendix/parameters.html#gpu
import distutils.dir_util
import os
import shutil
import sys
import time
import tarfile
import zipfile
from glob import glob

import paddle_serving_client.io as serving_io

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from paddlex.det import transforms
import paddlex as pdx
import sys
import os.path as osp


def mycopyfile(srcfile, dstpath):
    if not os.path.isfile(srcfile):
        print("%s not exist!" % srcfile)
    else:
        fpath, fname = os.path.split(srcfile)
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)
        shutil.copy(srcfile, dstpath + fname)
        print("copy %s -> %s" % (srcfile, dstpath + fname))


def unzip_file(zip_src, dst_dir):
    r = zipfile.is_zipfile(zip_src)
    if r:
        fz = zipfile.ZipFile(zip_src, 'r')
        # 判断文件夹是否存在
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        for file in fz.namelist():
            fz.extract(file, dst_dir)
        # os.remove(zip_src)
    else:
        print('This is not zip')


# 检测FTP服务器中是否有新的模型数据集
src_dir_list = glob('/home/xli/ftp/upload/new_model/' + '*')
if len(src_dir_list) == 0:
    sys.exit()
else:
    # 将新的模型数据集解压到model文件夹
    for src_file in src_dir_list:
        fpath, fname = os.path.split(src_file)
        print(fpath, fname)
        namelist = fname.split('.')
        if len(namelist) > 1:
            name = namelist[0]
            filetype = namelist[1]
            if filetype == 'zip':
                dst_dir = '/home/xli/ftp/upload/model/'
                unzip_file(src_file, dst_dir)

# 将model文件夹中的所有model数据集汇总到base文件夹中
model_dir_list = glob('/home/xli/ftp/upload/model/' + '*')
for model_file in model_dir_list:
    distutils.dir_util.copy_tree(model_file + '/JPEGImages', './base/JPEGImages')
    distutils.dir_util.copy_tree(model_file + '/Annotations', './base/Annotations')

# 将base文件夹中的数据集转换成COCO格式，并保存到dataset_coco文件夹中
shutil.rmtree('./dataset_coco')
pdx.tools.dataset_conversion('labelme', 'MSCOCO', './base/JPEGImages', './base/Annotations', './dataset_coco')

# 将dataset_coco文件夹中的数据集拷贝到CPA文件夹，准备开始训练
shutil.rmtree('./CPA')
shutil.copytree('./dataset_coco', './CPA')

# 切分CPA文件夹中的数据集
pdx.tools.split.split_coco_dataset(
    'CPA', 0.2, 0.1, 'CPA')

# 定义训练和验证时的transforms
# API说明 https://paddlex.readthedocs.io/zh_CN/develop/apis/transforms/det_transforms.html
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.ResizeByShort(
        short_size=800, max_size=1333), transforms.Padding(coarsest_stride=32)
])

eval_transforms = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), transforms.ResizeByShort(
        short_size=800, max_size=1333), transforms.Padding(coarsest_stride=32)
])

# 定义训练和验证所用的数据集
# API说明：https://paddlex.readthedocs.io/zh_CN/develop/apis/datasets.html#paddlex-datasets-cocodetection
train_dataset = pdx.datasets.CocoDetection(
    data_dir='CPA/JPEGImages',
    ann_file='CPA/train.json',
    transforms=train_transforms,
    shuffle=True)
eval_dataset = pdx.datasets.CocoDetection(
    data_dir='CPA/JPEGImages',
    ann_file='CPA/val.json',
    transforms=eval_transforms)

# 初始化模型，并进行训练
# 可使用VisualDL查看训练指标，参考https://paddlex.readthedocs.io/zh_CN/develop/train/visualdl.html
# num_classes 需要设置为包含背景类的类别数，即: 目标类别数量 + 1
num_classes = len(train_dataset.labels) + 1

# API说明：https://paddlex.readthedocs.io/zh_CN/develop/apis/models/instance_segmentation.html#maskrcnn
model = pdx.det.MaskRCNN(num_classes=num_classes, backbone='ResNet50')

# API说明：https://paddlex.readthedocs.io/zh_CN/develop/apis/models/instance_segmentation.html#train
# 各参数介绍与调整说明：https://paddlex.readthedocs.io/zh_CN/develop/appendix/parameters.html
model.train(
    num_epochs=12,
    train_dataset=train_dataset,
    train_batch_size=1,
    pretrain_weights='COCO',
    eval_dataset=eval_dataset,
    learning_rate=0.0025,
    warmup_steps=65,
    lr_decay_epochs=[8, 11],
    save_dir='output/mask_rcnn_r50_fpn',
    use_vdl=True)

# 将训练好的模型导出为inference格式
model = pdx.load_model('output/mask_rcnn_r50_fpn/best_model/')
model.export_inference_model('./inference_model')

# 将inference格式的模型转换成pdserving部署所需的格式
serving_io.inference_model_to_serving('./inference_model',
                                      serving_server='maskrcnn_serving_server',
                                      serving_client='maskrcnn_serving_client',
                                      model_filename='__model__',
                                      params_filename='__params__')

# 将模型的yml配置文件拷贝到server文件夹
mycopyfile('./inference_model/model.yml', './maskrcnn_serving_server/')

# 打包模型文件夹
tar_name = "{}.tar.gz".format('CPA')
tar = tarfile.open(tar_name, 'w:gz')
tar.add('maskrcnn_serving_server')
tar.close()

# 生成donefile文件
donefile_name = 'donefile'
os.system('touch {}'.format(donefile_name))

# 将模型打包文件与donefile文件拷贝到FTP服务器
src_dir = './'
dst_dir = '/home/xli/ftp/upload/'
mycopyfile('CPA.tar.gz', dst_dir)
mycopyfile('donefile', dst_dir)


# 删除new_model中的zip文件
for src_file in src_dir_list:
    os.remove(src_file)
print('Train Complete!')
