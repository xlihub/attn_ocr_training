# 环境变量配置，用于控制是否使用GPU
# 说明文档：https://paddlex.readthedocs.io/zh_CN/develop/appendix/parameters.html#gpu
import distutils.dir_util
import os
import shutil
import datetime
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


def create_folder(path):
    # 获得当前系统时间的字符串
    localtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    print('localtime=' + localtime)
    # 系统当前时间年份
    year = time.strftime('%Y', time.localtime(time.time()))
    # 月份
    month = time.strftime('%Y-%m', time.localtime(time.time()))
    # 日期
    day = time.strftime('%Y-%m-%d', time.localtime(time.time()))
    # 具体时间 小时分钟毫秒
    mdhms = time.strftime('%H:%M:%S', time.localtime(time.time()))

    fileYear = path + '/' + year
    fileMonth = fileYear + '/' + month
    fileDay = fileMonth + '/' + day
    filepath = fileDay + '/' + mdhms

    if not os.path.exists(fileYear):
        os.makedirs(fileYear)
        os.makedirs(fileMonth)
        os.makedirs(fileDay)
    else:
        if not os.path.exists(fileMonth):
            os.makedirs(fileMonth)
            os.makedirs(fileDay)
        else:
            if not os.path.exists(fileDay):
                os.makedirs(fileDay)

    os.makedirs(filepath)
    return filepath


def create_models_info(path):
    target_path = path

    all_content = os.listdir(target_path)
    print('All content numbers is', len(all_content))
    filepath = './models_info.txt'
    out = open(filepath, 'w')
    from utils import get_train_metrics
    log = get_train_metrics()
    best_epoch = log['train_log']['best_eval_metrics']['best_epoch']
    best_bbox = log['train_log']['best_eval_metrics']['best_bbox']
    out.write('best_epoch' + ':' + str(best_epoch) + '\n')
    out.write('best_bbox' + ':' + str(best_bbox) + '\n')
    total = 0
    for content in all_content:
        img_path = target_path + '/' + content + '/JPEGImages'
        if os.path.isdir(img_path):
            all_sub_content = os.listdir(img_path)
            out.write(content + ':' + str(len(all_sub_content)) + '\n')
            total += len(all_sub_content)
    out.write('Total:' + str(total) + '\n')
    out.close()


# 检测FTP服务器中是否有新的模型数据集
src_dir_list = glob('/home/cpard/ftp/upload/new_model/' + '*')
if len(src_dir_list) == 0:
    dstpath = './out.log'
    if os.path.exists(dstpath):
        os.remove('./out.log')
        os.remove('./err.log')
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
                dst_dir = '/home/cpard/ftp/upload/model/'
                unzip_file(src_file, dst_dir)

# 将model文件夹中的所有model数据集汇总到base文件夹中
model_dir_list = glob('/home/cpard/ftp/upload/model/' + '*')
for model_file in model_dir_list:
    distutils.dir_util.copy_tree(model_file + '/JPEGImages', '/home/cpard/TrainData/base/JPEGImages')
    distutils.dir_util.copy_tree(model_file + '/Annotations', '/home/cpard/TrainData/base/Annotations')

# 将base文件夹中的数据集转换成COCO格式，并保存到dataset_coco文件夹中
if os.path.exists('/home/cpard/TrainData/dataset_coco'):
    shutil.rmtree('/home/cpard/TrainData/dataset_coco')
pdx.tools.dataset_conversion('labelme', 'MSCOCO', '/home/cpard/TrainData/base/JPEGImages', '/home/cpard/TrainData/base/Annotations', '/home/cpard/TrainData/dataset_coco')

# 将dataset_coco文件夹中的数据集拷贝到CPA文件夹，准备开始训练
# shutil.rmtree('/home/cpard/TrainData/CPA')
# shutil.copytree('/home/cpard/TrainData/dataset_coco', '/home/cpard/TrainData/CPA')
distutils.dir_util.copy_tree('/home/cpard/TrainData/dataset_coco', '/home/cpard/TrainData/CPA')

# 切分CPA文件夹中的数据集
pdx.tools.split.split_coco_dataset(
    '/home/cpard/TrainData/CPA', 0.2, 0.1, '/home/cpard/TrainData/CPA')

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
    data_dir='/home/cpard/TrainData/CPA/JPEGImages',
    ann_file='/home/cpard/TrainData/CPA/train.json',
    transforms=train_transforms,
    shuffle=True)
eval_dataset = pdx.datasets.CocoDetection(
    data_dir='/home/cpard/TrainData/CPA/JPEGImages',
    ann_file='/home/cpard/TrainData/CPA/val.json',
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

# 将训练日志文件拷贝到server文件夹
mycopyfile('./out.log', './maskrcnn_serving_server/')

# 将models_info文件拷贝到server文件夹
path = '/home/cpard/ftp/upload/model'  # 文件夹建立位置
create_models_info(path)
mycopyfile('./models_info.txt', './maskrcnn_serving_server/')

# 将server文件夹拷贝到serving目录
distutils.dir_util.copy_tree('./maskrcnn_serving_server', '/home/cpard/ppocr/maskrcnn_serving_server')

# 将server文件夹拷贝到TrainModel目录
path = create_folder(r'/home/cpard/TrainModel')  # 文件夹建立位置
distutils.dir_util.copy_tree('./maskrcnn_serving_server', path + '/maskrcnn_serving_server')
"""
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
dst_dir = '/home/cpard/ftp/upload/'
mycopyfile('CPA.tar.gz', dst_dir)
mycopyfile('donefile', dst_dir)
"""

# 删除new_model中的zip文件
for src_file in src_dir_list:
    os.remove(src_file)
print('Train Complete!')
