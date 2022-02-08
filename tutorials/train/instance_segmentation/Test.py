import distutils
import shutil
import subprocess
import time
import zipfile
from glob import glob

import paddlex as pdx
import os
import tarfile
import sys
import os.path as osp


def get_train_metrics():
    """ 获取任务日志

    Args:
        data为dict, key包括
        'tid'任务id
    Return:
        train_log(dict): 'eta':剩余时间，'train_metrics': 训练指标，'eval_metircs': 评估指标，
        'download_status': 下载模型状态，'eval_done': 是否已保存模型，'train_error': 训练错误原因
    """
    from paddlex_restful.restful.utils import TrainLogReader
    log_file = osp.join('./', 'out.log')
    train_log = TrainLogReader(log_file)
    train_log.update()
    train_log = train_log.__dict__
    return {'status': 1, 'train_log': train_log}


# test_jpg = 'CPA/JPEGImages/IMG_742031_00001.jpg'
# model = pdx.load_model('output/mask_rcnn_r50_fpn/best_model')
#
# # predict接口并未过滤低置信度识别结果，用户根据需求按score值进行过滤
# result = model.predict(test_jpg)
#
# # 可视化结果存储在./visualized_test.jpg, 见下图
# pdx.det.visualize(test_jpg, result, threshold=0.5, save_dir='./')
# log = get_train_metrics()
# print(log)

# # # Packing model
# tar_name = "{}.tar.gz".format('CPA')
# tar = tarfile.open(tar_name, 'w:gz')
# tar.add('maskrcnn_serving_server')
# tar.close()
# #
# # # Generate donefile
# donefile_name = 'donefile'
# os.system('touch {}'.format(donefile_name))
#
# def mycopyfile(srcfile, dstpath):
#     if not os.path.isfile(srcfile):
#         print("%s not exist!" % srcfile)
#     else:
#         fpath, fname = os.path.split(srcfile)
#         if not os.path.exists(dstpath):
#             os.makedirs(dstpath)
#         shutil.copy(srcfile, dstpath + fname)
#         print("copy %s -> %s" % (srcfile, dstpath + fname))
#
#
# # src_dir = '/home/xli/ftp/upload/new/JPEGImages/'
# # dst_dir = './base/JPEGImages/'
# # src_dir2 = '/home/xli/ftp/upload/new/Annotations/'
# # dst_dir2 = './base/Annotations/'
# # src_dir_list = glob(src_dir + '*')
# # src_dir_list2 = glob(src_dir2 + '*')
# # for src_file in src_dir_list:
# #     mycopyfile(src_file, dst_dir)
# # for src_file in src_dir_list2:
# #     mycopyfile(src_file, dst_dir2)
#
# # pdx.tools.dataset_conversion('labelme', 'MSCOCO', './base/JPEGImages', './base/Annotations', './dataset_coco')
# # shutil.rmtree('./CPA')
def unzip_file(zip_src, dst_dir):
    r = zipfile.is_zipfile(zip_src)
    if r:
        fz = zipfile.ZipFile(zip_src, 'r')
        # 判断文件夹是否存在
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        for file in fz.namelist():
            fz.extract(file, dst_dir)
        os.remove(zip_src)
    else:
        print('This is not zip')


# src_dir_list = glob('/home/xli/ftp/upload/new/' + '*')
# if len(src_dir_list) == 0:
#     sys.exit()
# else:
#     for src_file in src_dir_list:
#         fpath, fname = os.path.split(src_file)
#         print(fpath, fname)
#         namelist = fname.split('.')
#         if len(namelist) > 1:
#             name = namelist[0]
#             filetype = namelist[1]
#             if filetype == 'zip':
#                 dst_dir = '/home/xli/ftp/upload/model/%s' % name
#                 unzip_file(src_file, dst_dir)
train = subprocess.Popen(args='./train.sh', shell=True, universal_newlines=True, encoding='utf-8')
print(train.poll())
time.sleep(5)
print(train.poll())
# model_dir_list = glob('/home/xli/ftp/upload/model/' + '*')
# for src_file in model_dir_list:
#     print(src_file)
#     distutils.dir_util.copy_tree(src_file + '/JPEGImages', './base/JPEGImages')
#     distutils.dir_util.copy_tree(src_file + '/Annotations', './base/Annotations')
# src_dir_list = glob('./output/mask_rcnn_r50_fpn/' + '*')
# for src_file in src_dir_list:
#     print(src_file)
#     fpath, fname = os.path.split(src_file)
#     # print(fname)
#     if not fname == 'pretrain':
#         print(fname)
#         shutil.rmtree(src_file)
# distutils.dir_util.copy_tree('/home/xli/ftp/upload/new/JPEGImages', './base/JPEGImages')
#
# # mycopyfile('./inference_model/model.yml', './maskrcnn_serving_server/')
# src_dir = './'
# dst_dir = '/home/xli/ftp/upload/'
# mycopyfile('CPA.tar.gz', dst_dir)
# mycopyfile('donefile', dst_dir)
