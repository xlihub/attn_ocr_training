import base64
import os.path as osp
from psutil import Process


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


# image转换成base64并加上 前缀data:image/jpeg;base64,
def image_to_base64(filename, path, **kwargs):
    """
    :param filename: image文件名
    :param path: image存放路径
    :param kwargs: 参数prefix(转换base64后需要加上的前缀)
    :return:
    """
    path = osp.join(path, filename)
    # 转为二进制格式
    with open(path, "rb") as f:
        data = str(base64.b64encode(f.read()), "utf-8")
        # 转换base64后加上前缀
        if "prefix" in kwargs.keys():
            data = kwargs["prefix"] + data
            # base64_data = bytes(('data: image/jpeg;base64,%s' % str(base64.b64encode(f.read()), "utf-8")), "utf-8")
        # 转换为bytes对象
        # base64_data = bytes(data, "utf-8")
        print("Succeed: %s >> 图片转换成base64" % path)
        return data


def terminate_process(pid):
    process = Process(pid)
    for child in process.children(recursive=True):
        child.kill()
    # process.kill()
