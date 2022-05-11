import base64
import os.path as osp
from psutil import Process
import time


def get_train_metrics():
    """ 获取任务日志

    Args:
        data为dict, key包括
        'tid'任务id
    Return:
        train_log(dict): 'eta':剩余时间，'train_metrics': 训练指标，'eval_metircs': 评估指标，
        'download_status': 下载模型状态，'eval_done': 是否已保存模型，'train_error': 训练错误原因
    """
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


class TrainLogReader(object):
    def __init__(self, log_file):
        self.log_file = log_file
        self.eta = None
        self.train_metrics = None
        self.eval_metrics = None
        self.best_eval_metrics = None
        self.download_status = None
        self.eval_done = False
        self.train_error = None
        self.train_stage = None
        self.running_duration = None

    def update(self):
        if not osp.exists(self.log_file):
            return
        if self.train_stage == "Train Error":
            return
        if self.download_status == "Failed":
            return
        if self.train_stage == "Train Complete":
            return
        logs = open(self.log_file, encoding='utf-8').read().strip().split('\n')
        self.eta = None
        self.train_metrics = None
        self.eval_metrics = None
        self.best_eval_metrics = None
        if self.download_status != "Done":
            self.download_status = None

        start_time_timestamp = osp.getctime(self.log_file)
        for line in logs[::1]:
            line = line.replace('\x1b[0m', '')
            try:
                start_time_str = " ".join(line.split()[0:2])
                start_time_array = time.strptime(start_time_str,
                                                 "%Y-%m-%d %H:%M:%S")
                start_time_timestamp = time.mktime(start_time_array)
                break
            except Exception as e:
                pass
        for line in logs[::-1]:
            line = line.replace('\x1b[0m', '')
            if line.count('Train Complete!'):
                self.train_stage = "Train Complete"
            if line.count('Training stop with error!'):
                self.train_error = line
            if self.train_metrics is not None \
                    and self.eval_metrics is not None and self.eval_done and self.eta is not None:
                break
            items = line.strip().split()
            if line.count('Model saved in'):
                self.eval_done = True
            if line.count('download completed'):
                self.download_status = 'Done'
                break
            if line.count('download failed'):
                self.download_status = 'Failed'
                break
            if self.download_status != 'Done':
                if line.count('[DEBUG]\tDownloading'
                              ) and self.download_status is None:
                    self.download_status = dict()
                    if not line.endswith('KB/s'):
                        continue
                    speed = items[-1].strip('KB/s').split('=')[-1]
                    download = items[-2].strip('M, ').split('=')[-1]
                    total = items[-3].strip('M, ').split('=')[-1]
                    self.download_status['speed'] = speed
                    self.download_status['download'] = float(download)
                    self.download_status['total'] = float(total)
            if self.eta is None:
                if line.count('eta') > 0 and (line[-3] == ':' or
                                              line[-4] == ':'):
                    eta = items[-1].strip().split('=')[1]
                    h, m, s = [int(x) for x in eta.split(':')]
                    self.eta = h * 3600 + m * 60 + s
            if self.train_metrics is None:
                if line.count('[INFO]\t[TRAIN]') > 0 and line.count(
                        'Step') > 0:
                    if not items[-1].startswith('eta'):
                        continue
                    self.train_metrics = dict()
                    metrics = items[4:]
                    for metric in metrics:
                        try:
                            name, value = metric.strip(', ').split('=')
                            value = value.split('/')[0]
                            if value.count('.') > 0:
                                value = float(value)
                            elif value == 'nan':
                                value = 'nan'
                            else:
                                value = int(value)
                            self.train_metrics[name] = value
                        except:
                            pass
            if self.eval_metrics is None:
                if line.count('[INFO]\t[EVAL]') > 0 and line.count(
                        'Finished') > 0:
                    if not line.strip().endswith(' .'):
                        continue
                    self.eval_metrics = dict()
                    metrics = items[5:]
                    for metric in metrics:
                        try:
                            name, value = metric.strip(', ').split('=')
                            value = value.split('/')[0]
                            if value.count('.') > 0:
                                value = float(value)
                            else:
                                value = int(value)
                            self.eval_metrics[name] = value
                        except:
                            pass

                if self.best_eval_metrics is None:
                    if line.count('[INFO]') > 0 and line.count(
                            'Current evaluated best model') > 0:
                        # if line.strip().find('Current evaluated best model') == -1:
                        #     continue
                        try:
                            for item in items:
                                epoch = item.find('epoch')
                                if not epoch == -1:
                                    best_epoch = item[:-1]
                                bbox = item.find('bbox_mmap')
                                if not bbox == -1:
                                    best_bbox = round(float(item.split('=')[1]), 6)
                            self.best_eval_metrics = dict()
                            self.best_eval_metrics['best_epoch'] = best_epoch
                            self.best_eval_metrics['best_bbox'] = best_bbox
                        except:
                            pass

        end_time_timestamp = osp.getmtime(self.log_file)
        t_diff = time.gmtime(end_time_timestamp - start_time_timestamp)
        self.running_duration = "{}小时{}分{}秒".format(
            t_diff.tm_hour, t_diff.tm_min, t_diff.tm_sec)
