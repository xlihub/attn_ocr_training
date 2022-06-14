import distutils.dir_util
import json
import os
import random
import shlex
import shutil
import subprocess
import sys
import time
from glob import glob

import requests
from flask import Flask, request, render_template, session, flash, redirect, \
    url_for, jsonify, Response
from celery import Celery
import os.path as osp
from utils import image_to_base64, terminate_process
from celery.utils.log import get_task_logger

app = Flask(__name__)
app.config['SECRET_KEY'] = 'top-secret!'

# Celery configuration
# app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
# app.config['result_backend'] = 'redis://localhost:6379/0'
# app.config['broker_transport_options'] = {'visibility_timeout': 18000}  # 5 hours
# app.config['result_backend_transport_options'] = {'visibility_timeout': 18000}  # 5 hours
app.config['CELERY_BROKER_URL'] = 'amqp://attnocr:attn3100@localhost:5672/attnocr'
app.config['result_backend'] = 'redis://localhost:6379/0'

# Initialize Celery
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
# celery.conf.result_backend = 'redis://localhost:6379/0'
# celery.conf.broker_transport_options = {'visibility_timeout': 18000}  # 5 hours
# celery.conf.broker_transport_options = {'max_retries': 5}
# celery.conf.worker_max_tasks_per_child = 40
celery.conf.update(app.config)

logger = get_task_logger(__name__)


@celery.task(bind=True)
def long_task(self):
    """Background task that runs a long function with progress reports."""
    # 将训练日志写入out.log与err.log文件
    mode = 'w'
    outlog = open(osp.join('./', 'out.log'), mode=mode, encoding='utf-8')
    errlog = open(osp.join('./', 'err.log'), mode=mode, encoding='utf-8')
    outlog.write("This log file path is {}\n".format(
        osp.join('./', 'out.log')))
    outlog.write("注意：标志为WARNING/INFO类的仅为警告或提示类信息，非错误信息\n")
    errlog.write("This log file path is {}\n".format(
        osp.join('./', 'err.log')))
    errlog.write("注意：标志为WARNING/INFO类的仅为警告或提示类信息，非错误信息\n")

    train = subprocess.Popen(args='./train.sh', shell=True, stdout=outlog, stderr=errlog, universal_newlines=True, encoding='utf-8')
    # try:
    #     outs, errs = train.communicate(timeout=1200)
    #     print(outs)
    # except subprocess.TimeoutExpired:
    #     train.kill()
    logger.info('celery in')
    from utils import get_train_metrics
    while train.poll() is None:
        # print(train.poll())
        log = get_train_metrics()
        if log['train_log']['eta'] is None:
            message = "正在准备训练环境..."
            self.update_state(state='PROGRESS',
                              meta={'current': 0, 'total': 12,
                                    'status': message})
        else:
            if log['train_log']['train_stage'] == 'Train Complete':
                os.remove('./out.log')
                os.remove('./err.log')
                src_dir_list = glob('./output/mask_rcnn_r50_fpn/' + '*')
                for src_file in src_dir_list:
                    fpath, fname = os.path.split(src_file)
                    if not fname == 'pretrain':
                        shutil.rmtree(src_file)
                self.update_state(state='PROGRESS',
                                  meta={'current': 12, 'total': 12,
                                        'status': '训练即将完成，请稍后...'})
                time.sleep(12)
                break
            else:
                epoch, message = create_train_message(log)
                self.update_state(state='PROGRESS',
                                  meta={'current': epoch, 'total': 12,
                                        'status': message})
                time.sleep(2)
    best_bbox = log['train_log']['best_eval_metrics']['best_bbox']
    return {'current': 100, 'total': 100, 'status': '训练完成！\n' + '本次训练评分:' + str(best_bbox)}


def create_train_message(train_log):
    eval_epoch = epoch_bbox = best_bbox = 0
    left_time = str(train_log['train_log']['eta'])
    train_epoch = train_log['train_log']['train_metrics']['Epoch']
    epoch = int(train_log['train_log']['train_metrics']['Epoch']) - 1
    if not train_log['train_log']['eval_metrics'] is None:
        eval_epoch = train_log['train_log']['eval_metrics']['Epoch']
        epoch_bbox = train_log['train_log']['eval_metrics']['bbox_mmap']
    if not train_log['train_log']['best_eval_metrics'] is None:
        best_bbox = train_log['train_log']['best_eval_metrics']['best_bbox']
    if eval_epoch == 0:
        message = '正在进行第 ' + str(train_epoch) + ' 轮训练，共 12 轮\n'
    else:
        message = '正在进行第 ' + str(train_epoch) + ' 轮训练，共 12 轮\n' + '已完成 ' + str(eval_epoch) + ' 轮训练，' + '第 ' + str(
            eval_epoch) + ' 轮训练评分: ' + str(epoch_bbox) + '，本次训练最高评分: ' + str(best_bbox) + '\n'
    message = message + "已运行: " + train_log['train_log']['running_duration'] + "  (估计剩余时间: " + left_time + "秒)"
    return epoch, message


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html', email=session.get('email', ''))


# 上传文件
@app.route('/upload_file', methods=['POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        # basepath = os.path.dirname(__file__)  # 当前文件所在路径
        upload_path = '/home/attnroot/ftp/upload/new_model/'
        print(f.filename)
        #######################################
        # 毫秒级时间戳
        # file_name = str(round(time.time() * 1000))
        # dir = str(time.strftime('%y%m%d', time.localtime()))
        # upload_path = os.path.join(basepath, 'uploads/'+dir + '/')
        # 判断文件夹是否存在
        if not os.path.exists(upload_path):
            os.makedirs(upload_path)
        #######################################
        # file_path = str(file_name)+str(f.filename)
        file_path = str(f.filename)
        f.save(upload_path+"/"+file_path)
    return Response(json.dumps(file_path), mimetype='application/json')


@app.route('/longtask', methods=['POST'])
def longtask():
    dstpath = './out.log'
    if not os.path.exists(dstpath):
        task = long_task.apply_async()
        return jsonify({}), 202, {'Location': url_for('taskstatus',
                                                      task_id=task.id)}
    else:
        from utils import get_train_metrics
        log = get_train_metrics()
        message = ''
        if log['train_log']['eta'] is not None:
            left_time = str(log['train_log']['eta'])
            message = "估计剩余时间: " + left_time + "秒"
        res = {'current': 100, 'total': 100, 'status': '训练进行中...',
               'result': '训练进行中,请稍后...\n' + message}
        return jsonify(res), 200



@app.route('/status/<task_id>')
def taskstatus(task_id):
    task = long_task.AsyncResult(task_id)
    from utils import get_train_metrics
    log = get_train_metrics()
    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'current': 0,
            'total': 1,
            'status': 'Pending...'
        }
    elif task.state != 'FAILURE':
        if task.state == 'REVOKED':
            print(task)
            response = {
                'state': task.state,
                'current': 1,
                'total': 1,
                'status': str(task.info),  # this is the exception raised
                'result': '训练已停止'
            }
        else:
            response = {
                'state': task.state,
                'current': task.info.get('current', 0),
                'total': task.info.get('total', 1),
                'status': task.info.get('status', '')
            }
            if 'result' in task.info:
                response['result'] = task.info['result']
    else:
        # something went wrong in the background job
        print(task)
        response = {
            'state': task.state,
            'current': 1,
            'total': 1,
            'status': str(task.info),  # this is the exception raised
        }
    return jsonify(response)


@app.route('/stop_longtask', methods=['POST'])
def stop_longtask():
    inspect = celery.control.inspect()
    active = inspect.active()
    print(active)
    print(longtask)
    tasks = active['attnocr@attnroot-System-Product-Name']
    if len(tasks) > 0:
        task_id = tasks[0]['id']
        task = long_task.AsyncResult(task_id)
        task_pid = tasks[0]['worker_pid']
        terminate_process(task_pid)
        task.revoke(terminate=True)
        os.remove('./out.log')
        os.remove('./err.log')
        src_dir_list = glob('./output/mask_rcnn_r50_fpn/' + '*')
        for src_file in src_dir_list:
            fpath, fname = os.path.split(src_file)
            if not fname == 'pretrain':
                shutil.rmtree(src_file)
    res = {'current': 100, 'total': 100, 'status': '训练已停止',
           'result': '训练已停止'}
    return jsonify(res), 200


# 上传文件
@app.route('/ocr_model_pic', methods=['POST'])
def ocr_model():
    if request.method == 'POST':
        f = request.files['file']
        upload_path = '/home/attnroot/ftp/upload/tmp/'
        print(f.filename)
        # 判断文件夹是否存在
        if not os.path.exists(upload_path):
            os.makedirs(upload_path)
        file_path = str(f.filename)
        save_path = upload_path + "/" + file_path
        f.save(save_path)

        url = 'http://127.0.0.1:8008/predict'
        if f.mimetype == 'image/jpeg':
            preb = 'data:image/jpeg;base64,'
        elif f.mimetype == 'image/png':
            preb = 'data:image/png;base64,'
        data = image_to_base64(f.filename, upload_path, prefix=preb)
        postdata = {"ImageBase64": data}
        # proxies = {
        #     "http": None,
        #     "https": None,
        # }
        r = requests.Session()
        r.proxies = {
            'http': None,
            'https': None
        }
        r = requests.post(url, json=postdata)
        r.encoding = 'utf-8'
        if r.status_code == 200:
            print(r.text)
            results = json.loads(r.text)
            if len(results) == 1:
                info = results[0]['InvoiceInfos']
                if len(info) == 0:
                    type = results[0]['InvoiceType']
                    if type == 'unknown':
                        res = {
                            'type': 'unknown',
                            'info': 'None'
                        }
                    else:
                        res = {
                            'type': type,
                            'info': 'None'
                        }
                else:
                    res = {
                        'type': results[0]['InvoiceType'],
                        'info': json.dumps(results[0]['InvoiceInfos'], ensure_ascii=False),
                        'extra': json.dumps(results[0]['InvoiceExtra'], ensure_ascii=False)
                    }
            else:
                res = []
                for result in results:
                    info = result['InvoiceInfos']
                    if len(info) == 0:
                        res_dic = {
                            'type': 'unknown',
                            'info': 'None'
                        }
                    else:
                        res_dic = {
                            'type': result['InvoiceType'],
                            'info': json.dumps(result['InvoiceInfos'], ensure_ascii=False),
                            'extra': json.dumps(results['InvoiceExtra'], ensure_ascii=False)
                        }
                    res.append(res_dic)
        else:
            print(r.text)
            res = r.text
        os.remove(save_path)
    return Response(json.dumps(res, ensure_ascii=False), mimetype='application/json')


# 上传文件
@app.route('/upload_temp_file', methods=['POST'])
def upload_temp():
    if request.method == 'POST':
        f = request.files['file']
        # basepath = os.path.dirname(__file__)  # 当前文件所在路径
        upload_path = '/home/attnroot/ftp/upload/new_template/'
        print(f.filename)
        #######################################
        # 毫秒级时间戳
        # file_name = str(round(time.time() * 1000))
        # dir = str(time.strftime('%y%m%d', time.localtime()))
        # upload_path = os.path.join(basepath, 'uploads/'+dir + '/')
        # 判断文件夹是否存在
        if not os.path.exists(upload_path):
            os.makedirs(upload_path)
        #######################################
        # file_path = str(file_name)+str(f.filename)
        file_path = str(f.filename)
        save_path = upload_path+"/"+file_path
        f.save(save_path)

        url = 'http://127.0.0.1:8008/update_temp_file'
        r = requests.Session()
        r.proxies = {
            'http': None,
            'https': None
        }
        r = requests.get(url)
        r.encoding = 'utf-8'
        if r.status_code == 200:
            print(r.text)
        else:
            print(r.text)
        distutils.dir_util.copy_tree('/home/attnroot/ftp/upload/new_template/', '/home/attnroot/ftp/upload/template/')
        os.remove(save_path)
    return Response(json.dumps(r.text), mimetype='application/json')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5050, debug=True)
