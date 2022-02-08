#!/bin/bash
pkill -f mask_rcnn_r50_fpn
ssh -p 10022 root@localhost "/usr/bin/pkill -f web_service"
/home/xli/anaconda3/envs/paddlex/bin/python /home/xli/Train/PaddleX/tutorials/train/instance_segmentation/mask_rcnn_r50_fpn.py
ssh -p 10022 root@localhost "sleep 12s"
ssh -p 10022 root@localhost "cd /paddle/PaddleOCR/deploy/pdserving; /usr/bin/python -u ./web_service.py"
