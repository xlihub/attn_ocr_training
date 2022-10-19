#!/bin/bash
pkill -f mask_rcnn_r50_fpn
ssh -p 10022 root@localhost "/usr/bin/pkill -f web_service"
/home/cpard/anaconda3/envs/paddlex/bin/python /home/cpard/attn_ocr_training/tutorials/train/instance_segmentation/mask_rcnn_r50_fpn.py
ssh -p 10022 root@localhost "sleep 5s"
ssh -p 10022 root@localhost "cd /paddle/attn_ocr_serving/deploy/pdserving; /usr/bin/python -u /paddle/attn_ocr_serving/deploy/pdserving/web_service.py &>log.txt&"
