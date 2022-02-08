import os
import cv2
import time
import random
import shutil
import logging
import output
# import multiprocessing
# import mmcv
from flask import Flask,jsonify,request
from mmdet.apis import init_detector, inference_detector
#import para
from para1 import *

logging.basicConfig(level=logging.NOTSET)




def work_proc(gpu_id, process_id):
    logging.debug('GPU_ID = % d, process_ID = % d' %(gpu_id, process_id))
    device_str = '{:d},'.format(gpu_id)
    os.environ['CUDA_VISIBLE_DEVICES'] = device_str

 
    # 模型初始化
    # initial_model_dict = multi_model_initialization(initial_model_name_list, initial_model_path_dict)
    logging.debug('****************object detection initialization is starting**********************')
    # 检测配置文件与模型是否存在
    list_cfg = ['xn', 'ybkk', 'waiguan', 'tyfh']
    for cfg in list_cfg:
        # if not os.path.exists('cfg_{}'.format(cfg)):
        #     logging.error("{} cfg is nonexist".format(cfg))
        # if not os.path.exists('weight_{}'.format(cfg)):
        #     logging.error("{} weight is nonexist".format(cfg))
        model_xn = init_detector(cfg_xn, weight_xn)
        model_ybkk = init_detector(cfg_ybkk, weight_ybkk)
        model_tyfh = init_detector(cfg_tyfh, weight_tyfh)
        model_waiguan = init_detector(cfg_waiguan, weight_waiguan)
        # logging.debug('{} model initialization is successful!'.format(cfg))
    logging.debug('************************all models initialization is successful!************************')
    # 开始检测
    logging.debug('************************model detection begins!************************')
    # 检测存放图片文件夹是否存在
    if not os.path.exists(data_to_detect):
        logging.error('path to save images is nonexists!')
    imgs = os.listdir(data_to_detect)
    # 检测图片是否存在
    for img in imgs:
        if not os.path.exists(os.path.join(data_to_detect, img)):
            logging.error('{} is nonexists!'.format(os.path.join(data_to_detect, img)))
    # 检测图片是否损坏
        jpg_to_detect = cv2.imread(os.path.join(data_to_detect, img))
        if jpg_to_detect is None:
            logging.error('{} is nonexists!')
        if len(jpg_to_detect.shape) > 3:
            img = cv2.cvtColor(jpg_to_detect, cv2.COLOR_BGRA2BGR)
    # 推理结果
        result_list = inference_detector(model_waiguan, jpg_to_detect)
        logging.debug(result_list)
        print(1111111111111)
        for i in range(0, len(model_waiguan.CLASSES)):
            class_name = model_waiguan.CLASSES[i]
            logging.debug('class_name = % s' %(class_name))
            for j in range(0, len(result_list[i])):
                res = result_list[i][j]
                logging.debug(res)
                if 1: 
                    score = float(res[4])
                    if label_conf_thresh_dict['bj_bpps'] > score:
                        continue
                    xmin = int(res[0])
                    ymin = int(res[1])
                    xmax = int(res[2])
                    ymax = int(res[3])
                    # if class_name in outputlist:
                    #     save_flag = True
                    logging.debug('img_name = {:}, score = {:}, bndbox = {:}'.format(img, score, (xmin,ymin,xmax,ymax)))
                    cv2.rectangle(jpg_to_detect, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
                    cv2.putText(jpg_to_detect, class_name, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
    logging.debug('************************model detection ends!************************')
                    

if __name__ == "__main__":
    work_proc(0, 0)










