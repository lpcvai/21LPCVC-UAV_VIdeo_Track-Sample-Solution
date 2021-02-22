import argparse
import os
import platform
import shutil
import time
from pathlib import Path

import cv2
#import torch
#import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np

# https://github.com/pytorch/pytorch/issues/3678
import sys
sys.path.insert(0, './yolov5')

from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import LoadStreams, LoadImages
from yolov5.utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer)
from yolov5.utils.torch_utils import select_device, load_classifier, time_synchronized
from yolov5.utils.plots import plot_one_box

from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort

import yaml
import solution


palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
id_mapping = {}


def bbox_rel(image_width, image_height,  *xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, cls_names, scores, ball_detect, identities=None, offset=(0,0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        # The following line was problematic. With this labeling should be fixed
        # Contributor YKS
        # id = i + 1
        try:
            id = int(id_mapping[identities[i]]) if identities is not None else 0    
        except KeyError:
            id = int(identities[i]) if identities is not None else 0    
            
        color = compute_color_for_labels(id)
        label = '%s %d %s %d' % (ball_detect[i], id, cls_names[i], scores[i])
        label += '%'
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
        cv2.rectangle(img, (x1, y1),(x2,y2), color, 3)
        cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    #return img


def detect(opt, device, save_img=False):
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')
    
    frame_num = 0
    fpses = []

    # Read Class Name Yaml
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)
    names = data_dict['names']

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE, 
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE, 
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET, use_cuda=True)

    # Initialize
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    # Load model
    #google_utils.attempt_download(weights)
    model = torch.load(weights, map_location=device)['model'].float()  # load to FP32
    #model = torch.save(torch.load(weights, map_location=device), weights)  # update model if SourceChangeWarning
    # model.fuse()
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)


        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                #for c in det[:, -1].unique():
                    #n = (det[:, -1] == c).sum()  # detections per class
                    #s += '%g %ss, ' % (n, names[int(c)])  # add to string

                bbox_xywh = []
                confs = []
                clses = []

                # Write results
                for *xyxy, conf, cls in det:
                    
                    img_h, img_w, _ = im0.shape  # get image shape
                    
                    x_c, y_c, bbox_w, bbox_h = bbox_rel(img_w, img_h, *xyxy)
                    obj = [x_c, y_c, bbox_w, bbox_h]
                    bbox_xywh.append(obj)
                    confs.append([conf.item()])
                    clses.append([cls.item()])
                    
                    #if save_img or view_img:  # Add bbox to image
                        #label = '%s %.2f' % (names[int(cls)], conf)
                        #plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                
                xywhs = torch.Tensor(bbox_xywh)
                confss = torch.Tensor(confs)
                clses = torch.Tensor(clses)
                # Pass detections to deepsort
                outputs = []
                groundtruths = solution.load_labels("./inputs/groundtruths.txt",img_w,img_h,frame_num)
                if (groundtruths.shape[0]==0):
                    outputs = deepsort.update(xywhs, confss, clses, im0)
                else:
                    # print(groundtruths)
                    xywhs = groundtruths[:,2:]
                    tensor = torch.tensor((), dtype=torch.int32)
                    confss = tensor.new_ones((groundtruths.shape[0], 1))
                    clses = groundtruths[:,0:1]
                    outputs = deepsort.update(xywhs, confss, clses, im0)
                

                if frame_num == 2:
                    for DS_ID in xyxy2xywh(outputs[:, :5]):
                        for real_ID in groundtruths[:,1:].tolist():
                            if (abs(DS_ID[0]-real_ID[1])/img_w < 0.005) and (abs(DS_ID[1]-real_ID[2])/img_h < 0.005) and (abs(DS_ID[2]-real_ID[3])/img_w < 0.005) and(abs(DS_ID[3]-real_ID[4])/img_w < 0.005):
                                id_mapping[DS_ID[4]] = int(real_ID[0])

                # draw boxes for visualization
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, 4]
                    clses = outputs[:, 5]
                    scores = outputs[:, 6]
                    
                    ball_detect = solution.detect_catches(im0, bbox_xyxy, clses, identities, colorDict)
                    
                    draw_boxes(im0, bbox_xyxy, [names[i] for i in clses], scores, ball_detect, identities)
                    diction = {}
                    for i in outputs:
                        diction[i[4]] = [(i[0] + i[2])/2, (i[1] + i[3])/2, i[2] - i[0],i[3] - i[1], i[5], i[4], i[7]]
                    collisions = {}
                    for entry in diction:
                        xcenter = diction[entry][0]
                        ycenter = diction[entry][1]
                        x_range = (xcenter - diction[entry][2]/2 , xcenter + diction[entry][2]/2 )
                        y_range = (ycenter - diction[entry][3]/2 , xcenter + diction[entry][3]/2 )
                        for collider in diction:
                            colliderx = diction[collider][0]
                            collidery = diction[collider][1]
                            if entry != collider:
                                if colliderx > x_range[0] and colliderx < x_range[1] and collidery > y_range[0] and collidery < y_range[1] :
                                    if (diction[collider][4]) :
                                        collisions[diction[collider][5]] = [diction[entry][6], diction[entry][5]]
                    print(collisions)
                   

            #Inference Time
            t3 = time_synchronized()
            fps = (1/(t3 - t1))
            fpses.append(fps)
            print('FPS=%.2f' % fps)

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)
            frame_num += 1
                    
        #print('Inference Time = %.2f' % (time_synchronized() - t1))
        #print('FPS=%.2f' % (1/(time_synchronized() - t1)))

    avgFps = (sum(fpses) / len(fpses))
    print('Average FPS=%.2f' % avgFps)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5/weights/yolov5s.pt', help='model.pt path')
    parser.add_argument('--data', type=str, default='yolov5/data/data.yaml', help='data yaml path') # Class names
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    # class 0 is person
    parser.add_argument('--classes', nargs='+', type=int, default=[0], help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)
    print(args)
    
    # Select GPU
    device = select_device(args.device)
    import torch
    import torch.backends.cudnn as cudnn
    half = device.type != 'cpu'  # half precision only supported on CUDA

    #Color dictonary for ball tracking where red : (lowerbound, upperbound) in bgr values
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5/weights/best.pt', help='model.pt path')
    parser.add_argument('--data', type=str, default='yolov5/data/ballPerson.yaml', help='data yaml path')
    parser.add_argument('--source', type=str, default='inference/images', help='source')
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, default=[0, 1], help='filter by class') #Default [0]
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)
    print(args)
    
    # Select GPU
    device = select_device(args.device)
    import torch
    import torch.backends.cudnn as cudnn
    half = device.type != 'cpu'  # half precision only supported on CUDA

    
    #Color dictonary for ball tracking where red : [(upper), (lower)] in HSV values
    #Use https://www.rapidtables.com/web/color/RGB_Color.html for hue 
    hueOffset = 5
    satOffset = 50
    valOffset = 50


    #BGR Values for each color tested
    yellowBGR = np.uint8([[[157, 255, 249]]])
    redBGR    = np.uint8([[[131, 147, 225]]])
    blueBGR   = np.uint8([[[247, 204,  41]]])
    greenBGR  = np.uint8([[[  0, 128,   0]]])
    orangeBGR = np.uint8([[[ 80, 151, 236]]])
    purpleBGR = np.uint8([[[192, 116, 122]]])


    colorListBGR = [yellowBGR, redBGR, blueBGR, greenBGR, orangeBGR, purpleBGR]
    colorListHSVTmp = []
    colorListHSV = []


    #Convert BGR to HSV
    for bgr in colorListBGR:
        colorListHSVTmp.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV))
    

    #Create ranges based off offsets
    for i in range(len(colorListBGR)):
        hsv = colorListHSVTmp[i][0][0]
        upper = (hsv[0] + hueOffset, hsv[1] + satOffset, hsv[2] + valOffset)
        lower = (hsv[0] - hueOffset, hsv[1] - satOffset, hsv[2] - valOffset)
        colorListHSV.append([upper, lower])


    colorDict = {
        "yellow" : [colorListHSV[0][0], colorListHSV[0][1]],
        "red"    : [colorListHSV[1][0], colorListHSV[1][1]],
        "blue"   : [colorListHSV[2][0], colorListHSV[2][1]],
        "green"  : [colorListHSV[3][0], colorListHSV[3][1]],
        "orange" : [colorListHSV[4][0], colorListHSV[4][1]],
        "purple" : [colorListHSV[5][0], colorListHSV[5][1]]
    }
    
    with torch.no_grad():
        detect(args, device)
