import argparse
import os
import platform
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
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
from solutionCopy import Track

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
id_mapping = {}
groundtruths_path = None


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
        # Contributor YKS
        
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


def detect(opt, device, save_img=False):
    out, source, weights, view_img, save_txt, imgsz, skipLimit = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.skip_frames
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    colorOrder = ['red', 'purple', 'blue', 'green', 'yellow', 'orange']
    frame_num = 0
    framestr = 'Frame {frame}'
    fpses = []
    frame_catch_pairs = []
    ball_person_pairs = {}

    for color in colorDict:
        ball_person_pairs[color] = 0
    
    print("FRAMES SKIPPED: " + str(skipLimit))

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
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]


    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    #Skip Variables
    skipThreshold = 0 #Current number of frames skipped


    for path, img, im0s, vid_cap in dataset:
        """
        path: path to the image
        img: current image
        im0s: original image with additional drawn bounding boxes
        vid_cap: an opencv .VideoCapture
        """ 
        img_clone = cv2.resize(im0s,(640,360))
        img_clone_np = np.asarray(img_clone)
        #img_clone_np = np.swapaxes(img_clone_np, 0, -1)
        #img_clone_np = np.swapaxes(img_clone_np, 0, 1)
        img_clone_np = cv2.cvtColor(img_clone_np, cv2.COLOR_BGR2GRAY)
        """
        if frame_num == 11: # create list of Track objects; uses bbox_xywh of the 10th frame
            tList = np.empty([1,len(bbox_xywh)]) #create empty array for track objects
            for bbox in bbox_xywh:
                tType =  # get David to help figure out this section
                tID =    # get David to help figure out this section
                tColor = # get David to help figure out this section
                tObj = Track(tType, tID, tColor, bbox)
                tList = np.append(tList, tObj)
                countinue
        """
            
        ## skipped frame
        if frame_num > 10 and skipThreshold < skipLimit: 
            p, s, im0 = path, '', im0s
            skipThreshold += 1
            frame_num += 1
            '''
            bbox_predicted = [] #or np.empty((0,4), int)
            clses = []
            identities = []
            for i in tList:
                if(i.tType==1):
                    i.predict_bbox(img_old_np, img_clone_np)
                    bbox_predicted.append(i.bbox*6)
                    clses.append(i.tType)
                    identities.append(i.tID)
                else:
                    bbox_predicted.append(i.bbox*6)
                    clses.append(i.tType)
                    identities.append(i.tID)
            
            
            mapped_id_list = []
            for ids in identities:
                if(ids in id_mapping):
                    mapped_id_list.append(int(id_mapping[ids]))
                else:
                    mapped_id_list.append(ids)
            img_old_np = img_clone_np
            ball_detect, frame_catch_pairs, ball_person_pairs = solution.detect_catches(im0, bbox_predicted, clses, mapped_id_list, frame_num, colorDict, frame_catch_pairs, ball_person_pairs, colorOrder, save_img)
            draw_boxes(im0, bbox_predicted, [names[i] for i in clses], scores, ball_detect, identities)
            tmp = framestr.format(frame = frame_num)
            t_size = cv2.getTextSize(tmp, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
            cv2.putText(im0, tmp, (0, (t_size[1] + 10)), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
            
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

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
            '''
            continue

        ##Non-skipped frame
        img_old_np = img_clone 
        skipThreshold = 0

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

                bbox_xywh = []
                confs = []
                clses = []
                tList = np.empty([1,9]) #create empty array for track objects
                # Write results
                for *xyxy, conf, cls in det:
                    
                    img_h, img_w, _ = im0.shape  # get image shape
                    x_c, y_c, bbox_w, bbox_h = bbox_rel(img_w, img_h, *xyxy)
                    obj = [x_c, y_c, bbox_w, bbox_h]
                    bbox_xywh.append(obj)
                    confs.append([conf.item()])
                    clses.append([cls.item()])
                    
              
                    
                xywhs = torch.Tensor(bbox_xywh)
                confss = torch.Tensor(confs)
                clses = torch.Tensor(clses)
                # Pass detections to deepsort
                outputs = []
                global groundtruths_path
                if not 'disable' in groundtruths_path:
                    # print('\nenabled', groundtruths_path)
                    groundtruths = solution.load_labels(groundtruths_path, img_w,img_h, frame_num)
                    if (groundtruths.shape[0]==0):
                        outputs = deepsort.update(xywhs, confss, clses, im0)
                    else:
                        # print(groundtruths)
                        xywhs = groundtruths[:,2:]
                        tensor = torch.tensor((), dtype=torch.int32)
                        confss = tensor.new_ones((groundtruths.shape[0], 1))
                        clses = groundtruths[:,0:1]
                        outputs = deepsort.update(xywhs, confss, clses, im0)
                    
                    
                    if frame_num >= 2:
                        for real_ID in groundtruths[:,1:].tolist():
                            for DS_ID in xyxy2xywh(outputs[:, :5]):
                                if (abs(DS_ID[0]-real_ID[1])/img_w < 0.005) and (abs(DS_ID[1]-real_ID[2])/img_h < 0.005) and (abs(DS_ID[2]-real_ID[3])/img_w < 0.005) and(abs(DS_ID[3]-real_ID[4])/img_w < 0.005):
                                    id_mapping[DS_ID[4]] = int(real_ID[0])
                else:
                    outputs = deepsort.update(xywhs, confss, clses, im0)

                # draw boxes for visualization
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, 4]
                    clses = outputs[:, 5]
                    scores = outputs[:, 6]
                    
                    #Temp solution to get correct id's 
                    mapped_id_list = []
                    for ids in identities:
                        if(ids in id_mapping):
                            mapped_id_list.append(int(id_mapping[ids]))
                        else:
                            mapped_id_list.append(ids)

                    ball_detect, frame_catch_pairs, ball_person_pairs = solution.detect_catches(im0, bbox_xyxy, clses, mapped_id_list, frame_num, colorDict, frame_catch_pairs, ball_person_pairs, colorOrder, save_img)
                    
                    t3 = time_synchronized()
                    draw_boxes(im0, bbox_xyxy, [names[i] for i in clses], scores, ball_detect, identities)
                else:
                    t3 = time_synchronized()


            #Draw frame number
            tmp = framestr.format(frame = frame_num)
            t_size = cv2.getTextSize(tmp, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
            cv2.putText(im0, tmp, (0, (t_size[1] + 10)), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)


            #Inference Time
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

            if frame_num >= 10:
                #use bbox_xywh to update object.bbox
                tList = []
                img_old_np = img_clone_np
                fast = cv2.FastFeatureDetector_create(1)
                for i in range(len(clses)): #SUBJECT TO CHANGE loop through each object
                        tType = clses[i]
                        tID = identities[i]
                        bbox = bbox_xyxy[i]/6
                        frameCrop = img_old_np[int(bbox[1]):int(bbox[1]+bbox[3]),int(bbox[0]):int(bbox[0]+bbox[2])]
                        featPoints  = fast.detect(frameCrop, None)
                        featPoints = np.float32([kp.pt for kp in featPoints])
                    #adjust coordinates with bounding box
                        adj = [bbox[0], bbox[1]]
                        if featPoints.shape != (0,):
                            featPoints += adj
                            featPoints = np.expand_dims(featPoints, axis=1)
                        else:
                            featPoints = []
                        tObj = Track(tType, tID, bbox,featPoints)
                        tList = np.append(tList, tObj)
            frame_num += 1
                    
        
        

    #t4 = time_synchronized()
    #avgFps = (sum(fpses) / len(fpses))
    #print('Average FPS = %.2f' % avgFps)
    #print('Total Runtime = %.2f' % (t4 - t0))
    
    outpath = os.path.basename(source)
    outpath = outpath[:-4]
    outpath = out + '/' + outpath + '_out.csv'
    print(outpath)
    solution.write_catches(outpath, frame_catch_pairs, colorOrder)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5/weights/best.pt', help='model.pt path')
    parser.add_argument('--data', type=str, default='ballPerson.yaml', help='data yaml path')
    parser.add_argument('--source', type=str, default='inference/images', help='source')
    parser.add_argument('--output', type=str, default='outputs', help='output folder')
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
    parser.add_argument('--groundtruths', default='./inputs/groundtruths.txt', help='path to the groundtruths.txt or \'disable\'')
    parser.add_argument('--save-img', action='store_true', help='save video to outputs')
    parser.add_argument('--skip-frames', type=int, default=1, help='number of frames skipped after each frame scanned')
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)

    groundtruths_path = args.groundtruths
    
    # Select GPU
    device = select_device(args.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    #Color tolerance for tracking 
    hueOffset = 2
    satOffset = 50
    valOffset = 50

    clr_offs = (hueOffset, satOffset, valOffset)
    
    colorDict = solution.generateDynColorDict(groundtruths_path, clr_offs, args)

    with torch.no_grad():
        detect(args, device, save_img=args.save_img)
