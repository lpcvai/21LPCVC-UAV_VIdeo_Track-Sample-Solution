import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path+'/yolov5')

import argparse
import cv2
import torch
import torch.backends.cudnn as cudnn
import solution
import track
import time

from yolov5.utils.general import check_img_size
from yolov5.utils.torch_utils import select_device

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)



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




def draw_boxes(img, bbox, cls_names, scores, ball_detect, id_mapping, identities=None, offset=(0,0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        
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




def default_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=dir_path + '/yolov5/weights/best.pt', help='model.pt path')
    parser.add_argument('--data', type=str, default=dir_path + '/ballPerson.yaml', help='data yaml path')
    parser.add_argument('--source', type=str, default=dir_path + '/inference/images', help='source')
    parser.add_argument('--output', type=str, default=dir_path + '/outputs', help='output folder')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, default=[0, 1], help='filter by class') #Default [0]
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default=dir_path + "/deep_sort/configs/deep_sort.yaml")
    parser.add_argument('--groundtruths', default= dir_path + '/inputs/groundtruths.txt', help='path to the groundtruths.txt or \'disable\'')
    parser.add_argument('--save-img', action='store_true', help='save video to outputs')
    parser.add_argument('--skip-frames', type=int, default=1, help='number of frames skipped after each frame scanned')
    return parser




def main(vid_src=None, grd_src=None):
    '''
    Main function that will be ran when performing submission testing. 
    We will provide the path of the video and groundtruths when testing.
    argv[1] = video path (--source input)
    argv[2] = groundtruths path (--groundtruths input)
    '''
    parser = default_parser()
    if vid_src == None and grd_src == None:
        vid_src = sys.argv[1]
        grd_src = sys.argv[2]
    args = parser.parse_args(args=['--source', vid_src, '--groundtruths', grd_src, '--output', './outputs'])
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
        track.detect(args, device, half, colorDict, save_img=False)




if __name__ == '__main__':
    parser = default_parser()
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

    t0 = time.perf_counter()   
    with torch.no_grad():
        track.detect(args, device, half, colorDict, save_img=args.save_img)
    t1 = time.perf_counter()


    print('Total Runtime = %.2f' % (t1 - t0))

