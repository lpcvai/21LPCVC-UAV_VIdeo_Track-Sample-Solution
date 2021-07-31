import cv2
import numpy as np
import pandas as pd
import torch
import csv
import statistics
from os import path
from yolov5.utils.datasets import LoadImages

#Constants
current_file_name = ''
current_file_data = None



#Input Functions for Sample Solution
def load_labels(file_name, image_width, image_height, frame_number=-1):
    '''
    Parameter:
        file_name:      path to the label file. groundtruths.txt
        image_width:    the width of image (video frame)
        image_height:   the height of image (video frame)
        frame_number:   the specific frame number that we want
                        if we want the whole label table the this should be -1
                        the default value is -1
    Return:
        When frame_number is -1:
            type:       pandas DataFrame 
            content:    all labels
            format:     ["Frame", "Class","ID","X","Y","Width","Height"]
        When frame_number is not -1:
            type:       pytorch tensor
            content:    coordinates of objects in the requested frame 
                        empty tensor if the requested frame doesn't exist in the label file
            format:     ["Class","ID","X","Y","Width","Height"]
    '''
    # data = pd.read_csv(file_name, sep=' ')
    global current_file_name
    global current_file_data
    if not path.exists(file_name):
        print("The file", file_name, "doesn't exist.")
        exit(1)
    if file_name != current_file_name:
        current_file_name = file_name
        current_file_data = pd.read_csv(current_file_name, sep=',')
        current_file_data['X'] = current_file_data['X'].apply(lambda x: x*image_width)
        current_file_data['Y'] = current_file_data['Y'].apply(lambda x: x*image_height)
        current_file_data['Width'] = current_file_data['Width'].apply(lambda x: x*image_width)
        current_file_data['Height'] = current_file_data['Height'].apply(lambda x: x*image_height)

    if frame_number==-1:
        return current_file_data
    frame = current_file_data[(current_file_data["Frame"]==frame_number)]
    pt_frame = torch.tensor(frame[["Class","ID","X","Y","Width","Height"]].values)
    return pt_frame




#Output Functions for Sample Solution
def detect_catches(image, bbox_xyxy, classes, ids, frame_num, colorDict, frame_catch_pairs, ball_person_pairs, colorOrder, save_img):
    #Create a list of bbox centers and ranges
    bbox_XYranges = bbox_xyxy2XYranges(bbox_xyxy)
    
    #Detect the color of each ball and return a dictionary matching id to color
    detected_ball_colors = detect_colors(image, bbox_XYranges, classes, ids, colorDict, save_img)

    #Detect collison between balls and people
    collisions = detect_collisions(classes, ids, frame_num, bbox_XYranges, detected_ball_colors)

    #Update dictionary pairs
    frame_catch_pairs, ball_person_pairs = update_dict_pairs(frame_num, collisions, frame_catch_pairs, ball_person_pairs, colorOrder)
    bbox_strings = format_bbox_strings(ids, classes, detected_ball_colors, collisions)

    
    return (bbox_strings, frame_catch_pairs, ball_person_pairs)
    



def detect_colors(image, bbox_XYranges, classes, ids, colorDict, save_img):
    #Cross_size is the number of radial members around the center
    #Bbox_offset is the radius of the bbox
    detected_ball_colors = {}
    det_clr = []
    bbox_offset = -1
    size = 5

    for i in range(len(classes)):
        #Checks if the class is a ball (1)
        if (classes[i] == 1): 
            #Extract region of interest HSV values
            area_colors = create_grid(image, bbox_offset, bbox_XYranges[i], size, save_img)

            #Check if the color is in a specified range
            result, color = check_color(colorDict, area_colors, det_clr)

            if (result == True):
                det_clr.append(color)
                detected_ball_colors[ids[i]] = [color, bbox_XYranges[i][0], bbox_XYranges[i][1]]
          
    return detected_ball_colors




def check_color(colorDict, area_colors, det_clr):
    #Count each area's color
    color_counts = dict.fromkeys(colorDict.keys(), 0)
    for color in colorDict:
        if (color not in det_clr):
            tmp = (area_colors>=colorDict[color][1]) == (area_colors<=colorDict[color][0])
            color_counts[color] = np.count_nonzero(tmp.all(axis=-1))

    most_color = max(color_counts, key=color_counts.get)
    return (True, most_color)




def get_roi_colors(image, bbox_XYranges, bbox_offset, size, save_img):
    #Get HSV values from image
    roi_arr = create_grid(image, bbox_offset, bbox_XYranges, size, save_img)
                
    #Return if area is 1 pixel
    if (bbox_offset == -1):
        return roi_arr

    #Average each area if area is greater than 1 pixel
    arr_size = len(roi_arr)
    area_clrs = np.empty((arr_size,  3), dtype=np.uint8)

    for i in range(arr_size):
        hue = np.mean(roi_arr[i][:,:,0])
        sat = np.mean(roi_arr[i][:,:,1])
        val = np.mean(roi_arr[i][:,:,2])

        ball_color = (hue, sat, val)
        area_clrs[i] = np.asarray(ball_color)


    #Sort areas by difference from the mean
    hues = area_clrs[:,0]
    avg_hue = int(np.mean(hues))
    avg_diff = np.absolute(np.subtract(hues, avg_hue, dtype=np.int16))
    area_clrs = area_clrs[np.argsort(avg_diff, axis=0)]

    area_colors = [(area[0], area[1], area[2]) for area in area_clrs]

    return area_colors




def create_grid(image, bbox_offset, bbox_XYranges, size, save_img):
    #Creates a crosshair centered in the bbox with seperate areas
    X = bbox_XYranges[0]
    Y = bbox_XYranges[1]

    #Divides bounding box into subsections
    num_splits = ((size - 1) * 2) + 4
    X_step = (bbox_XYranges[2][1] - bbox_XYranges[2][0]) // num_splits
    Y_step = (bbox_XYranges[3][1] - bbox_XYranges[3][0]) // num_splits

    width = (size + size + 1)
    num_roi =  width * width
    x = size * -1
    y = size * -1

    #Create pixel grid
    if (bbox_offset == -1):
        x1 = (np.arange(x, size+1) * X_step) + X
        y1 = (np.arange(y, size+1) * Y_step) + Y
        roi_arr = np.vstack(cv2.cvtColor(image[y1][:, x1], cv2.COLOR_BGR2HSV))
    

    #Create wide grid using old method
    if (bbox_offset != -1):
        bbox = []
        roi_arr = np.empty((num_roi, bbox_offset*2, bbox_offset*2, 3), dtype=np.uint8)
        for i in range(width):
            x1 = (x * X_step) + X
            for j in range(width):
                y1 = (y * Y_step) + Y
                ind = (i * width) + j
                roi_arr[ind] = image[(y1 - bbox_offset):(y1 + bbox_offset), (x1 - bbox_offset):(x1 + bbox_offset)]
                bbox.append(((x1 - bbox_offset),(y1 - bbox_offset),(x1 + bbox_offset),(y1 + bbox_offset)))
                y = y + 1
            y = size * -1
            x = x + 1
        #Convert values to HSV
        for i in range(num_roi):
            tmp = roi_arr[i]
            roi_arr[i] = cv2.cvtColor(tmp, cv2.COLOR_BGR2HSV)

    if (save_img):
        if (bbox_offset == -1):
            bbox = []
            for i in range(width):
                for j in range(width):
                    bbox.append((x1[i], y1[j], x1[i], y1[j]))

        draw_testingboxes(image, bbox)
            
    return roi_arr




def bbox_xyxy2XYranges(bbox_xyxy):
    bbox_XYranges = []

    #Create list of bbox centers and ranges
    for box in bbox_xyxy:
        #Get bbox corners
        xmin = int(box[0])
        ymin = int(box[1])
        xmax = int(box[2])
        ymax = int(box[3])

        #Get center of bounding box
        X = ((xmax - xmin) // 2) + xmin
        Y = ((ymax - ymin) // 2) + ymin

        #Create a range for collison detection
        X_range = (xmin, xmax)
        Y_range = (ymin, ymax)

        bbox_XYranges.append([X, Y, X_range, Y_range])

    return bbox_XYranges




def format_bbox_strings(ids, classes, detected_ball_colors, collisions):
    bbox_strings = [None] * len(classes)

    for i in range(len(classes)):
        #Person bbox info
        if (ids[i] in collisions):
            color = collisions[ids[i]]
            txt = 'Holding {color}'.format(color = color)

        #Ball bbox info    
        elif (ids[i] in detected_ball_colors):
            color = detected_ball_colors[ids[i]][0]
            txt = 'Detected {color}'.format(color = color)
        else:
            txt = ''
        bbox_strings[i] = txt

    return bbox_strings




def detect_collisions(classes, ids, frame_num, bbox_XYranges, detected_ball_colors):
    #collisions = {'id' : color, ....}
    collisions = {}
    #maxId = value after maxID is likely not tracked correctly
    maxId = 8

    for i in range(len(classes)):
        #Check if a person
        if ((classes[i] == 0) and (ids[i] < maxId)):

            #Get persons bbox range
            person_X_range = bbox_XYranges[i][2]
            person_Y_range = bbox_XYranges[i][3]

            #Check if the center of a ball is in a persons bounding box
            #detected_ball_colors = {'id' : [color, X, Y], ...}
            for ball in detected_ball_colors:
                ball_color = detected_ball_colors[ball][0]
                ball_X = detected_ball_colors[ball][1]
                ball_Y = detected_ball_colors[ball][2]

                if (ball_X >= person_X_range[0] and ball_X <= person_X_range[1] and ball_Y >= person_Y_range[0] and ball_Y <= person_Y_range[1] and (ball_color not in collisions.values())):
                    collisions[ids[i]] = ball_color
                    break

    return collisions




def update_dict_pairs(frame_num, collisions, frame_catch_pairs, ball_person_pairs, colorOrder):
    updateFrames = 0

    for person in collisions:
        color = collisions[person]
        tmp = {}
        #Ball color has not been held yet
        if (color not in ball_person_pairs):
            ball_person_pairs[color] = person

        #Ball is held by a new person 
        elif (ball_person_pairs[color] != person):
            ball_person_pairs[color] = person
            updateFrames = 1

    if (updateFrames):
        tmp = ''
        for color in colorOrder:
            if color in ball_person_pairs:
                tmp = tmp + str(ball_person_pairs[color]) + ' '
            else:
                tmp = tmp + '0' + ' '
        frame_catch_pairs.append([frame_num, tmp])

    return (frame_catch_pairs, ball_person_pairs)




def write_catches(output_path, frame_catch_pairs, colorOrder, colorDict):
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        possible_clrs = colorDict.keys()
        ball_ids = [colorDict[color][2] for color in possible_clrs]
        ball_ids.sort()
        ball_ids.insert(0, "frame")
        writer.writerow(ball_ids)
        frame_catch_pairs = smooth_frame_pairs(frame_catch_pairs)
        for i in range(len(frame_catch_pairs)):
            frame = frame_catch_pairs[i][0]
            pairs = frame_catch_pairs[i][1].split(' ')
            pairs.insert(0, frame)
            writer.writerow(pairs)

    return
        



def smooth_frame_pairs(frame_catch_pairs):
    max_diff = 5 
    size = len(frame_catch_pairs)
    smooth_pairs = []

    i = 0
    while i < size:
        frame = frame_catch_pairs[i][0]

        #Check if next item is in range
        if((i+1) < size):
            diff = frame_catch_pairs[i+1][0] - frame

            #Check if next frame is close
            if(diff < max_diff):
                color_ids = [[],[],[],[],[],[]]
                tmp_frames = frame_catch_pairs[i:]
                nxt_i = i

                for cur_frame in tmp_frames:
                    cur_ids = cur_frame[1][:-1]
                    cur_ids = cur_ids.split(' ')
                    cur_dif = cur_frame[0] - frame

                    if(cur_dif < max_diff):
                        for k in range(len(cur_ids)):
                            color_ids[k].append(cur_ids[k])
                        nxt_i = nxt_i + 1
                    else:
                        break
            
                tmp = ''
                for j in range(len(color_ids)):
                    mode = statistics.mode(color_ids[j])
                    tmp = tmp + mode + ' '
            
                i = nxt_i
                smooth_pairs.append([frame,tmp]) 
            else:
                smooth_pairs.append(frame_catch_pairs[i])
                i = i + 1

        else:
            smooth_pairs.append(frame_catch_pairs[i])
            i = i + 1

    return smooth_pairs



#Color Dict Functions
def default_colorDict():
    #Color dictonary for ball tracking where red : [(upper), (lower)] in HSV values
    #Static color definition is used when finding the dynamic color dict, or as a fallback. 
    #Use https://www.rapidtables.com/web/color/RGB_Color.html for hue 

    #Tolerance / range for each color
    hueOffset = 9
    satOffset = 75
    valOffset = 75

    #BGR Values for each color tested
    yellowBGR = np.uint8([[[ 81, 205, 217]]])
    redBGR    = np.uint8([[[ 78,  87, 206]]])
    blueBGR   = np.uint8([[[197, 137,  40]]])
    greenBGR  = np.uint8([[[101, 141,  67]]])
    orangeBGR = np.uint8([[[ 84, 136, 227]]])
    purpleBGR = np.uint8([[[142,  72,  72]]])


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
        "red"    : [colorListHSV[1][0], colorListHSV[1][1]],
        "purple" : [colorListHSV[5][0], colorListHSV[5][1]],
        "blue"   : [colorListHSV[2][0], colorListHSV[2][1]],
        "green"  : [colorListHSV[3][0], colorListHSV[3][1]],
        "yellow" : [colorListHSV[0][0], colorListHSV[0][1]],
        "orange" : [colorListHSV[4][0], colorListHSV[4][1]],  
    }

    return colorDict




def generateDynColorDict(groundtruths_path, clr_offs, args):
    dataset = LoadImages(args.source, img_size=args.img_size)
    frame_num = 0
    detected_ball_colors = {}
    detected_ball_ids = {}
    bbox_offset = 5
    cross_size = 3
    #view_gt = False
    static_colorDict = default_colorDict()

    for _, _, im0, _ in dataset:
        det_clr = []
        img_h, img_w, _ = im0.shape
        groundtruths = load_labels(groundtruths_path, img_w, img_h, frame_num)
        if(groundtruths.shape[0] == 0):
            break
        
        bbox_XYranges = gtballs_2XYranges(groundtruths)
        
        for bbox in bbox_XYranges:
            bbox_id = bbox[4]
            area_colors = get_roi_colors(im0, bbox, bbox_offset, cross_size, False)
            color, colorVals = get_colors(static_colorDict, area_colors, det_clr)
            
            if (color != 'NULL'):
                det_clr.append(color)
                if (color not in detected_ball_colors):
                    detected_ball_colors[color] = [colorVals]
                    detected_ball_ids[color]    = [bbox_id]
                else:
                    detected_ball_colors[color].append(colorVals)
                    detected_ball_ids[color].append(bbox_id)

        if (frame_num == 10):
            frame_num = int((dataset.nframes / 2) - 5)
            dataset.frame = frame_num
        else:
            frame_num += 1

    num_balls = len(bbox_XYranges)
    num_colors = len(detected_ball_colors)
    color_arr = np.empty(num_colors, dtype="U10")
    len_pairs = np.empty(num_colors, dtype=np.int32)

    i = 0
    for color in detected_ball_colors:
        color_arr[i] = color
        len_pairs[i] = len(detected_ball_colors[color])
        i += 1
    
    color_arr = color_arr[np.argsort(-1*len_pairs, axis=0)]
    dyn_colorDict = create_dyn_dict(color_arr, detected_ball_colors, detected_ball_ids, clr_offs, num_balls)

    print('\nDynamic Dictionary Created...')
    return dyn_colorDict




def create_dyn_dict(color_arr, detected_ball_colors, detected_ball_ids, offsets, num_balls):
    hueOffset = offsets[0]
    satOffset = offsets[1]
    valOffset = offsets[2]
    dyn_colorDict = {}

    for i in range(num_balls):
        #Average color values
        tmp = detected_ball_colors[color_arr[i]]
        hsv = tuple([int(sum(clmn) / len(tmp)) for clmn in zip(*tmp)])

        #Create ranges for color
        upper = np.asarray((hsv[0] + hueOffset, hsv[1] + satOffset, hsv[2] + valOffset), dtype=np.int16)
        lower = np.asarray((hsv[0] - hueOffset, hsv[1] - satOffset, hsv[2] - valOffset), dtype=np.int16)
        clr_id = detected_ball_ids[color_arr[i]]
        clr_id = max(clr_id, key = clr_id.count)
        dyn_colorDict[color_arr[i]] = [upper, lower, clr_id]

    return dyn_colorDict




def get_colors(colorDict, area_colors, det_clr):
    #Count each area's color
    color_counts = dict.fromkeys(colorDict.keys(), 0)
    for color in colorDict:
        upper = colorDict[color][0]
        lower = colorDict[color][1]

        if (color not in det_clr):
            for a_clr in area_colors:
                if ((a_clr <= upper) and (a_clr >= lower)):
                    color_counts[color] = color_counts[color] + 1
                        
    #Find most prominent color
    most_color = max(color_counts, key=color_counts.get)
    upper = colorDict[most_color][0]
    lower = colorDict[most_color][1]
    valid_areas = []

    for a_clr in area_colors:
        if ((a_clr <= upper) and (a_clr >= lower)):
            valid_areas.append(a_clr)

    if (len(valid_areas) == 0):
        most_color = 'NULL'
        colorVals = (0,0,0)
        return most_color, colorVals


    colorVals = tuple([int(sum(clmn) / len(valid_areas)) for clmn in zip(*valid_areas)])

    return most_color, colorVals




def gtballs_2XYranges(groundtruths):
    '''
    Parameter:
        groundtruths:   pytorch tensor ["Class","ID","X","Y","Width","Height"]
    Return:
        bbox_XYranges:  [[X, Y, X_rngs, Y_rngs, id], [X, Y, X_rngs, Y_rngs, id], ...]
    '''
    bbox_XYranges = []
    for truth in groundtruths:
        #Check if the class is a ball
        if (truth[0] == 1):
            X = int(truth[2])
            Y = int(truth[3])
            width = truth[4]
            height = truth[5]
            X_rngs = (int(X - (width/2)), int(X + (width/2)))
            Y_rngs = (int(Y - (height/2)), int(Y + (height/2)))

            bbox_XYranges.append([X, Y, X_rngs, Y_rngs, int(truth[1])])
    return bbox_XYranges



#Misc Functions
def draw_testingboxes(img, bbox):
    for box in bbox:
        x1, y1, x2, y2 = box
        color = (0,0,255)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
    return