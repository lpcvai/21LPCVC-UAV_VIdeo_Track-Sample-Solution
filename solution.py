import cv2
import numpy as np
import pandas as pd
import torch
import csv
import statistics
from os import path

#Constants
current_file_name = ''
current_file_data = None



#Input Functions for Sample Solution

def load_labels(file_name, image_width, image_height, frame_number=-1):
    '''
    Author: 
        Ziteng Jiao

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
def detect_catches(image, bbox_xyxy, classes, ids, frame_num, colorDict, frame_catch_pairs, ball_person_pairs, colorOrder):
    #Create a list of bbox centers and ranges
    bbox_XYranges = bbox_xyxy2XYranges(bbox_xyxy)
    

    #Detect the color of each ball and return a dictionary matching id to color
    detected_ball_colors = detect_colors(image, bbox_XYranges, classes, ids, colorDict)

    #Detect collison between balls and people
    collisions = detect_collisions(classes, ids, frame_num, bbox_XYranges, detected_ball_colors)

    #Update dictionary pairs
    frame_catch_pairs, ball_person_pairs = update_dict_pairs(frame_num, collisions, frame_catch_pairs, ball_person_pairs, colorOrder)
    bbox_strings = format_bbox_strings(ids, classes, detected_ball_colors, collisions)

    return (bbox_strings, frame_catch_pairs, ball_person_pairs)


def detect_colors(image, bbox_XYranges, classes, ids, colorDict):
    detected_ball_colors = {}
    bbox_offset = 5

    for i in range(len(classes)):

        #Checks if the class is a ball (1)
        if (classes[i] == 1): 
            #Extract region of interest HSV values
            #Image values are (height, width, colorchannels)
            X = bbox_XYranges[i][0]
            Y = bbox_XYranges[i][1]
            roi_bgr = image[(Y - bbox_offset):(Y + bbox_offset), (X - bbox_offset):(X + bbox_offset)]


            #Convert BGR image to HSV image
            roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
            hue  = np.mean(roi_hsv[:,:,0])
            sat = np.mean(roi_hsv[:,:,1])
            val   = np.mean(roi_hsv[:,:,2])
            ball_color = (hue, sat, val)


            #Check if the color is in a specified range
            for color in colorDict:
                upper = colorDict[color][0]
                lower = colorDict[color][1]

                if (ball_color <= upper) :
                    if (ball_color >= lower) :
                        detected_ball_colors[ids[i]] = [color, bbox_XYranges[i][0], bbox_XYranges[i][1]]
                        break

    return detected_ball_colors


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
        X = int(((xmax - xmin) / 2) + xmin)
        Y = int(((ymax - ymin) / 2) + ymin)

        #Create a range for collison detection
        X_range = (X - ((xmax - xmin) / 2), X + ((xmax - xmin) / 2))
        Y_range = (Y - ((ymax - ymin) / 2), Y + ((ymax - ymin) / 2))

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
            tmp = tmp + str(ball_person_pairs[color]) + ' '
        frame_catch_pairs.append([frame_num, tmp])

    return (frame_catch_pairs, ball_person_pairs)


def write_catches(output_path, frame_catch_pairs, colorOrder):
    colorOrder.insert(0, "frame")
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(colorOrder)
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







