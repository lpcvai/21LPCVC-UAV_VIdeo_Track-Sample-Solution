import cv2
import numpy as np
import pandas as pd
import torch

#Constants



#Input Functions for Sample Solution

def load_labels(file_name, image_width, image_height, frame_number=-1):
    '''
    return pandas DF when frame_number is -1
    return pytorch tensor when frame_number is a valid frame number
    '''
    data = pd.read_csv(file_name, sep=' ')

    data['X'] = data['X'].apply(lambda x: x*image_width)
    data['Y'] = data['Y'].apply(lambda x: x*image_height)
    data['Width'] = data['Width'].apply(lambda x: x*image_width)
    data['Height'] = data['Height'].apply(lambda x: x*image_height)

    if frame_number==-1:
        return data
    frame = data[(data["Frame"]==frame_number)]
    pt_frame = torch.tensor(frame[["Class","ID","X","Y","Width","Height"]].values)
    return pt_frame





#Output Functions for Sample Solution
def detect_catches(image, bbox_xyxy, classes, ids, colorDict):
    ball_detect = [None] * len(classes)
    detected_balls = {}
    bbox_offset = 5

    for i in range(len(classes)):
        ball_detect[i] = ''

        #Checks if the class is a ball (1)
        if (classes[i] == 1): 
            
            xmin = int(bbox_xyxy[i][0])
            ymin = int(bbox_xyxy[i][1])
            xmax = int(bbox_xyxy[i][2])
            ymax = int(bbox_xyxy[i][3])

            #Get center of bounding box
            X = int(((xmax - xmin) / 2) + xmin)
            Y = int(((ymax - ymin) / 2) + ymin)


            #Extract region of interest HSV values
            #Image values are (height, width, colorchannels)
            roi_bgr = image[(Y - bbox_offset):(Y + bbox_offset), (X - bbox_offset):(X + bbox_offset)]

            #Convert BGR image to HSV image
            roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
            hue  = np.mean(roi_hsv[:,:,0])
            sat = np.mean(roi_hsv[:,:,1])
            val   = np.mean(roi_hsv[:,:,2])


            #Check if the color is in a specified range
            ball_color = (hue, sat, val)

            for color in colorDict:
                upper = colorDict[color][0]
                lower = colorDict[color][1]

                if (ball_color <= upper) :
                    if (ball_color >= lower) :
                        detected_balls[color] = [X,Y]

                        txt = "Detected {colr}"
                        ball_detect[i] = txt.format(colr = color)
                        break

    return ball_detect




#Collision Detector

def detect_collisions(outputs):

    #diction format {id: [xcenter, ycenter, bboxwidth, bboxheight, class, identity, something}
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
    
    return(collisions)


def output_results(ball_detect, collisions, IDs, colorDict, frame_num):
    f = open("./outputs/catches.txt", "a")
    IDtoColor = {}
    for color in ball_detect:
        if(color != ''):
            IDtoColor[color.split(' ')[1]] = IDs[ball_detect.index(color)]

    f.write(f'{frame_num} | ')
    #frame_number = collisions
    for ball in colorDict:
        print(ball + " ", end='')
        item = IDtoColor.get(ball)

        if item:
            f.write(f'{item}  | ')
        else:
            f.write("0 | ")

    f.write("\n")
    
    f.close()
    pass


