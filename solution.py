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
                        
                        txt = "Detected {colr}"
                        ball_detect[i] = txt.format(colr = color)

    return ball_detect










def create_ballDict(colorDict):
    ballDict = {}
    
    for color in colorDict.keys:
        ballDict[color] = Ball(color)
    return ballDict



def update_colored_ball(ballDict, colorDict, x1, x2, y1, y2):
    #1. Get color in the center of the bbox

    #2. Compare returned color to ranges in colorDict

    #3. Update specific ball in ballDict
    ballDict[color].update_closest()


    return



#Ball class
class Ball : 
    ballDict = {}

    def __init__(self, color):
        self.color = color
        self.closestPerson = 0

    def update_ballDict(self):
        return

    def update_closest(self, bbox_xywh):
        return
        


