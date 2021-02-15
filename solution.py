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
    def __init__(self, color):
        self.color = color
        self.closestPerson = 0

    def update_closest(self, bbox_xywh):
        return
        


