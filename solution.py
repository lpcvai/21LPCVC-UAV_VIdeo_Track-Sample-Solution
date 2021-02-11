import cv2
import numpy as np


#Constants



#Input Functions for Sample Solution





























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
        


