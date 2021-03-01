import pandas as pd
import argparse
import os
from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir('./') if isfile(join('./', f))]
files = sorted(onlyfiles)
with open("groundtruths.txt", "w") as myfile:
    myfile.write("Frame Class ID X Y Width Height\n")
parser = argparse.ArgumentParser()
parser.add_argument('--ball', nargs='+', type=int, help='ids for balls')
args = parser.parse_args()
ball_ids = args.ball
for f in files:
    if 'frame_' in f:
        data = open(f, "r")
        frame = int(f[6:12])
        for line in data.readlines():
            with open("groundtruths.txt", "a") as myfile:
                myfile.write(str(frame)+' '+str(int(int(line[:2]) in ball_ids))+' '+line)
table = pd.read_csv("groundtruths.txt", sep=' ')
table.to_csv('groundtruths.csv', sep=',', index=False)
os.remove("groundtruths.txt")
        
