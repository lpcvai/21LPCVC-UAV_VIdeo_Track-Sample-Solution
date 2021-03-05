import pandas as pd
import argparse
import os
from os import listdir
from os.path import isfile, join


def create_input_annotations(args):
    ball_ids = args.ball
    detect_middle = args.detmid
    source = args.source
    dirr = listdir(source)

    onlyfiles = [join(source, f) for f in dirr if isfile(join(source, f))]
    files = sorted(onlyfiles)
    with open("groundtruths.txt", "w") as myfile:
        myfile.write("Frame Class ID X Y Width Height\n")
    

    if (detect_middle):
        num_frames = 0
        
        for f in files:
            if 'frame_' in f:
                num_frames = num_frames + 1

        num_frames = num_frames - 1
        lower = round(num_frames/2) - 5

        selfiles = []
        for i in range(22):
            if (i > 10):
                frame = str(i + lower - 11)
            else:
                frame = str(i)

            frame = frame.zfill(6)
            fname = 'frame_' + frame + '.txt'
            selfiles.append(join(source, fname))
    
        files = selfiles



    for f in files:
        if 'frame_' in f:
            data = open(f, "r")
            frame = int(f[-10:-4])

            for line in data.readlines():
                with open("groundtruths.txt", "a") as myfile:
                    line = line.split()

                    if (int(line[0]) in ball_ids):
                        clss = '1'
                    else:
                        clss = '0'

                    out_txt = "{} {} {} {} {} {} {}\n".format(str(frame), clss, str(int(line[0]) + 1), line[1], line[2], line[3], line[4])
                    myfile.write(out_txt)


    table = pd.read_csv("groundtruths.txt", sep=' ')
    table.to_csv('groundtruths.csv', sep=',', index=False)
    os.remove("groundtruths.txt")
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ball', nargs='+', type=int, default=[5, 6, 7, 8, 9, 11], help='ids for balls')
    parser.add_argument('--detmid', action='store_true', help='Detects the middle groundtruth frames in dataset')
    parser.add_argument('--source', type=str, default='obj_train_data/', help='Location of yolo annotations')
    args = parser.parse_args()

    create_input_annotations(args)