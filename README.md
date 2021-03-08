# Drone-Vision-Spring-2021-Sample-Solution Development

## Installation 
### READ BEFORE RUNNING
The sample solution has been updated, so it is important that you delete your old conda environment and create a new one following the installation procedure.
To delete the old environment
```
conda env remove -n ENV_NAME
```


### Prepare 
1. Create a virtual environment and call it SampleSolution
    ```
    conda create --name SampleSolution python=3.8
    conda activate SampleSolution
    ```
2. Clone this repo to install the sample solution:
    ```
    git clone https://github.com/dual19/Drone-Vision-Spring-2021-Sample-Solution.git
    cd Drone-Vision-Spring-2021-Sample-Solution
    
    conda activate SampleSolution
    conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
    
    pip install -r requirements.txt
    ```
    
    
3. The trained weights are provided [here](https://purdue0-my.sharepoint.com/:f:/g/personal/hu440_purdue_edu/EuCYkSRgyXVCh8PwwsHZ9lYBNfI4A4cLgdi5sHIlRSsZCQ?e=yjoJ2P).
The weights are called `best.pt` and should be placed under `yolov5/weights/`.
Make sure to use them to get the best detections.
The trained weights were created using a dataset containing over 12,000 images. More stats on the dataset can be found in `yolov5/weights/stats.txt`.
Specific stats about the training session can be viewed [here](https://wandb.ai/dual19/YOLOv5/runs/2hkzouqz?workspace=user-dual19) if you're interested. 


4. The DeepSORT weights need to be downloaded; they can be found [here](https://purdue0-my.sharepoint.com/:u:/g/personal/hu440_purdue_edu/EYvoc5gij4dNpcGJ5jnBW94BP5H5LU_dcW0dHtm_lX8aBQ?e=s8j3LW).
They should be called `ckpt.t7` and place it under `deep_sort/deep_sort/deep/checkpoint/`

## Input and Output Files
### Inputs
1. The first input will be a video. The sample videos can be located [here](https://purdue0-my.sharepoint.com/:f:/g/personal/hu440_purdue_edu/EhgD1DpfGhRArmSJDetBQ5wBDdJTxT9djzZG3XTjkeObPw?e=IgBqLE).


2. The second input is a csv file, containing the first 10 frames for the solution to acquire the correct labels. An additional 10 frames will be provided in the middle of the video, to recalibrate the labels if some identity switching occurs. The format for the input file in `inputs/"videoname".csv` should be similar to the example below. NOTE: The delimiter between each value in the actual csv file will be a comma (","), the | is just for visualization. The bounding box coordinate system is based off of the YOLO annotation format. 
```
  Frame | Class |   ID  |   X   |   Y   | Width | Height|
----------------------------------------------------------
    0   |   0   |   1   |0.41015|0.39583|0.02031|0.03425|
    0   |   0   |   2   |0.36835|0.61990|0.04557|0.18055|
    0   |   1   |   3   |0.41015|0.39583|0.03593|0.16296|
    1   |   0   |   1   |0.52942|0.39583|0.02031|0.03425|
    1   |   0   |   2   |0.36835|0.61990|0.04557|0.18055|
    1   |   1   |   3   |0.52942|0.39537|0.03593|0.16296|


- Frame: The frame number of the annotation
- Class: 0 for person, 1 for sports ball

- X      = absolute_x / image_width
- Y      = absolute_y / image_height
- Width  = absolute_width /image_width
- Height = absolute_height /image_height
```



### Outputs
1. The only output from the solution should be a text file. This text file will include the location of every ball when a single ball has been caught. The format for the output file in `outputs/"videoname"_out.csv` should be similar to the example below. NOTE: The delimiter between each value in the actual csv file will be a comma (","), the | is just for visualization.

```
  Frame | Yellow | Orange |  Red  | Purple |  Blue  | Green |
----------------------------------------------------------
    5   |   0    |   1    |   5   |   2    |   4    |   0    - Person 4 catches blue
    30  |   0    |   3    |   5   |   2    |   4    |   0    - Person 3 catches orange
    49  |   0    |   3    |   1   |   2    |   4    |   0    - Person 1 catches red
    60  |   0    |   3    |   1   |   2    |   5    |   0    - Person 5 catches blue
```


## Run
~~~
python3 track.py --source VIDEOSOURCE --groundtruths PATHTOCSV
~~~



## References
1) [Multi-class Yolov5 + Deep Sort with PyTorch](https://github.com/WuPedin/Multi-class_Yolov5_DeepSort_Pytorch)
2) [Yolov5_DeepSort_Pytorch](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch)   
3) [Ultralytics Yolov5](https://github.com/ultralytics/yolov5)  
4) [Deep_SORT_Pytorch](https://github.com/ZQPei/deep_sort_pytorch)       
5) [Simple Online and Realtime Tracking with a Deep Association Metric](https://arxiv.org/abs/1703.07402)
