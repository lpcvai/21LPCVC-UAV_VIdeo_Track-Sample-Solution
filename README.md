# Drone-Vision-Spring-2021-Sample-Solution Development


## Prepare 
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
    conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
    pip install -r requirements.txt
    ```

3.Download new weights or use the default yolov5s.pt
I already put the `yolov5s.pt` inside. If you need other models, 
please go to [official site of yolov5](https://github.com/ultralytics/yolov5). 
and place the downlaoded `.pt` file under `yolov5/weights/`.   
And I also aready downloaded the deepsort weights. 
You can also download it from [here](https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6), 
and place `ckpt.t7` file under `deep_sort/deep/checkpoint/`


## Run
~~~
# on video file
python main.py --input_path [VIDEO_FILE_NAME]
~~~



## References
1) [DeepSORT_YOLOv5_Pytorch](https://github.com/HowieMa/DeepSORT_YOLOv5_Pytorch)
2) [Yolov5_DeepSort_Pytorch](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch)   
3) [yolov5](https://github.com/ultralytics/yolov5)  
4) [deep_sort_pytorch](https://github.com/ZQPei/deep_sort_pytorch)       
5) [deep_sort](https://github.com/nwojke/deep_sort)   


Note: please follow the [LICENCE](https://github.com/ultralytics/yolov5/blob/master/LICENSE) of YOLOv5! 
