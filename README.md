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

3. The trained weights have been provided. Make sure to use them to get the best detections.
The trained weights are called `best.pt` and they are under `yolov5/weights/best.pt`.
The trained weights were created using a dataset containing over 12,000 images. More stats can be found in 'yolov5/weights/stats.txt'
The DeepSORT weights are aready downloaded, however they can also be found [here](https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6).
They should be called `ckpt.t7` and place it under `deep_sort/deep/checkpoint/`


## Run
~~~
python3 track.py --source VIDEOSOURCE --weights yolov5/weights/best.pt --data yolov5/data/ballPerson.yaml --classes 0 1
~~~



## References
1) [Multi-class Yolov5 + Deep Sort with PyTorch](https://github.com/WuPedin/Multi-class_Yolov5_DeepSort_Pytorch)
2) [Yolov5_DeepSort_Pytorch](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch)   
3) [yolov5](https://github.com/ultralytics/yolov5)  
4) [deep_sort_pytorch](https://github.com/ZQPei/deep_sort_pytorch)       
5) [deep_sort](https://github.com/nwojke/deep_sort)   
