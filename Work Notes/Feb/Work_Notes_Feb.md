# 2.8 - 2/15
## Ziteng Jiao
* Input the label file:

    Finished the `load_labels()` function in the `solution.py` file.

    `load_labels()` function reads the label file, converts the relative coordinate to the real coordinate, then returns the requested frame as PyTorch tensor or returns the whole table as pandas DaraFrame.

* Substitute (x,y,w,h) with our label:

    In `track.py`, I defined a `id_mapping` dictionary, which maps the DeepSORT identities for objects to the number we assign to them.

    Instead of always feed the Yolo output into the DeepSORT, I modified the logic to:

        If the current frame exists in the groundtruths.txt file:
            then feed the coordinate of the bounding boxes into the DeepSORT 
            with a confidence of 1.
        Otherwise:
            Use the Yolo output as the input of DeepSORT.
        
* `id_mapping` dictionary:

    The value of this dictionary is assigned at the 3rd frame ( i.e. `frame_number == 2` ) because it looks like DeepSORT doesn't have an output for the first 2 frames. I will modify this so it considers more frames instead of only one frame.

    I compared the real_ID and the DeepSORT_ID. If their difference in x, y, width, height are all less than 0.5%, then I assume they are the same object. If they are the same object:

        id_mapping[DeepSORT_ID] = real_ID

    This way we can use `id_mapping[DeepSORT_ID]` to get the custom ID we want.

    
</br>

---
</br></br>

# 2/15 - 2/22
## Ziteng Jiao

* `load_labels()` function:
  * Description for parameters and return values is added.
  * Save the opened file to reduce file IO frequency.

* `id_mapping`:

    Now will update every time when the frame is contained in the groundtruths.txt.
    
    Value contradiction handling:   
    If the key is different from what we assigned, we need to decide which is the correct key.

    <img src="./img/Ziteng Jiao/0926_ball01Air_00_46_Before.png" alt="Before 00_46 occlusion" height="200"/>
    <img src="./img/Ziteng Jiao/0926_ball01Air_00_46_Happening.png" alt="During 00_46 occlusion" height="200"/>

    If person 6 is recognized by DeepSORT as person 2 after this occlusion, then we need the handle the value contradiction because one person is mapped to 2 real_IDs.

    I tried to reconstruct the `id_mapping` to solve this, but it will map IDs incorrectly even when value contradiction doesn't happen. So I removed that part for now. I plan to rewrite `id_mapping` as a function so it will be more flexible.

* Test on more videos:

    Works well on simple videos.
        
    Problem Observed: Our DeepSORT doesn't do well in re-identification.

    <img src="./img/Ziteng Jiao/0926_ball01Air_01_10_Before.png" alt="During 01_10" width="600"/>
    <img src="./img/Ziteng Jiao/0926_ball01Air_01_10_After.png" alt="After 01_10" width="600"/>

    Possible solution:
    * Recalibration.

        I've tried this.
        Won't work if new identities are created multiple times.

    * Improve DeepSORT.

        Time consuming.  
        May not find a solution.

    * Try other models.

        >Ayden: There’s a couple deep models that are better than what we are using right now to hopefully improve performance with regards to that.
    
    


</br>

---
</br></br>

# 2/23 - 3/1
## Ziteng Jiao
* `load_labels()` function:
  * Print an error message when the requested file doesn't exist.
* Re-check groundtruths:
  * The `id_mapping` will update after the frame provided in the middle for re-checking.

    Before re-check:
    <img src="./img/Ziteng Jiao/0926_ball01Air_01_00_Before.png" alt="During 01_10 occlusion" height="200"/>
    After re-check:
    <img src="./img/Ziteng Jiao/0926_ball01Air_01_00_After.png" alt="After 01_10 occlusion" height="200"/>

  * If we write the result before re-check, the wrong id might be used.
  
  <img src="./img/Ziteng Jiao/wrong_id.png" alt="wrong_id" height="200"/>  

    * In the example above, the person is recognized as No.447. In the frame 1800, groundtruths tells the model that No.447 is the same person as No.2. However, the results for the frames 1413, 1431, 1439 have been written out. 
    * Solution: when detection for the whole video is done, read the output file (detected catches) and substitute the wrong id with the correct one.

* Bug fixed:
  * When the groundtruths.txt doesn't match the video, the program might crash.
    * Solution: add a `--groundtruths` flag.
    * Usage: `--groundtruths [path to the file or disable]`

          python3 track.py --groundtruths ./inputs/2p1b_01.txt
          python3 track.py --groundtruths disable
      
    * The default value is `'./inputs/groundtruths.txt'`.

* Test the program on other videos:
  * Not going well.
  * CVAT is extremely slow on my computer: loading a video or download an annotations takes up to an hour.
  * Referee team is working on creating more annotations.