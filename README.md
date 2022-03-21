# Fire-Detection-Using-YoloV4
# 3 Main Steps:
# Step 1: Create your own dataset with 2 files using LabelImg tool: 
- Images
- Text file that contains object's parameters: center X point, center Y point, width, height.
# Step 2: Retrain YoloV4 with your own dataset with some following files:
- yolo.names
- yolov4-tiny_custom.cfg
- Makefile
- YoloV4_on_custom_dataset.ipynb
# Step 3: Testing fire detection (images, videos, and camera) with some following files:
- FireDetection.py
- coco_fire.names
- yolov4-tiny_custom.cfg
- yolov4-tiny_custom_last.weights
# Results:
- ![output1](https://user-images.githubusercontent.com/52019849/115096000-67136a80-9f5e-11eb-8db0-406c54926b0a.PNG)
- ![output2](https://user-images.githubusercontent.com/52019849/115096003-68449780-9f5e-11eb-8eb6-7eceac055083.PNG)
- ![output3](https://user-images.githubusercontent.com/52019849/115096005-68dd2e00-9f5e-11eb-912d-324f5f8704d1.PNG)
