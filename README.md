# Fire-Detection-Using-YoloV4
# Some projects
# 3 Main Steps:
# Step 1: Create my own dataset with 2 files using LabelImg tool: 
- Images
- Text file that contains object's parameters: center X point, center Y point, width, height.
# Step 2: Retrain YoloV4 with my own dataset with some following files:
- yolo.names
- yolov4-tiny_custom.cfg
- Makefile
- YoloV4_on_custom_dataset.ipynb
# Step 3: Testing fire detection (images, videos, and camera) with some following files:
- FireDetection.py
- coco_fire.names
- yolov4-tiny_custom.cfg
- yolov4-tiny_custom_last.weights
