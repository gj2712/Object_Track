# Object_Track

## Getting started
pip install -r requirements.txt

## Downloading official pretrained weights
For Windows: You can download the yolov3 weights by clicking https://pjreddie.com/media/files/yolov3.weights.
Save it in weights folder with name yolo3.weights

## Saving your yolov3 weights as a TensorFlow model.
Run the script Transform.py.This will convert the yolov3 weights into TensorFlow .tf model files!
After running this
You will see these some files added to weights folder which will mean you have successfully completed this stage.
![Screenshot (123)](https://user-images.githubusercontent.com/63334651/111910592-02144400-8a88-11eb-9421-b89364db02e6.png)


## Running the Object Tracker
This is our main file which has the model 
Input format: 
python Object_tracker.py --video ./test_videos/video_3.mp4 --output ./test_videos/video_3_result.avi --line_coordinates 0,500,1920,500

i.e. from command line you need to give the input of test video file path, output video file path and the coordinates (these coordinates are used to create a line you
have to count the vehicles crossing this line)
