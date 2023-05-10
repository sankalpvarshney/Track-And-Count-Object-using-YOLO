# Tracking and counting of object using YOLO v8

This repository contains Python code for tracking vehicles (such as cars, buses, and bikes) as they enter and exit the road, thereby incrementing the counters for incoming and outgoing vehicles.

## Installation

```bash
1. git clone https://github.com/sankalpvarshney/Track-And-Count-Object-using-YOLO.git
2. cd Track-And-Count-Object-using-YOLO
3. conda create --prefix ./env python=3.8 -y
4. conda activate ./env
5. pip install ultralytics
6. git clone https://github.com/ifzhang/ByteTrack.git
7. cd ByteTrack
8. sed -i 's/onnx==1.8.1/onnx==1.9.0/g' requirements.txt
9. pip install -q -r requirements.txt
10. python setup.py -q develop
11. pip install -q cython_bbox
12. pip install -q onemetric
13. pip install -q loguru lap
14. pip install numpy==1.22.4
15. pip install supervision==0.1.0
```

## Usage

Firstly set the crossing line co-ordinates inside the code i.e yolov8tracker.py for the incoming and outgoing vehicles. And then execute the python code as mentioned below.
### Linux

```bash
python yolov8tracker.py -i <input_video_path> -o <output_video_path>
```

### Python

```python
from yolov8tracker import TrackObject
obj = TrackObject(<input_video_path>,<output_video_path>)
obj.process_video()
```

https://github.com/sankalpvarshney/Track-And-Count-Object-using-YOLO/assets/41926323/bbeb35b4-3f0f-49cd-b222-2bf92ac001f7



## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

