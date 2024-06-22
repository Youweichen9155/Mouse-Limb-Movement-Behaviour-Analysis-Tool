# Mouse-Limb-Movement-Behaviour-Analysis-Tool

OpenCV-based colour segmentation for datamining mouse limb movement behaviour.

Author: Youwei Chen 

Key Laboratory of Spine and Spinal Cord Injury Repair and Regeneration of Ministry of Education, Orthopaedic Department of Tongji Hospital, Tongji University

## Requirements

- Python 3.6+
- OpenCV
- NumPy
- Matplotlib
- Pandas

## Installation

1. Clone the repository:
git clone https://github.com/Youweichen9155/Mouse-Limb-Movement-Behaviour-Analysis-Tool

2. Install the required packages


## Preparing Mice for Movement Video

1. The 4 joints to be tested were marked on the mice with different colours.

2. Before shooting the video, use a ruler to record the length and width of the corresponding cm in the video frame.

3. Keep the frame level with the plane on which the mouse is standing.

4. Shoot 240 fps slow motion video. Note that the video exported from the mobile phone is at 30 fps, if the mice are moving on a running belt, the difference in frame rate has been converted in the programme, just enter the speed of the running belt.

## Usage

1. cd Mouse-Limb-Movement-Behaviour-Analysis-Tool

1. python analyze_video.py

2. Input the video path, the actual length and width of the video frame, and the frame rate interval.

3. After loading the video click on the 5 joints in the pop-up window to read the colours of the markers.

4. The final result will be a video of the original video superimposed on the joint connecting lines, a video containing only the joint connecting lines, a table of the coordinates of each joint, and a graph of the joint's motion trajectory.
