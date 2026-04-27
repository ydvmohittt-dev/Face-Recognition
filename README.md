# Face Recognition Based Attendance System

### Automatic Attendance Marking Using Real-Time Face Detection

The Face Recognition Based Attendance System is a project that uses face  of a individual or student  for recognition to identify students or employees and mark their attendance.
 It utilizes GPU acceleration through CUDA, making it fast and efficient. 



## Key Features

- Automatic Attendance Marking — The system detects faces through a camera and marks attendance on its own without any manual effort.

- Real-Time Face Recognition — Faces are identified instantly as soon as a person appears in front of the camera.

- GPU Accelerated Processing — NVIDIA CUDA is used to speed up face detection, making the system faster and more efficient.

- Secure Data Logging — Attendance records are automatically saved in CSV files, ensuring accurate and organized data storage

## Technologies Used

-Python 3.8+ — Main programming language used to build the entire system.

Dlib — Used for face detection and recognition using CNN-based deep learning model.

OpenCV — Used for capturing live camera feed and image processing.




## System Requirements

Python 3.11


## Installation Steps for Jetson Device

### Step 1: Setup Swap Memory

You need Swap face to run this project.
rum following commands:-
```bash
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
sudo bash -c 'echo "/swapfile none swap sw 0 0" >> /etc/fstab'
```

### Step 2: Install Required System Packages

 
You can run the following command:

```bash
sudo apt update && sudo apt upgrade -y
libx11-dev base-dev \
libgtk-3-dev libboost-python-dev libopenblas-dev liblapack-dev \
libjpeg-dev python3-dev python3-pip -y
```

### Step 3: Install cuDNN

TYou can run the following commands:

```bash
wget https://developer.download.nvidia.com/compute/cudnn/9.18.1/local_installers/cudnn-local-tegra-repo-ubuntu2204-9.18.1_1.0-1_arm64.deb
sudo dpkg -i cudnn-local-tegra-repo-ubuntu2204-9.18.1_1.0-1_arm64.
deb
sudo apt-get update
sudo apt-get -y install cudnn
```

### Step 4: Set CUDA Environment Variables

Need Cuda paths:
You can do this with the following commands:

```bash
export PATH=/usr//cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
export CUDACXX=/usr/local/cuda-12.6/bin/nvcc
source ~/.
bashrc
```




### Step 5: Clone Project and Setup Environment

Colne The project and set up the environment:
Run following commands:

```bash
git clone https://github.com/ydvmohittt-dev/Face-Recognition
cd FaceRecognition-Attendance
python3 -m venv ~/dlib_env
source ~/dlib_env/bin/activate
pip install --upgrade pip
pip install numpy==1.23.5 "opencv-contrib-python>=4.8"
```

### Step 6: Compile Dlib with CUDA Support

You can run the following commands:

```bash
git clone https://github.com/ydvmohittt-dev/Face-Recognition
cd dlib
mkdir build && cd build
cmake.
. -DUSE_AVX_INSTRUCTIONS=0
cmake --build.
--Config Release
cd.
.
python3 setup.py install
```


```bash
python3 -c "import dlib; print('CUDA Enabled:' + str(dlib.DLIB_USE_CUDA))"
```

### Step 7: Install Python Dependencies



```bash
cd ~/FaceRecognition-Attendance
pip install -r requirements.txt
```

## Project Folder Structure

The project is organized as follows:

```
FaceRecognition-Attendance/
│
├── Attendance_data/ → Stores registered face images
├── Attendance_Entry/ → Contains CSV attendance logs
├── main.py → Script for face recognition
├── cappture.py → to capture new candidate.
├── deletee.py → to delete already registered candidate.
├── requirements.txt → Python package requirements
└── README.md → Project documentation
```


Also i added commands like :
->summary.py---> it will check who are present and absent.
->Attendance_percentage---> it will give the monthwise attendance percentage of all the registerd candidates.

## How to Use

Follow these steps to use it:

**Step 1 – Activate the virtual environment:**

```bash
source ~/dlib_env/bin/activate
```

**Step 2 – Register a user:**

```bash
python3 capture.py
```

**Step 3 – Start the attendance system:**

```bash
python3 main.py
```

## Performance Optimization for Jetson

Run following commands for better performance:

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

```python
face_locations = face_recognition.
face_locations(rgb_small_frame, model="cnn")
```

## Common Issues and Fixes

Here are some common issues and their solutions:

| Problem | How to Fix |


bashrc` file ->
 Device freezes during setup  Swap memory was not configured.
Go back to Step 1 |
 Illegal instruction error -> Recompile Dlib with `USE_AVX_INSTRUCTIONS=0` 

## Project Members

The project team members are:
 Mohit Rao -> Developer
 Abhay Singh ->Testing and documentation.

## Institution Details

The institution details are:

- **Course:** B.Tech (ECE)
- **Subject:** Minor Project
- **Submitted To:** Mr. Ashish Parihar
- **College:** IIIT Bhopal
- **Academic Year:** 2025-26

## License

Available under MIT licence.
