CAFFE = /home/gs/dl/m/caffe
CC := g++

CFLAGS_CAFFE := -I$(CAFFE)/build/install/include -I /usr/include/hdf5/serial/ -DCPU_ONLY 
CFLAGS_OPENCV := 

LDFLAGS_CAFFE := -lcaffe 
LDFLAGS_OPENCV := -lopencv_core  -lopencv_highgui -lopencv_imgproc  -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect   -lopencv_legacy 

CFLAGS := -Wall -g $(CFLAGS_CAFFE) $(CFLAGS_OPENCV)
LDFLAGS := $(LDFLAGS_CAFFE) $(LDFLAGS_OPENCV)



all: driveassist

driveassist: driveassist.cpp driveassist.hpp
	$(CC)  $(CFLAGS) $(LDFLAGS) -o "driveassist" "driveassist.cpp"

