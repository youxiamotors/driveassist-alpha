all : driveassist

driveassist: driveassist.cpp driveassist.hpp
	g++  -Wall -g  -lopencv_core  -lopencv_highgui -lopencv_imgproc  -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect   -lopencv_legacy -fopenmp -o "driveassist" "driveassist.cpp"


clean:
	rm -rv driveassist
