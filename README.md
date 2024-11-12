# video2d6-cuda
This project implements an program written in CUDA that transforms the input video into a grid of D6 dices, where each dice face represents a block of the original video frame. The program uses the CUDA parallel computing framework to process the image efficiently.
It is based on https://github.com/Chlodwig-tech/img2d6-cuda

To compile:
  - nvcc -o video2d6 main.cu -I/usr/local/include/opencv4 -L/usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_videoio -Xcompiler -fPIC -ccbin=g++ -std=c++17

To use:
  - ./video2d6 video.mp4 DETAIL
  - ./video2d6 video.mp4 SIMPLE

  
