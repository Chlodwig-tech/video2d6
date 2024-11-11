# video2d6-cuda
video to d6 video transformer based on image to d6 transformer: https://github.com/Chlodwig-tech/img2d6-cuda

To compile:
  - nvcc -o video2d6 main.cu -I/usr/local/include/opencv4 -L/usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_videoio -Xcompiler -fPIC -ccbin=g++ -std=c++17

To use:
  - ./video2d6 video.mp4 DETAIL
  - ./video2d6 video.mp4 SIMPLE

  
