#include <iostream>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#define DICESIZE 10

#define CUDA_CALL(x, message) {if((x) != cudaSuccess) { \
    printf("Error - %s(%d)[%s]: %s\n", __FILE__, __LINE__, message, cudaGetErrorString(x)); \
    exit(EXIT_FAILURE); }}

#define DICE(dimg, idx, cond) \
    (dimg[idx] = dimg[idx + 1] = dimg[idx + 2] = (cond) ? 0 : 255)

__global__ void simple_img2d6Kernel(unsigned char *dimg, int width, int height, int channels){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ unsigned int avg[DICESIZE - 1][DICESIZE - 1];

    if(row >= height || col >= width)
        return;

    int idx = (row * width + col) * channels;
    if(threadIdx.x == DICESIZE - 1 || threadIdx.y == DICESIZE - 1){
        dimg[idx] = dimg[idx + 1] = dimg[idx + 2] = 0;
        return;
    }

    unsigned char grey = (unsigned char)(0.299f * dimg[idx] + 0.587f * dimg[idx + 1] + 0.114f * dimg[idx + 2]);
    int x = threadIdx.x, y = threadIdx.y;
    avg[x][y] = grey;
    __syncthreads();
    if(x == 0){
        for(int i = 1; i < DICESIZE - 1; i++){
            avg[0][y] += avg[i][y];
        }   
    }
    __syncthreads();
    if(x == 0 && y == 0){
        for(int i = 1; i < DICESIZE - 1; i++){
            avg[0][0] += avg[0][i];
        }
        int d = avg[0][0] / ((DICESIZE - 1) * (DICESIZE - 1)) *6 / 255;
        avg[0][0] = d > 5 ? 6 : d + 1;;
    }
    __syncthreads();

    int d = 7 - avg[0][0];
    switch(d){
        case 1:
            DICE(dimg, idx, x >= 3 && x <= 5 && y >= 3 & y <= 5);
            break;
        case 2:
            DICE(dimg, idx, (x >= 6 && y < 3) || (x < 3 && y >= 6));
            break;
        case 3:
            DICE(dimg, idx, (x >= 3 && x <= 5 && y >= 3 & y <= 5) || (x >= 6 && y < 3) || (x < 3 && y >= 6));
            break;
        case 4:
            DICE(dimg, idx, (x >= 6 && y < 3) || (x < 3 && y >= 6) || (x < 3 && y < 3) || (x >= 6 && y >= 6));
            break;
        case 5:
            DICE(dimg, idx, (x >= 6 && y < 3) || (x < 3 && y >= 6) || (x < 3 && y < 3) || (x >= 6 && y >= 6) || (x >=3 && x <= 5 && y >= 3 & y <= 5));
            break;
        case 6:
            DICE(dimg, idx, y < 3 || y >= 6);
            break;
    }
}
__global__ void detailed_img2d6Kernel(unsigned char *dimg, int width, int height, int channels){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ unsigned int avg[DICESIZE * 3][DICESIZE * 3];

    if(row >= height || col >= width)
        return;

    int idx = (row * width + col) * channels;
    unsigned char grey = (unsigned char)(0.299f * dimg[idx] + 0.587f * dimg[idx + 1] + 0.114f * dimg[idx + 2]);
    int x = threadIdx.x, y = threadIdx.y;
    avg[x][y] = grey;
    __syncthreads();
    if(x % 3 == 0 && y % 3 == 0){
        for(int i = 0; i < 3; i++){
            for(int j = 0; j < 3; j++){
                avg[x][y] += avg[x + i][y + i];
            }
        }
        int d = avg[x][y] / 9 * 6 / 255;
        avg[x][y] = d > 5 ? 6 : d + 1;
    }
    __syncthreads();

    int d = 7 - avg[x / 3 * 3][y / 3 * 3];
    switch(d){
        case 1:
            DICE(dimg, idx, x % 3 == 1 && y % 3 == 1);
            break;
        case 2:
            DICE(dimg, idx, (x % 3 == 0 && y % 3 == 2) || (x % 3 == 2 && y % 3 == 0));
            break;
        case 3:
            DICE(dimg, idx, (x % 3 == 0 && y % 3 == 2) || (x % 3 == 2 && y % 3 == 0) || (x % 3 == 1 && y % 3 == 1));
            break;
        case 4:
            DICE(dimg, idx, (x % 3 == 0 && y % 3 == 2) || (x % 3 == 2 && y % 3 == 0) || (x % 3 == 0 && y % 3 == 0) || (x % 3 == 2 && y % 3 == 2));
            break;
        case 5:
            DICE(dimg, idx, (x % 3 == 0 && y % 3 == 2) || (x % 3 == 2 && y % 3 == 0) || (x % 3 == 0 && y % 3 == 0) || (x % 3 == 2 && y % 3 == 2) || (x % 3 == 1 && y % 3 == 1));
            break;
        case 6:
            DICE(dimg, idx, y % 3 == 0 || y % 3 == 2);
            break;
    }
}

void processSimpleFrame(unsigned char* himg, int width, int height, int channels, cudaStream_t stream) {
    unsigned char *dimg;
    int size = width * height * channels * sizeof(unsigned char);
    CUDA_CALL(cudaMalloc((void **)&dimg, size), "cudaMalloc - dimg");
    CUDA_CALL(cudaMemcpyAsync(dimg, himg, size, cudaMemcpyHostToDevice, stream), "cudaMemcpy - himg -> dimg");

    int nn = DICESIZE;
    dim3 block_size(nn, nn);
    dim3 grid_size(
        (height - 1) / block_size.x + 1,
        (width - 1) / block_size.y + 1
    );
    simple_img2d6Kernel<<<grid_size, block_size, 0, stream>>>(dimg, width, height, channels);
    CUDA_CALL(cudaMemcpyAsync(himg, dimg, size, cudaMemcpyDeviceToHost, stream), "cudaMemcpy - dimg -> himg");
    CUDA_CALL(cudaFree(dimg), "cudaFree - dimg")
}

void processDetailFrame(unsigned char* himg, int width, int height, int channels, cudaStream_t stream) {
    unsigned char *dimg;
    int size = width * height * channels * sizeof(unsigned char);
    CUDA_CALL(cudaMalloc((void **)&dimg, size), "cudaMalloc - dimg");
    CUDA_CALL(cudaMemcpyAsync(dimg, himg, size, cudaMemcpyHostToDevice, stream), "cudaMemcpy - himg -> dimg");

    int nn = DICESIZE * 3;
    dim3 block_size(nn, nn);
    dim3 grid_size(
        (height - 1) / block_size.x + 1,
        (width - 1) / block_size.y + 1
    );
    detailed_img2d6Kernel<<<grid_size, block_size, 0, stream>>>(dimg, width, height, channels);
    CUDA_CALL(cudaMemcpyAsync(himg, dimg, size, cudaMemcpyDeviceToHost, stream), "cudaMemcpy - dimg -> himg");
    CUDA_CALL(cudaFree(dimg), "cudaFree - dimg")
}

int main(int argc, char** argv) {
    if (argc != 3) {
        printf("Usage: %s <video_path> <option (DETAIL|SIMPLE)>\n", argv[0]);
        return -1;
    }

    const char* input_video = argv[1];
    const char *option = argv[2];

    const char* output_video = "output.avi";

    cv::VideoCapture cap(input_video);
    if (!cap.isOpened()) {
        printf("Error opening video file!\n");
        return -1;
    }

    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int fps = static_cast<int>(cap.get(cv::CAP_PROP_FPS));
    int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');

    cv::VideoWriter writer(output_video, fourcc, fps, cv::Size(width, height));

    if (!writer.isOpened()) {
        printf("Error opening output video file!\n");
        return -1;
    }

    cv::Mat frame;
    bool b = true;
    cudaStream_t streams[2];
    for (int i = 0; i < 2; i++) {
        CUDA_CALL(cudaStreamCreate(&streams[i]), "cudaStreamCreate");
    }

    if (strcmp(option, "DETAIL") == 0) {
        while (cap.read(frame)) {
            unsigned char* frame_data = frame.data;
            processDetailFrame(frame_data, width, height, frame.channels(), streams[b]);
            writer.write(frame);
            b = !b;
        }
    } else if (strcmp(option, "SIMPLE") == 0) {
        while (cap.read(frame)) {
            unsigned char* frame_data = frame.data;
            processSimpleFrame(frame_data, width, height, frame.channels(), streams[b]);
            writer.write(frame);
            b = !b;
        }
    } else {
        printf("Error: Invalid option '%s'. Please use 'DETAIL' or 'SIMPLE'.\n", option);
        return -1;
    }    


    

    for (int i = 0; i < 2; i++) {
        CUDA_CALL(cudaStreamDestroy(streams[i]), "cudaStreamDestroy");
    }

    cap.release();
    writer.release();
    printf("Processing complete. Output saved as %s\n", output_video);
    return 0;
}
