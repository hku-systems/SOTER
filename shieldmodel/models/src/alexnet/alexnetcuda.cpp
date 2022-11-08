// AlexNet - CPU

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <torch/library.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <iostream>
#include <random>
#include <string>
#include <memory>
#include <vector>
#include <random>
#include <time.h>
#include <sys/time.h>

namespace F = torch::nn::functional;
 
double time_get()
{
    time_t time_sec;
    double nowtime=0;
    time_sec = time(NULL);
    struct timeval tmv;
    gettimeofday(&tmv, NULL);
    nowtime=time_sec*1000+tmv.tv_usec/1000;

    return nowtime;
}

struct alexnet : public torch::nn::Module
{
    torch::nn::Conv2d C1;
    torch::nn::Conv2d C3;
    torch::nn::Conv2d C6;
    torch::nn::Conv2d C8;
    torch::nn::Conv2d C10;
    torch::nn::Linear FC1;
    torch::nn::Linear FC2;
    torch::nn::Linear FC3;

    alexnet():
        C1(torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 11).padding(2).stride(4))),
        C3(torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 192, 5).padding(2))),
        C6(torch::nn::Conv2d(torch::nn::Conv2dOptions(192, 384, 3).padding(1))),
        C8(torch::nn::Conv2d(torch::nn::Conv2dOptions(384, 256, 3).padding(1))),
        C10(torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).padding(1))),
        FC1(torch::nn::Linear(9216, 4096)),
        FC2(torch::nn::Linear(4096, 4096)),
        FC3(torch::nn::Linear(4096, 1000))
        {
            register_module("C1", C1);
            register_module("C3", C3);
            register_module("C6", C6);
            register_module("C8", C8);
            register_module("C10", C10);
            register_module("FC1", FC1);
            register_module("FC2", FC2);
            register_module("FC3", FC3);
        }
    
    torch::Tensor forward(torch::Tensor input)
    {
        auto x = F::max_pool2d(F::relu(C1(input)), F::MaxPool2dFuncOptions(3).stride(2));
        x = F::max_pool2d(F::relu(C3(x)), F::MaxPool2dFuncOptions(3).stride(2));
        x = F::max_pool2d(F::relu(C10(F::relu(C8(F::relu(C6(x)))))), F::MaxPool2dFuncOptions(3).stride(2));
        x = x.view({ -1, num_flat_features(x) });
        x = F::relu(FC1(x));
        x = FC3(F::relu(FC2(x)));

        return x;
    }

    long num_flat_features(torch::Tensor x)
    {
        // To except the batch dimension:
        // auto size = x.size()[1:]
        // For AlexNet:
        auto size = x.sizes();
        auto num_features = 1;
        for (auto s : size)
        {
            num_features *= s;
        }
        return num_features;
    }
};

int main()
{
    std::cout << "AlexNet - CUDA version" << std::endl;
    std::cout << "{1, 3, 224, 224}" << std::endl;
    alexnet model;

    torch::Tensor output;

    model.to(at::kCUDA);
    
    //torch::Tensor output;
    torch::Tensor input = torch::ones({1, 3, 224, 224}).to(at::kCUDA);
    int count = 1000;
    int warmup = 3000;

    // warm up
    for (size_t i = 0; i < warmup; i++)
    {
        output = model.forward(input);
    }
    
    cudaEvent_t start, stop;
    float esp_time_gpu;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float total = 0;

    // start measurement
    for (size_t i = 0; i < count; i++)
    {
        cudaEventRecord(start, 0);
        output = model.forward(input);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&esp_time_gpu, start, stop);
        total += esp_time_gpu;
    }

    float latency;
    latency = total / ((float)count);
    std::cout << "For " << count << " inferences..." << std::endl;
    std::cout << "Time elapsed: " << total << " ms." << std::endl;
    std::cout << "Time consuming: " << latency << " ms per instance." << std::endl;

    return 0;
}