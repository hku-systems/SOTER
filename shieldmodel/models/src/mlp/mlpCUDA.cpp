// MLP

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <torch/torch.h>
#include <iostream>
#include <random>
#include <time.h>
#include <sys/time.h>

struct mlp : public torch::nn::Module
{
    torch::nn::Linear fc0;
    torch::nn::Linear fc3;
    torch::nn::Linear fc6;

    mlp():
        fc0(784, 512),
        fc3(512, 128),
        fc6(128, 10)
        {
            register_module("fc0", fc0);
            register_module("fc3", fc3);
            register_module("fc6", fc6);
        }
    
    torch::Tensor forward(torch::Tensor x)
    {
        x = x.view({ -1, 28 * 28 });
        x = torch::relu(fc0->forward(x));
        x = torch::relu(fc3->forward(x));
        x = fc6->forward(x);
        
        return x;
    }

};

int main()
{
    std::cout << "MLP {1, 1, 28, 28} CPU version" << std::endl;
    mlp model;
    model.to(at::kCUDA);

    torch::Tensor input = torch::rand({1, 1, 28, 28}).to(at::kCUDA);
    torch::Tensor output;

    int count1 = 1000;
    int count_warmup = 3000;
    // int count2 = 10000;

    // warm up
    for (size_t i = 0; i < count_warmup; i++)
    {
        output = model.forward(input);
    }
    
    cudaEvent_t start, stop;
    float esp_time_gpu;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float total = 0;

    // start measurement
    for (size_t i = 0; i < count1; i++)
    {
        cudaEventRecord(start, 0);
        output = model.forward(input);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&esp_time_gpu, start, stop);
        total += esp_time_gpu;
    }

    float latency;
    latency = total / ((float)count1);
    std::cout << "For " << count1 << " inferences..." << std::endl;
    std::cout << "Time elapsed: " << total << " ms." << std::endl;
    std::cout << "Time consuming: " << latency << " ms per instance." << std::endl;

    return 0;
}