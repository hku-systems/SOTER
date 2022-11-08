// MLP

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <torch/torch.h>
#include <iostream>
#include <random>
#include <time.h>
#include <sys/time.h>

struct p1 : public torch::nn::Module
{
    torch::nn::Linear fc0;
    p1():
        fc0(784, 512)
        {
            register_module("fc0", fc0);
        }
        torch::Tensor forward(torch::Tensor x)
        {
            x = x.view({ -1, 28 * 28 });
            x = torch::relu(fc0->forward(x));
            return x;
        }
};

struct p2 : public torch::nn::Module
{
    torch::nn::Linear fc3;
    p2():
        fc3(512, 128)
        {
            register_module("fc3", fc3);
        }
        torch::Tensor forward(torch::Tensor x)
        {
            x = torch::relu(fc3->forward(x));
            return x;
        }
};

struct p3 : public torch::nn::Module
{
    torch::nn::Linear fc6;
    p3():
        fc6(128, 10)
        {
            register_module("fc6", fc6);
        }
        torch::Tensor forward(torch::Tensor x)
        {
            x = fc6->forward(x);
            return x;
        }
};

struct mlp : public torch::nn::Module
{
    p1 cpu1;
    p2 cuda2;
    p3 cpu3;
    
    void ini()
    {
        cpu1.to(at::kCPU);
        cuda2.to(at::kCUDA);
        cpu3.to(at::kCPU);
        return ;
    }
    torch::Tensor forward(torch::Tensor x)
    {
        x = x.to(at::kCPU);
        x = cpu1.forward(x);
        x = x.to(at::kCUDA);
        x = cuda2.forward(x);
        x = x.to(at::kCPU);
        x = cpu3.forward(x);        
        return x;
    }
};

struct mlp2 : public torch::nn::Module
{
    p1 cpu1;
    p2 cuda2;
    p3 cpu3;
    
    void ini()
    {
        cpu1.to(at::kCPU);
        cuda2.to(at::kCUDA);
        cpu3.to(at::kCPU);
        return ;
    }
    torch::Tensor forward(torch::Tensor x)
    {
        x = x.to(at::kCPU);
        x = cpu1.forward(x);
        std::cout << x.sizes() << std::endl;
        x = x.to(at::kCUDA);
        x = cuda2.forward(x);
        std::cout << x.sizes() << std::endl;
        x = x.to(at::kCPU);
        x = cpu3.forward(x);
        return x;
    }
};

int main()
{
    std::cout << "MLP 80% outsourced to CUDA version" << std::endl;
    mlp model;
    
    model.ini();

    torch::Tensor input = torch::rand({1, 1, 28, 28});
    torch::Tensor output;

    int count = 1000;
    int warmup = 9000;
    // int count2 = 10000;

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
    for (size_t i = 0; i < count1; i++)
    {
        cudaEventRecord(start, 0);
        output = model.forward(input);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&esp_time_gpu, start, stop);
        total += esp_time_gpu;
    }
    output = model2.forward(input);

    float latency;
    latency = total / ((float)count);
    std::cout << "For " << count << " inferences..." << std::endl;
    std::cout << "Time elapsed: " << total << " ms." << std::endl;
    std::cout << "Time consuming: " << latency << " ms per instance." << std::endl;

    return 0;
}