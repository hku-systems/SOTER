// AlexNet - 80% outsource

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <torch/library.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <iostream>
#include <random>
#include <string>
#include <io.h>
#include <memory>
#include <vector>
#include <random>
#include <time.h>
#include <sys/time.h>

namespace F = torch::nn::functional;

struct p1 : public torch::nn::Module
{
    torch::nn::Conv2d C1;
    p1():
        C1(torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 11).padding(2).stride(4)))
        {
            register_module("C1", C1);
        }
    torch::Tensor forward(torch::Tensor x)
    {
        x = C1(x);
        return x;
    }
};

struct p2 : public torch::nn::Module
{
    torch::nn::Conv2d C3;
    p2():
        C3(torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 192, 5).padding(2)))
        {
            register_module("C3", C3);
        }
    torch::Tensor forward(torch::Tensor x)
    {
        x = F::max_pool2d(F::relu(x), F::MaxPool2dFuncOptions(3).stride(2));
        x = F::relu(C3(x));
        return x;
    }
};

struct p3 : public torch::nn::Module
{
    torch::Tensor forward(torch::Tensor x)
    {
        x = F::max_pool2d(x, F::MaxPool2dFuncOptions(3).stride(2));
        return x;
    }
};

struct p4 : public torch::nn::Module
{
    torch::nn::Conv2d C6;
    torch::nn::Conv2d C8;
    torch::nn::Conv2d C10;
    p4():
        C6(torch::nn::Conv2d(torch::nn::Conv2dOptions(192, 384, 3).padding(1))),
        C8(torch::nn::Conv2d(torch::nn::Conv2dOptions(384, 256, 3).padding(1))),
        C10(torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).padding(1)))
        {
            register_module("C6", C6);
            register_module("C8", C8);
            register_module("C10", C10);
        }    
    torch::Tensor forward(torch::Tensor x)
    {
        x = C10(F::relu(C8(F::relu(C6(x)))));
        return x;
    }
};

struct p5 : public torch::nn::Module
{
    torch::nn::Linear FC1;
    p5():
        FC1(torch::nn::Linear(9216, 4096))
        {
            register_module("FC1", FC1);
        }
    torch::Tensor forward(torch::Tensor x)
    {
        x = F::max_pool2d(F::relu(x), F::MaxPool2dFuncOptions(3).stride(2));
        x = x.view({ -1, num_flat_features(x) });
        x = FC1(x);
        return x;
    }
    long num_flat_features(torch::Tensor x)
    {
        auto size = x.sizes();
        auto num_features = 1;
        for (auto s : size)
        {
            num_features *= s;
        }
        return num_features;
    }
};

struct p6 : public torch::nn::Module
{
    torch::Tensor forward(torch::Tensor x)
    {
        x = F::relu(x);
        return x;
    }
};

struct p7 : public torch::nn::Module
{
    torch::nn::Linear FC2;
    torch::nn::Linear FC3;
    p7():
        FC2(torch::nn::Linear(4096, 4096)),
        FC3(torch::nn::Linear(4096, 1000))
        {
            register_module("FC2", FC2);
            register_module("FC3", FC3);
        }
    torch::Tensor forward(torch::Tensor x)
    {
        x = FC3(F::relu(FC2(x)));
        return x;
    }
};

struct alexnet
{
    p1 cpu1;
    p2 cuda2;
    p3 cpu3;
    p4 cuda4;
    p5 cpu5;
    p6 cuda6;
    p7 cpu7;

    void ini()
    {
        cpu1.to(at::kCPU);
        cuda2.to(at::kCUDA);
        cpu3.to(at::kCPU);
        cuda4.to(at::kCUDA);
        cpu5.to(at::kCPU);
        cuda6.to(at::kCUDA);
        cpu7.to(at::kCPU);
    }

    torch::Tensor forward(torch::Tensor x)
    {
        x = x.to(at::kCPU);
        x = cpu1.forward(x);
        
        //std::cout << x.sizes() << std::endl;

        x = x.to(at::kCUDA);
        x = cuda2.forward(x);

        //std::cout << x.sizes() << std::endl;

        x = x.to(at::kCPU);
        x = cpu3.forward(x);

        //std::cout << x.sizes() << std::endl;

        x = x.to(at::kCUDA);
        x = cuda4.forward(x);

        //std::cout << x.sizes() << std::endl;

        x = x.to(at::kCPU);
        x = cpu5.forward(x);

        //std::cout << x.sizes() << std::endl;

        x = x.to(at::kCUDA);
        x = cuda6.forward(x);

        //std::cout << x.sizes() << std::endl;

        x = x.to(at::kCPU);
        x = cpu7.forward(x);

        return x;
    }
};

struct alexnet2
{
    p1 cpu1;
    p2 cuda2;
    p3 cpu3;
    p4 cuda4;
    p5 cpu5;
    p6 cuda6;
    p7 cpu7;

    void ini()
    {
        cpu1.to(at::kCPU);
        cuda2.to(at::kCUDA);
        cpu3.to(at::kCPU);
        cuda4.to(at::kCUDA);
        cpu5.to(at::kCPU);
        cuda6.to(at::kCUDA);
        cpu7.to(at::kCPU);
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

        std::cout << x.sizes() << std::endl;

        x = x.to(at::kCUDA);
        x = cuda4.forward(x);

        std::cout << x.sizes() << std::endl;

        x = x.to(at::kCPU);
        x = cpu5.forward(x);

        std::cout << x.sizes() << std::endl;

        x = x.to(at::kCUDA);
        x = cuda6.forward(x);

        std::cout << x.sizes() << std::endl;

        x = x.to(at::kCPU);
        x = cpu7.forward(x);

        return x;
    }
};

int main()
{
    std::cout << "AlexNet - 50 % outsourced to CUDA" << std::endl;
    std::cout << "{1, 3, 224, 224}" << std::endl;

    alexnet model;
    alexnet2 model2;
    model.ini();
    model2.ini();

    torch::Tensor output;

    //torch::Tensor output;
    torch::Tensor input = torch::ones({1, 3, 224, 224});
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
    output = model2.forward(input);

    float latency;
    latency = total / ((float)count);
    std::cout << "For " << count << " inferences..." << std::endl;
    std::cout << "Time elapsed: " << total << " ms." << std::endl;
    std::cout << "Time consuming: " << latency << " ms per instance." << std::endl;
    
    return 0;
}