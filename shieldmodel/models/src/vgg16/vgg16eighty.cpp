// VGG16 - 80% in CUDA

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
#include <time.h>
#include <sys/time.h>

namespace F = torch::nn::functional;

struct part1 : public torch::nn::Module
{
    torch::nn::Conv2d C1;

    part1():
        C1(torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 3).padding(1)))
        {
            register_module("C1", C1);
        }

    torch::Tensor fwd_p1(torch::Tensor input)
    {
        auto x = F::relu(C1(input));
        return x;
    }
};

struct part2 : public torch::nn::Module
{
    torch::nn::Conv2d C3;

    part2():
        C3(torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).padding(1)))
        {
            register_module("C3", C3);
        }
    
    torch::Tensor fwd_p2(torch::Tensor input)
    {
        auto x = F::max_pool2d(F::relu(C3(input)), F::MaxPool2dFuncOptions(2));
        return x;
    }
};

struct part3 : public torch::nn::Module
{
    torch::nn::Conv2d C6;

    part3():
        C6(torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).padding(1)))
        {
            register_module("C6", C6);
        }
    
    torch::Tensor fwd_p3(torch::Tensor input)
    {
        auto x = F::relu(C6(input));
        return x;
    }
};

struct part4 : public torch::nn::Module
{
    torch::nn::Conv2d C8;

    part4():
        C8(torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3).padding(1)))
        {
            register_module("C8", C8);
        }

    torch::Tensor fwd_p4(torch::Tensor input)
    {
        auto x = F::max_pool2d(F::relu(C8(input)), F::MaxPool2dFuncOptions(2));
        return x;
    }
};

struct part5 : public torch::nn::Module
{
    torch::nn::Conv2d C11;

    part5():
        C11(torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3).padding(1)))
        {
            register_module("C11", C11);
        }
    
    torch::Tensor fwd_p5(torch::Tensor input)
    {
        auto x = F::relu(C11(input));
        return x;
    }
};

struct part6 : public torch::nn::Module
{
    torch::nn::Conv2d C13;
    torch::nn::Conv2d C15;
    torch::nn::Conv2d C18;
    torch::nn::Conv2d C20;
    torch::nn::Conv2d C22;
    torch::nn::Conv2d C25;
    torch::nn::Conv2d C27;
    torch::nn::Conv2d C29;
    torch::nn::Linear FC32;
    torch::nn::Linear FC35;
    torch::nn::Linear FC38;

    part6():
        C13(torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).padding(1))),
        C15(torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).padding(1))),
        C18(torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 512, 3).padding(1))),
        C20(torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).padding(1))),
        C22(torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).padding(1))),
        C25(torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).padding(1))),
        C27(torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).padding(1))),
        C29(torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).padding(1))),
        FC32(torch::nn::Linear(512 * 7 *  7, 4096)),
        FC35(torch::nn::Linear(4096, 4096)),
        FC38(torch::nn::Linear(4096, 1000))
        {
            register_module("C13", C13);
            register_module("C15", C15);
            register_module("C18", C18);
            register_module("C20", C20);
            register_module("C22", C22);
            register_module("C25", C25);
            register_module("C27", C27);
            register_module("C29", C29);
            register_module("FC32", FC32);
            register_module("FC35", FC35);
            register_module("FC38", FC38);
        }

    torch::Tensor fwd_p6(torch::Tensor input)
    {
        auto x = F::max_pool2d(F::relu(C15(F::relu(C13(input)))), F::MaxPool2dFuncOptions(2));
        x = F::max_pool2d(F::relu(C22(F::relu(C20(F::relu(C18(x)))))), F::MaxPool2dFuncOptions(2));
        x = F::max_pool2d(F::relu(C29(F::relu(C27(F::relu(C25(x)))))), F::MaxPool2dFuncOptions(2));
        x = x.view({ -1, num_flat_features(x) });
        x = F::relu(FC32(x));
        x = F::relu(FC35(x));
        x = FC38(x);
        return x;
    }

    long num_flat_features(torch::Tensor x)
    {
        // To except the batch dimension for vgg16_bn:
        // auto size = x.size()[1:]
        // For vgg16:
        auto size = x.sizes();
        auto num_features = 1;
        for (auto s : size)
        {
            num_features *= s;
        }
        return num_features;
    }
};

struct vgg16_eighty_split
{
    part1 p1;
    part2 p2;
    part3 p3;
    part4 p4;
    part5 p5;
    part6 p6;

    torch::Tensor forward(torch::Tensor input)
    {
        input = input.to(at::kCPU);
        p1.to(at::kCPU);
        auto x = p1.fwd_p1(input);

        x = x.to(at::kCUDA);
        p2.to(at::kCUDA);
        x = p2.fwd_p2(x);

        x = x.to(at::kCPU);
        p3.to(at::kCPU);
        x = p3.fwd_p3(x);

        x = x.to(at::kCUDA);
        p4.to(at::kCUDA);
        x = p4.fwd_p4(x);

        x = x.to(at::kCPU);
        p5.to(at::kCPU);
        x = p5.fwd_p5(x);

        x = x.to(at::kCUDA);
        p6.to(at::kCUDA);
        x = p6.fwd_p6(x);

        return x;
    }
};

int main()
{
    std::cout << "vgg16 - 80% outsourced to CUDA version." << std::endl;
    vgg16_eighty_split model;
    
    int count1 = 1000;
    int count_warmup = 3000;

    torch::Tensor input = torch::rand({1, 3, 224, 224});
    torch::Tensor output;

    for (size_t i = 0; i < count_warmup; i++)
    {
        output = model.forward(input);
    }

    cudaEvent_t start, stop;
    float esp_time_gpu;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float total = 0;

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