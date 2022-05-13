// vgg16 - CPU version

#include <torch/library.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <iostream>
#include <string>
#include <io.h>
#include <memory>
#include <vector>
#include <time.h>
#include <sys/time.h>

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

struct vgg16 : public torch::nn::Module
{
        torch::nn::Conv2d C1;
        torch::nn::Conv2d C3;
        torch::nn::Conv2d C6;
        torch::nn::Conv2d C8;
        torch::nn::Conv2d C11;
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

        vgg16():
                C1(torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 3).padding(1))),
                C3(torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).padding(1))),
                C6(torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).padding(1))),
                C8(torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3).padding(1))),
                C11(torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3).padding(1))),
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
                        register_module("C1", C1);
                        register_module("C3", C3);
                        register_module("C6", C6);
                        register_module("C8", C8);
                        register_module("C11", C11);
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
        
        torch::Tensor forward(torch::Tensor input)
        {
            namespace func = torch::nn::functional;
            // block 1
            auto x = func::max_pool2d(func::relu(C3(func::relu(C1(input)))), func::MaxPool2dFuncOptions(2));
        
            // block 2
            x = func::max_pool2d(func::relu(C8(func::relu(C6(x)))), func::MaxPool2dFuncOptions(2));
        
            // block 3
            x = func::max_pool2d(func::relu(C15(func::relu(C13(func::relu(C11(x)))))), func::MaxPool2dFuncOptions(2));
        
            // block 4
            x = func::max_pool2d(func::relu(C22(func::relu(C20(func::relu(C18(x)))))), func::MaxPool2dFuncOptions(2));
        
            // block 5
            x = func::max_pool2d(func::relu(C29(func::relu(C27(func::relu(C25(x)))))), func::MaxPool2dFuncOptions(2));
            x = x.view({ -1, num_flat_features(x) });
            
            // classifier
            x = func::relu(FC32(x));
            x = func::relu(FC35(x));
            x = FC38(x);
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

int main(){
    vgg16 model;
    int count1 = 1000;
    std::cout << "vgg16 CPU version" << std::endl;

    torch::Tensor input = torch::ones({1, 3, 224, 224});// .to(at::kCUDA);
    torch::Tensor output;

    double start, stop, duration;
    duration = 0;
    for (size_t i = 0; i < count1; i++)
    {
        start = time_get();
        output = model.forward(input);
        stop = time_get();
        duration += stop - start;
    }
    double latency = duration / ((double)count1);

    std::cout << "For 1,000 inferences..." << std::endl;
    std::cout << "Time elapsed: " << duration << " ms." << std::endl;
    std::cout << "Time consuming: " << latency << " ms per instance." << std::endl;

    return 0;
}