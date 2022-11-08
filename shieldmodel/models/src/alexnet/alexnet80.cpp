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
        x = F::relu(C1(x));
        return x;
    }
};

struct p2 : public torch::nn::Module
{
    torch::nn::Conv2d C3;
    torch::nn::Conv2d C6;
    p2():
        C3(torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 192, 5).padding(2))),
        C6(torch::nn::Conv2d(torch::nn::Conv2dOptions(192, 384, 3).padding(1)))
        {
            register_module("C3", C3);
            register_module("C6", C6);
        }
    torch::Tensor forward(torch::Tensor x)
    {
        x = F::max_pool2d(x, F::MaxPool2dFuncOptions(3).stride(2));
        x = F::max_pool2d(F::relu(C3(x)), F::MaxPool2dFuncOptions(3).stride(2));
        x = C6(x);
        return x;
    }
};


struct p3 : public torch::nn::Module
{
    torch::Tensor forward(torch::Tensor x)
    {
        x = F::relu(x);
        return x;
    }
};

struct p4 : public torch::nn::Module
{
    torch::nn::Conv2d C8;
    torch::nn::Conv2d C10;
    p4():
        C8(torch::nn::Conv2d(torch::nn::Conv2dOptions(384, 256, 3).padding(1))),
        C10(torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).padding(1)))
        {
            register_module("C8", C8);
            register_module("C10", C10);
        }    
    torch::Tensor forward(torch::Tensor x)
    {
        x = F::max_pool2d(F::relu(C10(F::relu(C8(x)))), F::MaxPool2dFuncOptions(3).stride(2));
        return x; 
    }
};

struct p5 : public torch::nn::Module
{
    torch::Tensor forward(torch::Tensor x)
    {
        x = x.view({ -1, num_flat_features(x) });
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
    torch::nn::Linear FC1;
    torch::nn::Linear FC2;
    torch::nn::Linear FC3;
    p6():
        FC1(torch::nn::Linear(9216, 4096)),
        FC2(torch::nn::Linear(4096, 4096)),
        FC3(torch::nn::Linear(4096, 1000))
        {
            register_module("FC1", FC1);
            register_module("FC2", FC2);
            register_module("FC3", FC3);
        }
    torch::Tensor forward(torch::Tensor x)
    {
        x = F::relu(FC1(x));
        x = FC3(F::relu(FC2(x)));
        return x; 
    }
};

class alexnet_part : public torch::nn::Module {
    public:
    virtual torch::Tensor forward(torch::Tensor x) = 0;
    virtual void morphpara() = 0;
};

struct op_relu : public torch::nn::Module
{
    torch::nn::ReLU op_relu;
    torch::Tensor forward(torch::Tensor x)
    {
        x = op_relu(x);
        return x;   
    }
};

struct op_maxp : public torch::nn::Module
{
    torch::nn::MaxPool2d mxp2d;
    op_maxp():
        mxp2d(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2)))
        {
            register_module("mxp2d", mxp2d);
        }
    torch::Tensor forward(torch::Tensor x)
    {
        x = mxp2d(x);
        return x;
    }
};

struct op_view : public torch::nn::Module
{
    torch::Tensor forward(torch::Tensor x)
    {
        x = x.view({ -1, num_flat_features(x) });
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

struct op_linear : public torch::nn::Module
{
    torch::nn::Linear linear;
    op_linear(int a, int b):
        linear(a, b)
        {
            register_module("linear", linear);
        }
    torch::Tensor forward(torch::Tensor x)
    {
        x = linear(x);
        return x;
    }
};

class alexnet_warmup : public alexnet_part
{
public:
    op_relu op_relu0; //p1
    torch::nn::Conv2d cv0; 

    op_maxp maxp0; //p2
    torch::nn::Conv2d cv1; 
    op_relu op_relu1;
    op_maxp maxp1;
    torch::nn::Conv2d cv2; 

    op_relu op_relu2; //p3

    torch::nn::Conv2d cv3; //p4
    op_relu op_relu3;
    torch::nn::Conv2d cv4; 
    op_relu op_relu4;
    op_maxp maxp2;

    op_linear fc0; //p5
    op_relu op_relu5;
    op_linear fc1;
    op_relu op_relu6;
    op_linear fc2;
 
    alexnet_warmup():
        cv0(torch::nn::Conv2dOptions(3, 64, 11).padding(2).stride(4)),
        cv1(torch::nn::Conv2dOptions(64, 192, 5).padding(2)),
        cv2(torch::nn::Conv2dOptions(192, 384, 3).padding(1)),
        cv3(torch::nn::Conv2dOptions(384, 256, 3).padding(1)),
        cv4(torch::nn::Conv2dOptions(256, 256, 3).padding(1)),
        fc0(9216, 4096),
        fc1(4096, 4096),
        fc2(4096, 1000)
        {
            op_relu0.to(at::kCUDA);
            op_relu1.to(at::kCUDA);
            op_relu2.to(at::kCUDA);
            op_relu3.to(at::kCUDA);
            op_relu4.to(at::kCUDA);
            op_relu5.to(at::kCUDA);
            op_relu6.to(at::kCUDA);
            cv0->to(at::kCUDA);
            cv1->to(at::kCUDA);
            cv2->to(at::kCUDA);
            cv3->to(at::kCUDA);
            cv4->to(at::kCUDA);
            maxp0.to(at::kCUDA);
            maxp1.to(at::kCUDA);
            maxp2.to(at::kCUDA);
            fc0.to(at::kCUDA);
            fc1.to(at::kCUDA);
            fc2.to(at::kCUDA);
        }
    
    torch::Tensor forward(torch::Tensor x)
    {
        // torch::Tensor temp;
        x = cv0(x); 
        std::cout<<x.sizes()<<std::endl; //193600
        x = op_relu0.forward(x);
        std::cout<<x.sizes()<<std::endl;
        x = maxp0.forward(x);
        std::cout<<x.sizes()<<std::endl; //46656

        x = cv1(x); 
        std::cout<<x.sizes()<<std::endl;
        x = op_relu1.forward(x);
        std::cout<<x.sizes()<<std::endl;
        x = maxp1.forward(x);
        std::cout<<x.sizes()<<std::endl;

        x = cv2(x); 
        std::cout<<x.sizes()<<std::endl;
        x = op_relu2.forward(x);
        std::cout<<x.sizes()<<std::endl;
        x = cv3(x); 
        std::cout<<x.sizes()<<std::endl;
        x = op_relu3.forward(x);
        std::cout<<x.sizes()<<std::endl;
        x = cv4(x); 
        std::cout<<x.sizes()<<std::endl;
        x = op_relu4.forward(x);
        std::cout<<x.sizes()<<std::endl;
        x = maxp2.forward(x);
        std::cout<<x.sizes()<<std::endl; // 9216

        x = x.view({ -1, num_flat_features(x)});
        std::cout<<x.sizes()<<std::endl;   // 9216

        x = fc0.forward(x);
        std::cout<<x.sizes()<<std::endl;
        x = op_relu5.forward(x);
        std::cout<<x.sizes()<<std::endl;
        x = fc1.forward(x);
        std::cout<<x.sizes()<<std::endl;
        x = op_relu6.forward(x);
        std::cout<<x.sizes()<<std::endl;
        x = fc2.forward(x);
        std::cout<<x.sizes()<<std::endl;
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
        num_features /= 1;
        return num_features;
    }

    void morphpara(){
        //nothing to do
    }
};

struct alexnet
{
    p1 cpu1; //Conv2d -> relu
    p2 cuda2; //maxpool->conv->relu->maxpool
    p3 cpu3; //relu
    p4 cuda4; //conv->relu->conv->relu->maxpool
    p5 cpu5; //view
    p6 cuda6; //fc->relu->fc->relu->fc

    void ini()
    {
        cpu1.to(at::kCPU);
        cuda2.to(at::kCUDA);
        cpu3.to(at::kCPU);
        cuda4.to(at::kCUDA);
        cpu5.to(at::kCPU);
        cuda6.to(at::kCUDA);
    }

    torch::Tensor forward(torch::Tensor x)
    {
        x = x.to(at::kCPU);
        x = cpu1.forward(x);

        x = x.to(at::kCUDA);
        x = cuda2.forward(x);

        x = x.to(at::kCPU);
        x = cpu3.forward(x);

        x = x.to(at::kCUDA);
        x = cuda4.forward(x);

        x = x.to(at::kCPU);
        x = cpu5.forward(x);

        x = x.to(at::kCUDA);
        x = cuda6.forward(x);

        return x;
    }
};

int main()
{
    std::cout << "AlexNet - 80 % outsourced to CUDA" << std::endl;
    std::cout << "{1, 3, 224, 224}" << std::endl;

    // alexnet model;
    // model.ini();
    alexnet_part* models[] = {
        new alexnet_warmup()
    };

    for (auto &m : models) {
        m->eval();
        // m->to(at::kCUDA);
    }

    torch::Tensor output;

    //torch::Tensor output;
    torch::Tensor input = torch::ones({1, 3, 224, 224}).to(at::kCUDA);;
    int count = 3000;
    int warmup = 1;

    // warm up
    for (size_t i = 0; i < warmup; i++)
    {
        output = models[0]->forward(input);
    }
    
    // cudaEvent_t start, stop;
    // float esp_time_gpu;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);
    // float total = 0;

    // // start measurement
    // for (size_t i = 0; i < count; i++)
    // {
    //     cudaEventRecord(start, 0);
    //     output = models[0]->forward(input);
    //     cudaEventRecord(stop, 0);
    //     cudaEventSynchronize(stop);
    //     cudaEventElapsedTime(&esp_time_gpu, start, stop);
    //     total += esp_time_gpu;
    // }
    // // output = models[0]->forward(input);

    // float latency;
    // latency = total / ((float)count);
    // std::cout << "For " << count << " inferences..." << std::endl;
    // std::cout << "Time elapsed: " << total << " ms." << std::endl;
    // std::cout << "Time consuming: " << latency << " ms per instance." << std::endl;
    
    return 0;
}