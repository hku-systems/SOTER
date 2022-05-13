#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <torch/torch.h>
#include <iostream>
#include <random>
#include <time.h>
#include <sys/time.h>
#include <string>
#include <math.h>

namespace F = torch::nn::functional;
int nbatches;


torch::nn::Conv2dOptions conv_options(int64_t in_planes, int64_t out_planes, int64_t kerner_size, int64_t stride=1, int64_t padding=0, bool with_bias=false)
{
  torch::nn::Conv2dOptions conv_options = torch::nn::Conv2dOptions(in_planes, out_planes, kerner_size);
  conv_options.stride(stride);
  conv_options.padding(padding);
  conv_options.bias(with_bias);
  return conv_options;
}

struct operator1 : public torch::nn::Module
{
    torch::Tensor forward(torch::Tensor x)
    {
        x = F::relu(x);
        return x;   
    }
};

struct operator2 : public torch::nn::Module
{
    torch::nn::Conv2d conv;
    operator2(torch::nn::Conv2dOptions a):
        conv(a)
        {
            register_module("conv", conv);
        }
    torch::Tensor forward(torch::Tensor x)
    {
        x = conv(x);
        return x;
    }
};

struct operator3 : public torch::nn::Module
{
    torch::nn::Conv2d conv;
    operator3(int a, int b, int c, int d):
        conv(conv_options(a, b, c, d))
        {
            register_module("conv", conv);
        }
    torch::Tensor forward(torch::Tensor x)
    {
        x = conv(x);
        return x;
    }
};

struct operator4 : public torch::nn::Module
{
    torch::nn::MaxPool2d mxp2d;
    operator4(int a, int b, int c):
        mxp2d(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(a).padding(c).stride(b)))
        {
            register_module("mxp2d", mxp2d);
        }
    torch::Tensor forward(torch::Tensor x)
    {
        x = mxp2d(x);
        return x;
    }
};

struct operator5 : public torch::nn::Module
{
    torch::nn::AvgPool2d avg2d;
    operator5(int a, int b):
        avg2d(torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(a).stride(b)))
        {
            register_module("avg2d", avg2d);
        }
    torch::Tensor forward(torch::Tensor x)
    {
        x = avg2d(x);
        return x;
    }
};

struct operator6 : public torch::nn::Module
{
    torch::nn::AdaptiveMaxPool2d admxp2d;
    operator6(int a, int b):
        admxp2d(torch::nn::AdaptiveMaxPool2d(torch::nn::AdaptiveMaxPool2dOptions({a, b})))
        {
            register_module("admxp2d", admxp2d);
        }
    torch::Tensor forward(torch::Tensor x)
    {
        x = admxp2d(x);
        return x;
    }
};

struct operator7 : public torch::nn::Module
{
    torch::nn::Linear linear;
    operator7(int a, int b):
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

struct operator8 : public torch::nn::Module
{
    torch::nn::Conv2d conv;
    operator8(int a, int b, int c):
        conv(conv_options(a, b, c))
        {
            register_module("conv", conv);
        }
    torch::Tensor forward(torch::Tensor x)
    {
        x = conv(x);
        return x;
    }
};

struct vgg19 : public torch::nn::Module
{
    operator1 relu;
    operator4 mxp2d0;
    torch::nn::Conv2d c;
    torch::nn::Conv2d cv0;
    torch::nn::Conv2d cv1;
    torch::nn::Conv2d cv2;
    operator2 cv3;
    torch::nn::Conv2d cv4;
    operator2 cv5;
    operator2 cv6;
    operator2 cv7;
    torch::nn::Conv2d cv8;
    operator2 cv9;
    operator2 cv10;
    operator2 cv11;
    operator2 cv12;
    operator2 cv13;
    operator2 cv14;
    operator7 fc0;
    operator7 fc1;
    operator7 fc2;

    vgg19():
        mxp2d0(2, 2, 0),
        c(conv_options(3, 64, 3, 1, 1)),
        cv0(conv_options(64, 64, 3, 1, 1)),
        cv1(conv_options(64, 128, 3, 1, 1)),
        cv2(conv_options(128, 128, 3, 1, 1)),
        cv3(conv_options(128, 256, 3, 1, 1)),
        cv4(conv_options(256, 256, 3, 1, 1)),
        cv5(conv_options(256, 256, 3, 1, 1)),
        cv6(conv_options(256, 256, 3, 1, 1)),
        cv7(conv_options(256, 512, 3, 1, 1)),
        cv8(conv_options(512, 512, 3, 1, 1)),
        cv9(conv_options(512, 512, 3, 1, 1)),
        cv10(conv_options(512, 512, 3, 1, 1)),
        cv11(conv_options(512, 512, 3, 1, 1)),
        cv12(conv_options(512, 512, 3, 1, 1)),
        cv13(conv_options(512, 512, 3, 1, 1)),
        cv14(conv_options(512, 512, 3, 1, 1)),
        fc0(25088, 4096),
        fc1(4096, 4096),
        fc2(4096, 1000)
        {
            relu.to(at::kCUDA);
            mxp2d0.to(at::kCUDA);
            register_module("c", c);
            register_module("cv0", cv0);
            register_module("cv1", cv1);
            register_module("cv2", cv2);
            cv3.to(at::kCUDA);
            register_module("cv4", cv4);
            cv5.to(at::kCUDA);
            cv6.to(at::kCUDA);
            cv7.to(at::kCUDA);
            register_module("cv8", cv8);
            cv9.to(at::kCUDA);
            cv10.to(at::kCUDA);
            cv11.to(at::kCUDA);
            cv12.to(at::kCUDA);
            cv13.to(at::kCUDA);
            cv14.to(at::kCUDA);
            fc0.to(at::kCUDA);
            fc1.to(at::kCUDA);
            fc2.to(at::kCUDA);
        }
    
    torch::Tensor forward(torch::Tensor x)
    {
        torch::Tensor temp;

        x = c(x);

        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);
        x = cv0(x);
        x = F::relu(x);
        temp = x.to(at::kCUDA);
        temp = mxp2d0.forward(temp);
        x = temp.to(at::kCPU);
        x = cv1(x);
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);
        x = cv2(x);
        x = F::relu(x);
        temp = x.to(at::kCUDA);
        temp = mxp2d0.forward(temp);
        temp = cv3.forward(temp);
        x = temp.to(at::kCPU);
        x = F::relu(x);
        x = cv4(x);
        x = F::relu(x);
        temp = x.to(at::kCUDA);
        temp = cv5.forward(temp);
        temp = relu.forward(temp);
        temp = cv6.forward(temp);
        x = temp.to(at::kCPU);
        x = F::relu(x);
        temp = x.to(at::kCUDA);
        temp = mxp2d0.forward(temp);
        temp = cv7.forward(temp);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);
        x = cv8(x);
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = cv9.forward(temp);
        x = temp.to(at::kCPU);
        x = F::relu(x);
        temp = x.to(at::kCUDA);
        temp = cv10.forward(temp);
        x = temp.to(at::kCPU);
        x = F::relu(x);
        temp = x.to(at::kCUDA);
        temp = mxp2d0.forward(temp);
        temp = cv11.forward(temp);
        temp = relu.forward(temp);
        temp = cv12.forward(temp);
        temp = relu.forward(temp);
        temp = cv13.forward(temp);
        x = temp.to(at::kCPU);
        x = F::relu(x);
        temp = x.to(at::kCUDA);
        temp = cv14.forward(temp);
        temp = relu.forward(temp);
        temp = mxp2d0.forward(temp);
        x = temp.to(at::kCPU);
        x = x.view({ -1, num_flat_features(x)});
        temp = x.to(at::kCUDA);
        temp = fc0.forward(temp);
        temp = relu.forward(temp);
        temp = fc1.forward(temp);
        x = temp.to(at::kCPU);
        x = F::relu(x);
        temp = x.to(at::kCUDA);
        temp = fc2.forward(temp);
        x = temp.to(at::kCPU);

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
        num_features /= nbatches;
        return num_features;
    }

};

int main(int argc, char* argv[])
{
    std::cout << "Please input the batch size..." << std::endl;
    nbatches = std::stoi(argv[1]);
    std::cout << "Read integer: " << nbatches << std::endl;

    std::cout << "vgg19 {" << nbatches << " , 3, 224, 224} 60% outsourced to CUDA version" << std::endl;
    std::cout << "Kernel Switch: 24" << std::endl;
    std::cout << "Associative Op: 41" << std::endl;
    std::cout << "Outsourced Op: 27" << std::endl;
    vgg19 model;
    int count = 1000;
    int warmup = 1000;

    torch::Tensor input = torch::rand({nbatches, 3, 224, 224});
    torch::Tensor output;

    for (size_t i = 0; i < warmup; i++)
    {
        output = model.forward(input);
    }
    
    cudaEvent_t start, stop;
    float esp_time_gpu;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float total = 0;

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
    std::cout << "completed." << std::endl;

    return 0;
}