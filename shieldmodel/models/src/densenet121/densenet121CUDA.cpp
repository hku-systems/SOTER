// DenseNet121 - CUDA version

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <torch/torch.h>
#include <iostream>
#include <random>
#include <time.h>
#include <sys/time.h>

class flatten : public torch::nn::Module
{
public:
    torch::Tensor forward(const torch::Tensor& input)
    {
        return input.view({input.sizes()[0], -1});
    }
};

class denselayer : public torch::nn::Module
{
public:
    denselayer (const int inChannels, const int NumChannels):
        layer_net(torch::nn::ReLU(),
                    torch::nn::Conv2d(torch::nn::Conv2dOptions(inChannels, 128, 1).stride(1)),
                    torch::nn::ReLU(),
                    torch::nn::Conv2d(torch::nn::Conv2dOptions(128, NumChannels, 3).padding(1).stride(1)))
                    
        {register_module("layer_net", layer_net);}
                    
        torch::Tensor forward(const torch::Tensor& input)
        {
            return layer_net->forward(input);
        }
     
private:
    torch::nn::Sequential layer_net;
};

class denseblock : public torch::nn::Module
{
public:
    denseblock (const int numLayers, const int inChannels, const int NumChannels) : numLayers_(numLayers), inChannels_(inChannels), NumChannels_(NumChannels)
    {
        for(int i = 0; i < numLayers_; i++)
        {
            dense_block_->push_back(denselayer((inChannels_ + NumChannels_ * i), NumChannels_));
        }

        register_module("dense_block_", dense_block_);
    }

    torch::Tensor forward(const torch::Tensor& input)
    {
        torch::Tensor output = input;
        for(auto& net : *dense_block_)
        {
            auto x = net.forward(output);
            output = torch::cat({output, x}, 1);
        }

        return output;
    }
private:
    int numLayers_;
    int inChannels_;
    int NumChannels_;
    torch::nn::Sequential dense_block_;
};

torch::nn::Sequential transition_block(const int inChannels, const int NumChannels)
{
    torch::nn::Sequential tran_net;
    tran_net->push_back(torch::nn::ReLU());
    tran_net->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(inChannels, NumChannels, 1).stride(1)));
    tran_net->push_back(torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(2).stride(2)));

    return tran_net;
}

class densenet121 : public torch::nn::Module
{
public:
    densenet121()
    {
        network->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 7).padding(3).stride(2)));
        network->push_back(torch::nn::ReLU());
        network->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).padding(1).stride(2)));

        int num_channels = 64;
        const int growth_rate = 32;
        std::array<int, 4> num_layers_in_dense_blocks = {6, 12, 24, 16};
        for (unsigned int i = 0; i < num_layers_in_dense_blocks.size(); ++i)
        {
            network->push_back(denseblock(num_layers_in_dense_blocks[i], num_channels, growth_rate));
            num_channels += num_layers_in_dense_blocks[i] * growth_rate;

            if(i != (num_layers_in_dense_blocks.size() - 1))
            {
                network->extend(*transition_block(num_channels, num_channels/2));
                num_channels /= 2;
            }
        }
        network->push_back(torch::nn::ReLU());
        network->push_back(torch::nn::AdaptiveMaxPool2d(torch::nn::AdaptiveMaxPool2dOptions({1, 1})));
        network->push_back(flatten());
        network->push_back(torch::nn::Linear(num_channels, 1000));

        register_module("network", network);
    }

    torch::Tensor forward(torch::Tensor& input)
    {
        torch::Tensor output = network->forward(input);
        return torch::log_softmax(output, /*dim=*/1);
    }

private:
    torch::nn::Sequential network;

};

int main()
{
    densenet121 model;
    std::cout << "DenseNet121 - CUDA version" << std::endl;
    
    torch::Tensor input = torch::ones({1, 3, 224, 224}).to(at::kCUDA);
    torch::Tensor output;
    
    model.to(at::kCUDA);

    int count = 1000;
    int warmup = 3000;

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

    return 0;
}