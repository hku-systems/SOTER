// DenseNet121 - CPU version

#include <torch/torch.h>
#include <iostream>
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
        network->push_back(torch::nn::Linear(num_channels, 32768));
        network->push_back(torch::nn::Linear(32768, 4096));
        network->push_back(torch::nn::Linear(4096, 1000));

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
    std::cout << "DenseNet121 - Graphene version" << std::endl;
    
    int count = 1000;
    auto input = torch::ones({1, 3, 224, 224});
    torch::Tensor output;
    
    double start = time_get();
    double stop = time_get();
    double duration = 0;
    for (int i = 0; i < count; i++)
    {
        start = time_get();
        output = model.forward(input);
        stop = time_get();
        duration += (stop - start);
    }
    double latency = duration / ((double)count);
    std::cout << "For 1,000 " << count << " inferences..." << std::endl;
    std::cout << "Time elapsed: " << duration << " ms." << std::endl;
    std::cout << "Fetch here. Time consuming: " <<latency << " ms per inference." << std::endl;

    std::cout << "Completed." << std::endl;

    return 0;
}