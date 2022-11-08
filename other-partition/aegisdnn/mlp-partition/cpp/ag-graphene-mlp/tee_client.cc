#include <stdlib.h>
#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include <random>
#include <time.h>
#include <sys/time.h>
#include <string>
#include <math.h>
#include <fstream>
#include <arpa/inet.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>
#include <memory>
#include <grpcpp/grpcpp.h>
#ifdef BAZEL_BUILD
#include "examples/protos/helloworld.grpc.pb.h"
#else
#include "helloworld.grpc.pb.h"
#endif
#define PORT 8080

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using helloworld::Greeter;
using helloworld::HelloReply;
using helloworld::HelloRequest;

namespace F = torch::nn::functional;
int nbatches = 1;
float D4reshape_fact = 30.0;
float D2reshape_fact = 10.0;
int sock = 0, valread;
struct sockaddr_in serv_addr;
std::string msg = "w";
std::string msg_start = "s";
std::string reply = "0";
int scalar = 4;
int count = 1000; // number of inference tests
int challenge_flag = 8888;
int query_flag = 8889;

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

class GreeterClient {
    public:
        GreeterClient(std::shared_ptr<Channel> channel) : stub_(Greeter::NewStub(channel)) {}

    // Assembles the client's payload, sends it and presents the response back
    // from the server.
    std::string SayHello(const std::string& user, int64_t tag = 0) {
        // Data we are sending to the server.
        HelloRequest request;
        request.set_name(user);
        request.set_tag(tag);
        // Container for the data we expect from the server.
        HelloReply reply;

        // Context for the client. It could be used to convey extra information to
        // the server and/or tweak certain RPC behaviors.
        ClientContext context;

        // The actual RPC.
        Status status = stub_->SayHello(&context, request, &reply);

        // Act upon its status.
        if (status.ok()) {
            return reply.message();
        } else {
            std::cout << status.error_code() << ": " << status.error_message() << std::endl;
            return "RPC failed";
        }
    }
    private:
        std::unique_ptr<Greeter::Stub> stub_;
};

GreeterClient *greeter;
grpc::ChannelArguments ch_args;

std::string request(const char* user_r, int len, int tag) {
    std::string users(user_r, len);
    return greeter->SayHello(users, tag);
}

struct mlp:  public torch::nn::Module
{
    int record_flag = 0;

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
    std::vector<std::function<torch::Tensor(torch::Tensor)>> forwards;
 
    mlp():
        cv0(torch::nn::Conv2dOptions(3, 64, 11).padding(2).stride(4)),
        cv1(torch::nn::Conv2dOptions(64, 192, 5).padding(2)),
        cv2(torch::nn::Conv2dOptions(192, 384, 3).padding(1)),
        cv3(torch::nn::Conv2dOptions(384, 256, 3).padding(1)),
        cv4(torch::nn::Conv2dOptions(256, 256, 3).padding(1)),
        fc0(9216, 4096),
        fc1(4096, 4096),
        fc2(4096, 1000)
        {
            forwards.push_back(std::bind(&mlp::forward1_new, this, std::placeholders::_1));
            forwards.push_back(std::bind(&mlp::forward2_new, this, std::placeholders::_1));
        }
    
    torch::Tensor forward1_new(torch::Tensor x) {
        // std::cout<<"forward1_new"<<std::endl;
        return x;
    }
    torch::Tensor forward2_new(torch::Tensor x) {
        return x.view({ -1, num_flat_features(x)});
    }

    torch::Tensor forward(torch::Tensor x) {
       
        torch::Tensor intermedia;
        torch::Tensor tmp;
        // online inference & fp check 
        std::cout<<"[Inference phase] Inference & integrity check ("<< (record_flag+1) << "/"<<count<<")" <<std::endl; 
        auto reply = greeter->SayHello(msg_start, query_flag);
        record_flag ++;
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
};

int main(int argc, char** argv) {
    std::string target_str;
    std::string arg_str("--target");
    if (argc > 1) {
    std::string arg_val = argv[1];
    size_t start_pos = arg_val.find(arg_str);
    if (start_pos != std::string::npos) {
        start_pos += arg_str.size();
        if (arg_val[start_pos] == '=') {
        target_str = arg_val.substr(start_pos + 1);
        } else {
        std::cout << "The only correct argument syntax is --target="
                    << std::endl;
        return 0;
        }
    } else {
        std::cout << "The only acceptable argument is --target=" << std::endl;
        return 0;
    }
    } else {
    target_str = "localhost:50051";
    }

    mlp model;
    model.eval();

    ch_args.SetMaxReceiveMessageSize(28 * 1024 * 1024); 
    greeter = new GreeterClient(grpc::CreateCustomChannel(target_str, grpc::InsecureChannelCredentials(), ch_args));
    torch::Tensor input = torch::rand({1, 1, 80, 80},torch::dtype(torch::kFloat32));
    torch::Tensor output;

    // GPU warm up
    reply = greeter->SayHello(msg, 0);

    struct timeval tvs, tve;
    gettimeofday(&tvs, 0);
    for (size_t i = 0; i < count; i++){
        output = model.forward(input);
    }
    gettimeofday(&tve, 0);
    float ms_time = (tve.tv_sec - tvs.tv_sec) * 1000 + (tve.tv_usec - tvs.tv_usec) / 1000;
    std::cout << "For " << count << " inferences..." << std::endl;
    std::cout << "Time elapsed: " << ms_time << " ms." << std::endl;
    std::cout << "Fetch here. Time consuming: " << ms_time/count << " ms per inference." << std::endl;
    std::cout << "Completed successfully !!!" << std::endl;

    return 0;
}
