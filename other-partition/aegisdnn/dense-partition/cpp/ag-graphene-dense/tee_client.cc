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
std::string msg1 = "1";
std::string msg2 = "2";
std::string msg3 = "3";
std::string msg4 = "4";
std::string msg5 = "5";
std::string msg6 = "6";
std::string msg = "w";
std::string msg_y = "Y";
std::string reply = "0";
int scalar = 4;
int count = 1000; // number of inference tests
int total_para = 0;
void num_flat_features0(torch::Tensor x)
{
    auto size = x.sizes();
    auto num_features = 1;
    for (auto s : size)
    {
        num_features *= s;
    }
    total_para += num_features;
    return;
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
    operator2(int a, int b, int c, int d, int e):
        conv(torch::nn::Conv2dOptions(a, b, c).padding(d).stride(e))
        {
            register_module("conv", conv);
        }
    torch::Tensor forward(torch::Tensor x)
    {
        x = conv(x);
        num_flat_features0(x);
        return x;
    }
};

struct operator3 : public torch::nn::Module
{
    torch::nn::Conv2d conv;
    operator3(int a, int b, int c, int d):
        conv(torch::nn::Conv2dOptions(a, b, c).stride(d))
        {
            register_module("conv", conv);
        }
    torch::Tensor forward(torch::Tensor x)
    {
        x = conv(x);
        num_flat_features0(x);
        return x;
    }
};

struct operator4 : public torch::nn::Module
{
    torch::nn::MaxPool2d mxp2d;
    operator4(int a, int b, int c):
        mxp2d(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(a).padding(b).stride(c)))
        {
            register_module("mxp2d", mxp2d);
        }
    torch::Tensor forward(torch::Tensor x)
    {
        x = mxp2d(x);
        num_flat_features0(x);
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
        num_flat_features0(x);
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
        num_flat_features0(x);
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
        num_flat_features0(x);
        return x;
    }
};

class GreeterClient {
 public:
  GreeterClient(std::shared_ptr<Channel> channel)
      : stub_(Greeter::NewStub(channel)) {}

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
      std::cout << status.error_code() << ": " << status.error_message()
                << std::endl;
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
    // std::cout << "Greeter received: "<< std::endl;
}

struct densenet : public torch::nn::Module
{
    int record_flag = 0;

    operator1 relu;
    operator6 admaxp2d0;
    operator7 linear0;
    operator7 linear2;
    std::vector<std::function<torch::Tensor(torch::Tensor)>> forwards;
        
    densenet():
        admaxp2d0(1, 1),
        linear0(1024, 1000),
        linear2(9000, 1000)
        {
            forwards.push_back(std::bind(&densenet::forward1_new, this, std::placeholders::_1));
            forwards.push_back(std::bind(&densenet::forward2_new, this, std::placeholders::_1));
            forwards.push_back(std::bind(&densenet::forward3_new, this, std::placeholders::_1));
        }

    torch::Tensor forward1_new(torch::Tensor x) {
        return x;
    }
    torch::Tensor forward2_new(torch::Tensor x) {
        x = relu.forward(x);
        x = admaxp2d0.forward(x);
        x = x.view({x.sizes()[0], -1});
        return x;
    }
    torch::Tensor forward3_new(torch::Tensor x) {
        x = linear2.forward(x);
        return x;
    }
    torch::Tensor forward(torch::Tensor x) {
        torch::Tensor intermedia;
        torch::Tensor tmp;
        // online inference & fp check 
        std::cout<<"[Inference phase] Inference & integrity check ("<< (record_flag+1) << "/"<<count<<")" <<std::endl; 
        for (int i = 0; i < 3;i++) {
            // intercat = torch::cat({fp_check, intermedia},0);
            intermedia = forwards[i](x);
            std::stringstream ss;
            torch::save(intermedia, ss);
            auto reply = request(ss.str().data(), ss.str().size(), i + 1);
            std::istringstream is(reply);
            torch::load(tmp, is);             
            x = tmp;
        }
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
        num_features /= nbatches;
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

    densenet model;
    model.eval();

    ch_args.SetMaxReceiveMessageSize(28 * 1024 * 1024); 
    greeter = new GreeterClient(grpc::CreateCustomChannel(target_str, grpc::InsecureChannelCredentials(), ch_args));
    torch::Tensor input = torch::rand({1, 3, 224, 224},torch::dtype(torch::kFloat32));
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
