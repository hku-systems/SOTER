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

struct vgg19 : public torch::nn::Module
{
    int record_flag = 0;
    
    operator1 relu;
    operator4 mxp2d0;
    torch::nn::Conv2d c;
    operator2 cv0;
    operator2 cv1;
    operator2 cv2;
    operator2 cv3;
    operator2 cv4;
    operator2 cv5;
    operator2 cv6;
    operator2 cv7;
    operator2 cv8;
    operator2 cv9;
    operator2 cv10;
    operator2 cv11;
    operator2 cv12;
    operator2 cv13;
    torch::nn::Conv2d cv14;
    operator7 fc0;
    operator7 fc1;
    operator7 fc2;
    std::vector<std::function<torch::Tensor(torch::Tensor)>> forwards;
        
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
            register_module("c", c);
	        register_module("cv14", cv14);
            forwards.push_back(std::bind(&vgg19::forward1_new, this, std::placeholders::_1));
            forwards.push_back(std::bind(&vgg19::forward2_new, this, std::placeholders::_1));
            forwards.push_back(std::bind(&vgg19::forward3_new, this, std::placeholders::_1));
            forwards.push_back(std::bind(&vgg19::forward4_new, this, std::placeholders::_1));

        }

    torch::Tensor forward1_new(torch::Tensor x) {
        // std::cout<<"forward1_new"<<std::endl;
        return x;
    }
    torch::Tensor forward2_new(torch::Tensor x) {
        // x = F::relu(x);
        x = torch::max_pool2d(x, 2, 2, 0);
        return x;
    }
    torch::Tensor forward3_new(torch::Tensor x) {
        
        x = x.view({ -1, num_flat_features(x)});
        return x;
    }

    torch::Tensor forward4_new(torch::Tensor x) { 
        x = fc2.forward(x);
        return x;
    }

    torch::Tensor forward(torch::Tensor x) {
        torch::Tensor intermedia;
        torch::Tensor tmp;
        // online inference & fp check 
        std::cout<<"[Inference phase] Inference & integrity check ("<< (record_flag+1) << "/"<<count<<")" <<std::endl; 
        for (int i = 0; i < 4;i++) {
            // intercat = torch::cat({fp_check, intermedia},0);
            intermedia = forwards[i](x);
            std::stringstream ss;
            torch::save(intermedia, ss);
            auto reply = request(ss.str().data(), ss.str().size(), i + 1);
            std::istringstream is(reply);
            torch::load(tmp, is);             
            x = tmp/scalar; // restore 11ms
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

    vgg19 model;
    model.eval();

    ch_args.SetMaxReceiveMessageSize(28 * 1024 * 1024); 
    greeter = new GreeterClient(grpc::CreateCustomChannel(target_str, grpc::InsecureChannelCredentials(), ch_args));
    torch::Tensor input = torch::rand({1, 3, 224, 224},torch::dtype(torch::kFloat32));
    torch::Tensor output;

    // GPU warm up
    reply = greeter->SayHello(msg, 0);
    // std::cout << "[Preprocessing phase (1/3)] Model partitioned & Parameter morphed!" <<std::endl;

    int prepare = 2;
    for (size_t i = 0; i < prepare; i++){
        output = model.forward(input); 
    }

    struct timeval tvs, tve;
    gettimeofday(&tvs, 0);
    for (size_t i = 0; i < count; i++){
        output = model.forward(input);
    }
    gettimeofday(&tve, 0);
    float ms_time = (tve.tv_sec - tvs.tv_sec) * 1000 + (tve.tv_usec - tvs.tv_usec) / 1000;
    std::cout << "For " << count << " inferences ..." << std::endl;
    std::cout << "Time elapsed: " << ms_time << " ms." << std::endl;
    std::cout << "Fetch here. Time consuming: " << ms_time/count << " ms per inference." << std::endl;
    std::cout << "Completed successfully !!!" << std::endl;

    return 0;
}
