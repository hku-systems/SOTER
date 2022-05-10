#include "torch/script.h"
#include <torch/torch.h>
#include <iostream>
#include <random>
#include <time.h>
#include <sys/time.h>
#include <string>
#include <math.h>
#include <fstream>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>
#include <memory>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>

#ifdef BAZEL_BUILD
#include "examples/protos/helloworld.grpc.pb.h"
#else
#include "helloworld.grpc.pb.h"
#endif

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using helloworld::Greeter;
using helloworld::HelloReply;
using helloworld::HelloRequest;

#define PORT 8080

namespace F = torch::nn::functional;
int nbatches = 1;
float D4reshape_fact = 30.0;
float D2reshape_fact = 10.0;

int server_fd, new_socket, valread;
struct sockaddr_in address;
int opt = 1;
int addrlen = sizeof(address);
char buffer[1024] = { 0 };

torch::Tensor input = torch::rand({1, 3, 224, 224}, torch::dtype(torch::kFloat32)).to(at::kCUDA);
torch::Tensor output;
torch::Tensor output_test;

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

class vgg19_part : public torch::nn::Module {
    public:
    virtual torch::Tensor forward(torch::Tensor x) = 0;
    virtual void morphpara() = 0;
};

class vgg19_warmup : public vgg19_part
{
public:
    operator1 relu;
    operator4 mxp2d0;
    torch::nn::Conv2d c;
    torch::nn::Conv2d cv0;
    torch::nn::Conv2d cv1;
    torch::nn::Conv2d cv2;
    torch::nn::Conv2d cv3;
    torch::nn::Conv2d cv4;
    torch::nn::Conv2d cv5;
    torch::nn::Conv2d cv6;
    torch::nn::Conv2d cv7;
    torch::nn::Conv2d cv8;
    torch::nn::Conv2d cv9;
    torch::nn::Conv2d cv10;
    torch::nn::Conv2d cv11;
    torch::nn::Conv2d cv12;
    torch::nn::Conv2d cv13;
    torch::nn::Conv2d cv14;
    torch::nn::Linear fc0;
    torch::nn::Linear fc1;
    torch::nn::Linear fc2;

    vgg19_warmup():
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
            register_module("cv3", cv3);
            register_module("cv4", cv4);
            register_module("cv5", cv5);
            register_module("cv6", cv6);
            register_module("cv7", cv7);
            register_module("cv8", cv8);
            register_module("cv9", cv9);
            register_module("cv10", cv10);
            register_module("cv11", cv11);
            register_module("cv12", cv12);
            register_module("cv13", cv13);
            register_module("cv14", cv14);
            register_module("fc0", fc0);
            register_module("fc1", fc1);
            register_module("fc2", fc2);
        }
    
    torch::Tensor forward(torch::Tensor x)
    {
        torch::Tensor temp;

        x = c(x);

        x = F::relu(x);
        x = cv0(x);
        x = F::relu(x);
        x = torch::max_pool2d(x, 2, 2, 0);
        x = cv1(x);
        x = F::relu(x);
        x = cv2(x);
        x = F::relu(x);
        x = torch::max_pool2d(x, 2, 2, 0);
        x = cv3(x);
        x = F::relu(x);
        x = cv4(x);
        x = F::relu(x);
        x = cv5(x);
        x = F::relu(x);
        x = cv6(x);
        x = F::relu(x);
        x = torch::max_pool2d(x, 2, 2, 0);
        x = cv7(x);
        x = F::relu(x);
        x = cv8(x);
        x = F::relu(x);
        x = cv9(x);
        x = F::relu(x);
        x = cv10(x);
        x = F::relu(x);
        x = torch::max_pool2d(x, 2, 2, 0);
        x = cv11(x);
        x = F::relu(x);
        x = cv12(x);
        x = F::relu(x);
        x = cv13(x);
        x = F::relu(x);
        x = cv14(x);
        x = F::relu(x);
        x = torch::max_pool2d(x, 2, 2, 0);
        x = x.view({ -1, num_flat_features(x)});
        x = fc0(x);
        x = F::relu(x);
        x = fc1(x);
        x = F::relu(x);
        x = fc2(x);
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
    void morphpara(){
        // nothing to do, inherit from virtual func
    }
};

class vgg19_gpu_part1_new : public vgg19_part
{
public:
    operator1 relu;
    operator4 mxp2d0;
    torch::nn::Conv2d c;
    torch::nn::Conv2d cv0;
    torch::nn::Conv2d cv1;
    torch::nn::Conv2d cv2;
    torch::nn::Conv2d cv3;
    torch::nn::Conv2d cv4;
    torch::nn::Conv2d cv5;
    torch::nn::Conv2d cv6;
    torch::nn::Conv2d cv7;
    torch::nn::Conv2d cv8;
    torch::nn::Conv2d cv9;
    torch::nn::Conv2d cv10;
    torch::nn::Conv2d cv11;
    torch::nn::Conv2d cv12;
    torch::nn::Conv2d cv13;
    torch::nn::Conv2d cv14;

    vgg19_gpu_part1_new():
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
        cv14(conv_options(512, 512, 3, 1, 1))
        {
            relu.to(at::kCUDA);
            mxp2d0.to(at::kCUDA);
            // register_module("c", c);
            c->to(at::kCUDA);
            cv0->to(at::kCUDA);
            cv1->to(at::kCUDA);
            cv2->to(at::kCUDA);
            cv3->to(at::kCUDA);
            cv4->to(at::kCUDA);
            cv5->to(at::kCUDA);
            cv6->to(at::kCUDA);
            cv7->to(at::kCUDA);
            cv8->to(at::kCUDA);
            cv9->to(at::kCUDA);
            cv10->to(at::kCUDA);
            cv11->to(at::kCUDA);
            cv12->to(at::kCUDA);
            cv13->to(at::kCUDA);
            cv14->to(at::kCUDA);
        }
    
    torch::Tensor forward(torch::Tensor x)
    {
        torch::Tensor temp;
        x = c(x);
        x = F::relu(x);
        x = cv0(x);
        x = F::relu(x);
        x = torch::max_pool2d(x, 2, 2, 0);
        x = cv1(x);
        x = F::relu(x);
        x = cv2(x);
        x = F::relu(x);
        x = torch::max_pool2d(x, 2, 2, 0);
        x = cv3(x);
        x = F::relu(x);
        x = cv4(x);
        x = F::relu(x);
        x = cv5(x);
        x = F::relu(x);
        x = cv6(x);
        x = F::relu(x);
        x = torch::max_pool2d(x, 2, 2, 0);
        x = cv7(x);
        x = F::relu(x);
        x = cv8(x);
        x = F::relu(x);
        x = cv9(x);
        x = F::relu(x);
        x = cv10(x);
        x = F::relu(x);
        x = torch::max_pool2d(x, 2, 2, 0);
        x = cv11(x);
        x = F::relu(x);
        x = cv12(x);
        x = F::relu(x);
        x = cv13(x);
        x = F::relu(x);
        x = cv14(x);
        x = F::relu(x);
        return x;
    }

    void morphpara(){
        int scalar = 4;
        c->weight = c->weight * scalar;
        cv0->weight = cv0->weight * scalar;
        cv1->weight = cv1->weight * scalar;
        cv2->weight = cv2->weight * scalar;
        cv3->weight = cv3->weight * scalar;
        cv4->weight = cv4->weight * scalar;
        cv5->weight = cv5->weight * scalar;
        cv6->weight = cv6->weight * scalar;
        cv7->weight = cv7->weight * scalar;
        cv8->weight = cv8->weight * scalar;
        cv9->weight = cv9->weight * scalar;
        cv10->weight = cv10->weight * scalar;
        cv11->weight = cv11->weight * scalar;
        cv12->weight = cv12->weight * scalar;
        cv13->weight = cv13->weight * scalar;
        cv14->weight = cv14->weight * scalar;
    }

};

class vgg19_gpu_part2_new : public vgg19_part
{
public:
    operator1 relu;
    operator4 mxp2d0;
    torch::nn::Linear fc0;
    torch::nn::Linear fc1;
    torch::nn::Linear fc2;

    vgg19_gpu_part2_new():
        mxp2d0(2, 2, 0),
        fc0(25088, 4096),
        fc1(4096, 4096),
        fc2(4096, 1000)
        {
            relu.to(at::kCUDA);
            mxp2d0.to(at::kCUDA);
            fc0->to(at::kCUDA);
            fc1->to(at::kCUDA);
            fc2->to(at::kCUDA);
        }
    
    torch::Tensor forward(torch::Tensor x)
    {  
        x = fc0(x);
        x = F::relu(x);
        x = fc1(x);
        x = F::relu(x);
        x = fc2(x);
        return x;
    }

    void morphpara(){
        int scalar = 4;
        fc0->weight = fc0->weight * scalar;
        fc0->bias = fc0->bias * scalar;
        fc1->weight = fc1->weight * scalar;
        fc1->bias = fc1->bias * scalar;
        fc2->weight = fc2->weight * scalar;
        fc2->bias = fc2->bias * scalar;        
    }

};


class vgg19_gpu_part1_fp: public vgg19_part
{
public:
    torch::nn::Conv2d c;
    vgg19_gpu_part1_fp():
        c(conv_options(3, 64, 3, 1, 1))
        {
            c->to(at::kCUDA);
        } 
    torch::Tensor forward(torch::Tensor x)
    {
        x = c(x);
        return x;
    }
    void morphpara(){}
};

class vgg19_gpu_part2_fp: public vgg19_part
{
public:
    torch::nn::Linear fc0;
    vgg19_gpu_part2_fp():
        fc0(25088, 4096)
        {
            fc0->to(at::kCUDA);
        } 
    torch::Tensor forward(torch::Tensor x)
    {
        x = fc0(x);
        return x;
    }
    void morphpara(){}
};

vgg19_part* models[] = {
    new vgg19_warmup(),
    new vgg19_gpu_part1_new(),
    new vgg19_gpu_part2_new(),
    new vgg19_gpu_part1_fp(),
    new vgg19_gpu_part2_fp()
};

// Logic and data behind the server's behavior.
class GreeterServiceImpl final : public Greeter::Service {
  std::string prefix;
  
  Status SayHello(ServerContext* context, const HelloRequest* request,
                  HelloReply* reply) override {
    int tag = request->tag();
    if (tag == 0) {
        for (size_t i = 0; i < 100; i++) {
            output = models[tag]->forward(input);
        }
        std::cout << "[Preprocessing phase] Receive TEE signal, GPU is now warming up ..."<< std::endl;
        std::cout << "[Inference phase] GPU is serving ..." << std::endl;
    } else {
        torch::Tensor inter_active;
        std::istringstream ss(request->name());
        torch::load(inter_active, ss);
        output = models[tag]->forward(inter_active.to(at::kCUDA)); //56ms
        // output = torch::rand({1, 3, 224, 224}).to(at::kCUDA); //23ms
        std::stringstream so;
        torch::save(output.to(at::kCPU), so);
        reply->set_message(so.str());
    }  
    return Status::OK;
  }
};

void RunServer() {
    for (auto &m : models) {
        m->eval();
        m->morphpara();
        m->to(at::kCUDA);
    }

    std::string server_address("0.0.0.0:50051");
    GreeterServiceImpl service;

    grpc::EnableDefaultHealthCheckService(true);
    grpc::reflection::InitProtoReflectionServerBuilderPlugin();
    ServerBuilder builder;
    // Listen on the given address without any authentication mechanism.
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    // Register "service" as the instance through which we'll communicate with
    // clients. In this case it corresponds to an *synchronous* service.
    builder.RegisterService(&service);
    builder.SetMaxReceiveMessageSize(28 * 1024 * 1024);
    builder.SetMaxSendMessageSize(28 * 1024 * 1024);
    // Finally assemble the server.
    std::unique_ptr<Server> server(builder.BuildAndStart());
    std::cout << "[Preprocessing phase] GPU is listening on " << server_address << std::endl;

    // Wait for the server to shutdown. Note that some other thread must be
    // responsible for shutting down the server for this call to ever return.
    server->Wait();
}

int main(int argc, char* argv[])
{
    // torch::Tensor t1 = torch::rand({1, 3, 224, 224});
    // torch::Tensor t2 = torch::rand({1, 3, 224, 224});
    // output = torch::cat({t1, t2},0);
    // std::cout << output.sizes() << std::endl;
    RunServer();
    return 0;
}
