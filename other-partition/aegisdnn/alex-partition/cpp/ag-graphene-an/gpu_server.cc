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
#include <torch/library.h>
#include <ATen/ATen.h>
#include <vector>
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
int server_fd, new_socket, valread;
struct sockaddr_in address;
torch::Tensor input = torch::rand({1, 3, 224, 224}, torch::dtype(torch::kFloat32)).to(at::kCUDA);
torch::Tensor output;
int challenge_flag = 8888;
int query_flag = 8889;

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
        x = cv0(x); 
        x = op_relu0.forward(x);
        x = maxp0.forward(x);

        x = cv1(x); 
        x = op_relu1.forward(x);
        x = maxp1.forward(x);

        x = cv2(x); 
        x = op_relu2.forward(x);
        x = cv3(x); 
        x = op_relu3.forward(x);
        x = cv4(x); 
        x = op_relu4.forward(x);
        x = maxp2.forward(x);

        x = x.view({ -1, num_flat_features(x)});

        x = fc0.forward(x);
        x = op_relu5.forward(x);
        x = fc1.forward(x);
        x = op_relu6.forward(x);
        x = fc2.forward(x);
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
        //nothing to do in warmup
    }
};

class alexnet_gpu_part1 : public alexnet_part
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
 
    alexnet_gpu_part1():
        cv0(torch::nn::Conv2dOptions(3, 64, 11).padding(2).stride(4)),
        cv1(torch::nn::Conv2dOptions(64, 192, 5).padding(2)),
        cv2(torch::nn::Conv2dOptions(192, 384, 3).padding(1)),
        cv3(torch::nn::Conv2dOptions(384, 256, 3).padding(1)),
        cv4(torch::nn::Conv2dOptions(256, 256, 3).padding(1))
        {
            op_relu0.to(at::kCUDA);
            op_relu1.to(at::kCUDA);
            op_relu2.to(at::kCUDA);
            op_relu3.to(at::kCUDA);
            op_relu4.to(at::kCUDA);
            cv0->to(at::kCUDA);
            cv1->to(at::kCUDA);
            cv2->to(at::kCUDA);
            cv3->to(at::kCUDA);
            cv4->to(at::kCUDA);
            maxp0.to(at::kCUDA);
            maxp1.to(at::kCUDA);
            maxp2.to(at::kCUDA);
        }
    
    torch::Tensor forward(torch::Tensor x)
    {
        x = cv0(x); 
        x = op_relu0.forward(x);
        x = maxp0.forward(x);
        x = cv1(x); 
        x = op_relu1.forward(x);
        x = maxp1.forward(x);
        x = cv2(x); 
        x = op_relu2.forward(x);
        x = cv3(x); 
        x = op_relu3.forward(x);
        x = cv4(x); 
        x = op_relu4.forward(x);
        x = maxp2.forward(x);
        return x;
    }

    void morphpara(){
    }
};


class alexnet_gpu_part2 : public alexnet_part
{
public:
    torch::nn::Linear fc0; //p5
    op_relu op_relu5;
    torch::nn::Linear fc1;
    op_relu op_relu6;
    torch::nn::Linear fc2;
    alexnet_gpu_part2():
        fc0(9216, 4096),
        fc1(4096, 4096),
        fc2(4096, 1000)
        {
            op_relu5.to(at::kCUDA);
            op_relu6.to(at::kCUDA);
            fc0->to(at::kCUDA);
            fc1->to(at::kCUDA);
            fc2->to(at::kCUDA);
        }
    
    torch::Tensor forward(torch::Tensor x)
    { 
        x = fc0(x);
        x = op_relu5.forward(x);
        x = fc1(x);
        x = op_relu6.forward(x);
        x = fc2(x);
        return x;
    }

    void morphpara(){
    }
};

alexnet_part* models[] = {
    new alexnet_warmup(),
    new alexnet_gpu_part1(),
    new alexnet_gpu_part2()
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
    } else if (tag == challenge_flag) {
        torch::Tensor inter_active;
        std::istringstream ss(request->name());
        torch::load(inter_active, ss);
        output = models[0]->forward(inter_active.to(at::kCUDA));
        std::stringstream so;
        torch::save(output.to(at::kCPU), so);
        reply->set_message(so.str());
    } else if (tag == query_flag) {
        output = models[0]->forward(input);
        std::stringstream so;
        torch::save(output.to(at::kCPU), so);
        reply->set_message(so.str());
    } else {
        torch::Tensor inter_active;
        std::istringstream ss(request->name());
        torch::load(inter_active, ss);
        output = models[tag]->forward(inter_active.to(at::kCUDA)); 
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
    RunServer();
    return 0;
}
