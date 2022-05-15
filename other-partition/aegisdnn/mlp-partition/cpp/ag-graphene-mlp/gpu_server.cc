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
torch::Tensor input = torch::rand({1, 1, 80, 80}, torch::dtype(torch::kFloat32)).to(at::kCUDA);
torch::Tensor output;
int challenge_flag = 8888;
int query_flag = 8889;
int scalar = 4;

class mlp_part : public torch::nn::Module {
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

class mlp_warmup : public mlp_part
{
public:
    op_linear fc0; //p5
    op_relu op_relu5;
    op_linear fc1;
    op_relu op_relu6;
    op_linear fc2;
    op_linear fc3;
 
    mlp_warmup():
        fc0(6400, 4096),
        fc1(4096, 512),
        fc2(512, 128),
        fc3(128, 10)
        {
            op_relu5.to(at::kCUDA);
            op_relu6.to(at::kCUDA);
            fc0.to(at::kCUDA);
            fc1.to(at::kCUDA);
            fc2.to(at::kCUDA);
            fc3.to(at::kCUDA);
        }
    
    torch::Tensor forward(torch::Tensor x)
    {
        x = x.view({ -1, 80 * 80 });
        x = fc0.forward(x);
        x = op_relu5.forward(x);
        x = fc1.forward(x);
        x = op_relu6.forward(x);
        x = fc2.forward(x);
        x = op_relu6.forward(x);
        x = fc3.forward(x);
        // std::cout<<"x "<<x.sizes()<<std::endl;
        return x;
    }
    void morphpara(){
        //nothing to do in warmup
    }
};

class mlp_gpu_part1 : public mlp_part
{
public:
    torch::nn::Linear fc0; //p5
    op_relu op_relu5;
    torch::nn::Linear fc1;
    op_relu op_relu6;
    torch::nn::Linear fc2;
    torch::nn::Linear fc3;
 
    mlp_gpu_part1():
        fc0(6400, 4096),
        fc1(4096, 512),
        fc2(512, 128),
        fc3(128, 10)
        {
            op_relu5.to(at::kCUDA);
            op_relu6.to(at::kCUDA);
            fc0->to(at::kCUDA);
            fc1->to(at::kCUDA);
            fc2->to(at::kCUDA);
            fc3->to(at::kCUDA);
        }
    
    torch::Tensor forward(torch::Tensor x)
    {
        x = x.view({ -1, 80 * 80 });
        x = fc0->forward(x);
        x = op_relu5.forward(x);
        x = fc1->forward(x);
        x = op_relu6.forward(x);
        x = fc2->forward(x);
        x = op_relu6.forward(x);
        x = fc3->forward(x);
        // std::cout<<"x "<<x.sizes()<<std::endl;
        return x;
    }
    void morphpara(){
    }
};


mlp_part* models[] = {
    new mlp_warmup(),
    new mlp_gpu_part1()
};

// Logic and data behind the server's behavior.
class GreeterServiceImpl final : public Greeter::Service {
  std::string prefix;
  
  Status SayHello(ServerContext* context, const HelloRequest* request,
                  HelloReply* reply) override {
    int tag = request->tag();
    if (tag == 0) {
        std::cout << "[Preprocessing phase] Receive TEE signal, GPU is now warming up ..."<< std::endl;
        for (size_t i = 0; i < 100; i++) {
            output = models[tag]->forward(input);
        }
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
        // std::stringstream so;
        // torch::save(output.to(at::kCPU), so);
        // reply->set_message(so.str());
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
