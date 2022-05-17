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
int total_para = 0;
int server_fd, new_socket, valread;
struct sockaddr_in address;
int opt = 1;
int addrlen = sizeof(address);
char buffer[1024] = { 0 };

torch::Tensor input = torch::rand({1, 3, 224, 224}, torch::dtype(torch::kFloat32)).to(at::kCUDA);
torch::Tensor output;
torch::Tensor output_test;

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

class dense_part : public torch::nn::Module {
    public:
    virtual torch::Tensor forward(torch::Tensor x) = 0;
    virtual void morphpara() = 0;
};

class dense_warmup : public dense_part
{
public:
    torch::nn::Conv2d c0;
    //torch::nn::BatchNorm2d b0;
    operator4 mxp2d0;
    operator1 relu;

    operator3 cv_a_0;
    torch::nn::Conv2d cv_a_1;
    operator3 cv_a_2;
    operator3 cv_a_3;
    operator3 cv_a_4;
    operator3 cv_a_5;
    operator3 cv_a_6;
    operator3 cv_a_7;
    torch::nn::Conv2d cv_a_8;
    operator3 cv_a_9;
    operator3 cv_a_10;
    operator3 cv_a_11;
    operator3 cv_a_12;
    operator3 cv_a_13;
    operator3 cv_a_14;
    operator3 cv_a_15;
    operator3 cv_a_16;
    torch::nn::Conv2d cv_a_17;
    operator3 cv_a_18;
    operator3 cv_a_19;
    operator3 cv_a_20;
    operator3 cv_a_21;
    operator3 cv_a_22;
    operator3 cv_a_23;
    operator3 cv_a_24;
    operator3 cv_a_25;
    operator3 cv_a_26;
    operator3 cv_a_27;
    operator3 cv_a_28;
    operator3 cv_a_29;
    operator3 cv_a_30;
    operator3 cv_a_31;
    operator3 cv_a_32;
    operator3 cv_a_33;
    operator3 cv_a_34;
    torch::nn::Conv2d cv_a_35;
    operator3 cv_a_36;
    torch::nn::Conv2d cv_a_37;
    operator3 cv_a_38;
    operator3 cv_a_39;
    operator3 cv_a_40;
    operator3 cv_a_41;
    operator3 cv_a_42;
    torch::nn::Conv2d cv_a_43;
    operator3 cv_a_44;
    operator3 cv_a_45;
    operator3 cv_a_46;
    operator3 cv_a_47;
    operator3 cv_a_48;
    torch::nn::Conv2d cv_a_49;
    operator3 cv_a_50;
    operator3 cv_a_51;
    operator3 cv_a_52;
    operator3 cv_a_53;
    torch::nn::Conv2d cv_a_54;
    operator3 cv_a_55;
    operator3 cv_a_56;
    operator3 cv_a_57;
    operator2 cv_b_0;
    operator2 cv_b_1;
    operator2 cv_b_2;
    operator2 cv_b_3;
    operator2 cv_b_4;
    operator2 cv_b_5;
    torch::nn::Conv2d cv_b_6;
    operator2 cv_b_7;
    torch::nn::Conv2d cv_b_8;
    operator2 cv_b_9;
    operator2 cv_b_10;
    operator2 cv_b_11;
    operator2 cv_b_12;
    operator2 cv_b_13;
    operator2 cv_b_14;
    operator2 cv_b_15;
    operator2 cv_b_16;
    operator2 cv_b_17;
    operator2 cv_b_18;
    torch::nn::Conv2d cv_b_19;
    operator2 cv_b_20;
    operator2 cv_b_21;
    operator2 cv_b_22;
    operator2 cv_b_23;
    operator2 cv_b_24;
    torch::nn::Conv2d cv_b_25;
    operator2 cv_b_26;
    operator2 cv_b_27;
    operator2 cv_b_28;
    operator2 cv_b_29;
    operator2 cv_b_30;
    operator2 cv_b_31;
    torch::nn::Conv2d cv_b_32;
    operator2 cv_b_33;
    operator2 cv_b_34;
    operator2 cv_b_35;
    operator2 cv_b_36;
    torch::nn::Conv2d cv_b_37;
    operator2 cv_b_38;
    operator2 cv_b_39;
    operator2 cv_b_40;
    operator2 cv_b_41;
    operator2 cv_b_42;
    torch::nn::Conv2d cv_b_43;
    operator2 cv_b_44;
    operator2 cv_b_45;
    operator2 cv_b_46;
    operator2 cv_b_47;
    operator2 cv_b_48;
    operator2 cv_b_49;
    operator2 cv_b_50;
    torch::nn::Conv2d cv_b_51;
    operator2 cv_b_52;
    torch::nn::Conv2d cv_b_53;
    operator2 cv_b_54;
    torch::nn::Conv2d cv_b_55;
    torch::nn::Conv2d cv_b_56;
    operator2 cv_b_57;

    //torch::nn::BatchNorm2d b1;
    operator3 c1;
    //torch::nn::BatchNorm2d b2;
    operator3 c2;
    operator5 avg0;
    //torch::nn::BatchNorm2d b3;
    torch::nn::Conv2d c3;
    //torch::nn::BatchNorm2d b4;
    operator6 admaxp2d0;
    operator7 linear0;
    operator5 avg1;

    dense_warmup():
        c0(torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 7).padding(3).stride(2))),
        //b0(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(64))),
        mxp2d0(3, 1, 2),
        cv_a_0(64, 128, 1, 1),
        cv_a_1(torch::nn::Conv2dOptions(96, 128, 1).stride(1)),
        cv_a_2(128, 128, 1, 1),
        cv_a_3(160, 128, 1, 1),
        cv_a_4(192, 128, 1, 1),
        cv_a_5(224, 128, 1, 1),
        cv_a_6(128, 128, 1, 1),
        cv_a_7(160, 128, 1, 1),
        cv_a_8(torch::nn::Conv2dOptions(192, 128, 1).stride(1)),
        cv_a_9(224, 128, 1, 1),
        cv_a_10(256, 128, 1, 1),
        cv_a_11(288, 128, 1, 1),
        cv_a_12(320, 128, 1, 1),
        cv_a_13(352, 128, 1, 1),
        cv_a_14(384, 128, 1, 1),
        cv_a_15(416, 128, 1, 1),
        cv_a_16(448, 128, 1, 1),
        cv_a_17(torch::nn::Conv2dOptions(480, 128, 1).stride(1)),
        cv_a_18(256, 128, 1, 1),
        cv_a_19(288, 128, 1, 1),
        cv_a_20(320, 128, 1, 1),
        cv_a_21(352, 128, 1, 1),
        cv_a_22(384, 128, 1, 1),
        cv_a_23(416, 128, 1, 1),
        cv_a_24(448, 128, 1, 1),
        cv_a_25(480, 128, 1, 1),
        cv_a_26(512, 128, 1, 1),
        cv_a_27(544, 128, 1, 1),
        cv_a_28(576, 128, 1, 1),
        cv_a_29(608, 128, 1, 1),
        cv_a_30(640, 128, 1, 1),
        cv_a_31(672, 128, 1, 1),
        cv_a_32(704, 128, 1, 1),
        cv_a_33(736, 128, 1, 1),
        cv_a_34(768, 128, 1, 1),
        cv_a_35(torch::nn::Conv2dOptions(800, 128, 1).stride(1)),
        cv_a_36(832, 128, 1, 1),
        cv_a_37(torch::nn::Conv2dOptions(864, 128, 1).stride(1)),
        cv_a_38(896, 128, 1, 1),
        cv_a_39(928, 128, 1, 1),
        cv_a_40(960, 128, 1, 1),
        cv_a_41(992, 128, 1, 1),
        cv_a_42(512, 128, 1, 1),
        cv_a_43(torch::nn::Conv2dOptions(544, 128, 1).stride(1)),
        cv_a_44(576, 128, 1, 1),
        cv_a_45(608, 128, 1, 1),
        cv_a_46(640, 128, 1, 1),
        cv_a_47(672, 128, 1, 1),
        cv_a_48(704, 128, 1, 1),
        cv_a_49(torch::nn::Conv2dOptions(736, 128, 1).stride(1)),
        cv_a_50(768, 128, 1, 1),
        cv_a_51(800, 128, 1, 1),
        cv_a_52(832, 128, 1, 1),
        cv_a_53(864, 128, 1, 1),
        cv_a_54(torch::nn::Conv2dOptions(896, 128, 1).stride(1)),
        cv_a_55(928, 128, 1, 1),
        cv_a_56(960, 128, 1, 1),
        cv_a_57(992, 128, 1, 1),
        cv_b_0(128, 32, 3, 1, 1),
        cv_b_1(128, 32, 3, 1, 1),
        cv_b_2(128, 32, 3, 1, 1),
        cv_b_3(128, 32, 3, 1, 1),
        cv_b_4(128, 32, 3, 1, 1),
        cv_b_5(128, 32, 3, 1, 1),
        cv_b_6(torch::nn::Conv2dOptions(128, 32, 3).padding(1).stride(1)),
        cv_b_7(128, 32, 3, 1, 1),
        cv_b_8(torch::nn::Conv2dOptions(128, 32, 3).padding(1).stride(1)),
        cv_b_9(128, 32, 3, 1, 1),
        cv_b_10(128, 32, 3, 1, 1),
        cv_b_11(128, 32, 3, 1, 1),
        cv_b_12(128, 32, 3, 1, 1),
        cv_b_13(128, 32, 3, 1, 1),
        cv_b_14(128, 32, 3, 1, 1),
        cv_b_15(128, 32, 3, 1, 1),
        cv_b_16(128, 32, 3, 1, 1),
        cv_b_17(128, 32, 3, 1, 1),
        cv_b_18(128, 32, 3, 1, 1),
        cv_b_19(torch::nn::Conv2dOptions(128, 32, 3).padding(1).stride(1)),
        cv_b_20(128, 32, 3, 1, 1),
        cv_b_21(128, 32, 3, 1, 1),
        cv_b_22(128, 32, 3, 1, 1),
        cv_b_23(128, 32, 3, 1, 1),
        cv_b_24(128, 32, 3, 1, 1),
        cv_b_25(torch::nn::Conv2dOptions(128, 32, 3).padding(1).stride(1)),
        cv_b_26(128, 32, 3, 1, 1),
        cv_b_27(128, 32, 3, 1, 1),
        cv_b_28(128, 32, 3, 1, 1),
        cv_b_29(128, 32, 3, 1, 1),
        cv_b_30(128, 32, 3, 1, 1),
        cv_b_31(128, 32, 3, 1, 1),
        cv_b_32(torch::nn::Conv2dOptions(128, 32, 3).padding(1).stride(1)),
        cv_b_33(128, 32, 3, 1, 1),
        cv_b_34(128, 32, 3, 1, 1),
        cv_b_35(128, 32, 3, 1, 1),
        cv_b_36(128, 32, 3, 1, 1),
        cv_b_37(torch::nn::Conv2dOptions(128, 32, 3).padding(1).stride(1)),
        cv_b_38(128, 32, 3, 1, 1),
        cv_b_39(128, 32, 3, 1, 1),
        cv_b_40(128, 32, 3, 1, 1),
        cv_b_41(128, 32, 3, 1, 1),
        cv_b_42(128, 32, 3, 1, 1),
        cv_b_43(torch::nn::Conv2dOptions(128, 32, 3).padding(1).stride(1)),
        cv_b_44(128, 32, 3, 1, 1),
        cv_b_45(128, 32, 3, 1, 1),
        cv_b_46(128, 32, 3, 1, 1),
        cv_b_47(128, 32, 3, 1, 1),
        cv_b_48(128, 32, 3, 1, 1),
        cv_b_49(128, 32, 3, 1, 1),
        cv_b_50(128, 32, 3, 1, 1),
        cv_b_51(torch::nn::Conv2dOptions(128, 32, 3).padding(1).stride(1)),
        cv_b_52(128, 32, 3, 1, 1),
        cv_b_53(torch::nn::Conv2dOptions(128, 32, 3).padding(1).stride(1)),
        cv_b_54(128, 32, 3, 1, 1),
        cv_b_55(torch::nn::Conv2dOptions(128, 32, 3).padding(1).stride(1)),
        cv_b_56(torch::nn::Conv2dOptions(128, 32, 3).padding(1).stride(1)),
        cv_b_57(128, 32, 3, 1, 1),
        c1(256, 128, 1, 1),
        c2(512,256, 1, 1),
        avg0(2, 2),
        //b3(1024),
        c3(torch::nn::Conv2d(torch::nn::Conv2dOptions(1024, 512, 1).stride(1))),
        //b4(1024),
        admaxp2d0(1, 1),
        linear0(1024, 1000),
        avg1(2, 2)
        {
            c0->to(at::kCUDA);
            mxp2d0.to(at::kCUDA);
            relu.to(at::kCUDA);
            cv_a_0.to(at::kCUDA);
            cv_a_1->to(at::kCUDA);
            cv_a_2.to(at::kCUDA);
            cv_a_3.to(at::kCUDA);
            cv_a_4.to(at::kCUDA);
            cv_a_5.to(at::kCUDA);
            cv_a_6.to(at::kCUDA);
            cv_a_7.to(at::kCUDA);
            cv_a_8->to(at::kCUDA);
            cv_a_9.to(at::kCUDA);
            cv_a_10.to(at::kCUDA);
            cv_a_11.to(at::kCUDA);
            cv_a_12.to(at::kCUDA);
            cv_a_13.to(at::kCUDA);
            cv_a_14.to(at::kCUDA);
            cv_a_15.to(at::kCUDA);
            cv_a_16.to(at::kCUDA);
            cv_a_17->to(at::kCUDA);
            cv_a_18.to(at::kCUDA);
            cv_a_19.to(at::kCUDA);
            cv_a_20.to(at::kCUDA);
            cv_a_21.to(at::kCUDA);
            cv_a_22.to(at::kCUDA);
            cv_a_23.to(at::kCUDA);
            cv_a_24.to(at::kCUDA);
            cv_a_25.to(at::kCUDA);
            cv_a_26.to(at::kCUDA);
            cv_a_27.to(at::kCUDA);
            cv_a_28.to(at::kCUDA);
            cv_a_29.to(at::kCUDA);
            cv_a_30.to(at::kCUDA);
            cv_a_31.to(at::kCUDA);
            cv_a_32.to(at::kCUDA);
            cv_a_33.to(at::kCUDA);
            cv_a_34.to(at::kCUDA);
            cv_a_35->to(at::kCUDA);
            cv_a_36.to(at::kCUDA);
            cv_a_37->to(at::kCUDA);
            cv_a_38.to(at::kCUDA);
            cv_a_39.to(at::kCUDA);
            cv_a_40.to(at::kCUDA);
            cv_a_41.to(at::kCUDA);
            cv_a_42.to(at::kCUDA);
            cv_a_43->to(at::kCUDA);
            cv_a_44.to(at::kCUDA);
            cv_a_45.to(at::kCUDA);
            cv_a_46.to(at::kCUDA);
            cv_a_47.to(at::kCUDA);
            cv_a_48.to(at::kCUDA);
            cv_a_49->to(at::kCUDA);
            cv_a_50.to(at::kCUDA);
            cv_a_51.to(at::kCUDA);
            cv_a_52.to(at::kCUDA);
            cv_a_53.to(at::kCUDA);
            cv_a_54->to(at::kCUDA);
            cv_a_55.to(at::kCUDA);
            cv_a_56.to(at::kCUDA);
            cv_a_57.to(at::kCUDA);
            cv_b_0.to(at::kCUDA);
            cv_b_1.to(at::kCUDA);
            cv_b_2.to(at::kCUDA);
            cv_b_3.to(at::kCUDA);
            cv_b_4.to(at::kCUDA);
            cv_b_5.to(at::kCUDA);
            cv_b_6->to(at::kCUDA);
            cv_b_7.to(at::kCUDA);
            cv_b_8->to(at::kCUDA);
            cv_b_9.to(at::kCUDA);
            cv_b_10.to(at::kCUDA);
            cv_b_11.to(at::kCUDA);
            cv_b_12.to(at::kCUDA);
            cv_b_13.to(at::kCUDA);
            cv_b_14.to(at::kCUDA);
            cv_b_15.to(at::kCUDA);
            cv_b_16.to(at::kCUDA);
            cv_b_17.to(at::kCUDA);
            cv_b_18.to(at::kCUDA);
            cv_b_19->to(at::kCUDA);
            cv_b_20.to(at::kCUDA);
            cv_b_21.to(at::kCUDA);
            cv_b_22.to(at::kCUDA);
            cv_b_23.to(at::kCUDA);
            cv_b_24.to(at::kCUDA);
            cv_b_25->to(at::kCUDA);
            cv_b_26.to(at::kCUDA);
            cv_b_27.to(at::kCUDA);
            cv_b_28.to(at::kCUDA);
            cv_b_29.to(at::kCUDA);
            cv_b_30.to(at::kCUDA);
            cv_b_31.to(at::kCUDA);
            cv_b_32->to(at::kCUDA);
            cv_b_33.to(at::kCUDA);
            cv_b_34.to(at::kCUDA);
            cv_b_35.to(at::kCUDA);
            cv_b_36.to(at::kCUDA);
            cv_b_37->to(at::kCUDA);
            cv_b_38.to(at::kCUDA);
            cv_b_39.to(at::kCUDA);
            cv_b_40.to(at::kCUDA);
            cv_b_41.to(at::kCUDA);
            cv_b_42.to(at::kCUDA);
            cv_b_43->to(at::kCUDA);
            cv_b_44.to(at::kCUDA);
            cv_b_45.to(at::kCUDA);
            cv_b_46.to(at::kCUDA);
            cv_b_47.to(at::kCUDA);
            cv_b_48.to(at::kCUDA);
            cv_b_49.to(at::kCUDA);
            cv_b_50.to(at::kCUDA);
            cv_b_51->to(at::kCUDA);
            cv_b_52.to(at::kCUDA);
            cv_b_53->to(at::kCUDA);
            cv_b_54.to(at::kCUDA);
            cv_b_55->to(at::kCUDA);
            cv_b_56->to(at::kCUDA);
            cv_b_57.to(at::kCUDA);
            c1.to(at::kCUDA);
            c2.to(at::kCUDA);
            avg0.to(at::kCUDA);
            c3->to(at::kCUDA);
            admaxp2d0.to(at::kCUDA);
            linear0.to(at::kCUDA);
        }
    
    torch::Tensor forward(torch::Tensor x)
    {
        torch::Tensor x_before;

        x = c0(x);
        x = relu.forward(x);
        x = mxp2d0.forward(x);

        // block 1
        x_before = x;
        x = relu.forward(x);
        x = cv_a_0.forward(x);
        x = relu.forward(x);
        x = cv_b_0.forward(x);
        x = torch::cat({x_before, x}, 1);
        
        x_before = x;
        x = relu.forward(x);
        x = cv_a_1(x);
        x = relu.forward(x);
        x = cv_b_1.forward(x);
        x = torch::cat({x_before, x}, 1);

        x_before = x;
        x = relu.forward(x);
        x = cv_a_2.forward(x);
        x = relu.forward(x);
        x = cv_b_2.forward(x);
        x = torch::cat({x_before, x}, 1);

        x_before = x;
        x = relu.forward(x);
        x = cv_a_3.forward(x);
        x = relu.forward(x);
        x = cv_b_3.forward(x);
        x = torch::cat({x_before, x}, 1);

        x_before = x;
        x = relu.forward(x);
        x = cv_a_4.forward(x);
        x = relu.forward(x);
        x = cv_b_4.forward(x);
        x = torch::cat({x_before, x}, 1);

        x_before = x;
        x = relu.forward(x);
        x = cv_a_5.forward(x);
        x = relu.forward(x);
        x = cv_b_5.forward(x);
        x = torch::cat({x_before, x}, 1);
        x = relu.forward(x);
        x = c1.forward(x);
        x = avg1.forward(x);

        // block 2
        x_before = x;
        x = relu.forward(x);
        x = cv_a_6.forward(x);
        x = relu.forward(x);
        x = cv_b_6(x);
        x = torch::cat({x_before, x}, 1);

        x_before = x;
        x = relu.forward(x);
        x = cv_a_7.forward(x);
        x = relu.forward(x);
        x = cv_b_7.forward(x);
        x = torch::cat({x_before, x}, 1);

        x_before = x;
        x = relu.forward(x);
        x = cv_a_8(x);
        x = relu.forward(x);
        x = cv_b_8(x);
        x = torch::cat({x_before, x}, 1);

        x_before = x;
        x = relu.forward(x);
        x = cv_a_9.forward(x);
        x = relu.forward(x);
        x = cv_b_9.forward(x);
        x = torch::cat({x_before, x}, 1);

        x_before = x;
        x = relu.forward(x);
        x = cv_a_10.forward(x);
        x = relu.forward(x);
        x = cv_b_10.forward(x);
        x = torch::cat({x_before, x}, 1);

        x_before = x;
        x = relu.forward(x);
        x = cv_a_11.forward(x);
        x = relu.forward(x);
        x = cv_b_11.forward(x);
        x = torch::cat({x_before, x}, 1);

        x_before = x;
        x = relu.forward(x);
        x = cv_a_12.forward(x);
        x = relu.forward(x);
        x = cv_b_12.forward(x);
        x = torch::cat({x_before, x}, 1);

        x_before = x;
        x = relu.forward(x);
        x = cv_a_13.forward(x);
        x = relu.forward(x);
        x = cv_b_13.forward(x);
        x = torch::cat({x_before, x}, 1);

        x_before = x;
        x = relu.forward(x);
        x = cv_a_14.forward(x);
        x = relu.forward(x);
        x = cv_b_14.forward(x);
        x = torch::cat({x_before, x}, 1);

        x_before = x;
        x = relu.forward(x);
        x = cv_a_15.forward(x);
        x = relu.forward(x);
        x = cv_b_15.forward(x);
        x = torch::cat({x_before, x}, 1);

        x_before = x;
        x = relu.forward(x);
        x = cv_a_16.forward(x);
        x = relu.forward(x);
        x = cv_b_16.forward(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = relu.forward(x);
        x = cv_a_17(x);
        x = relu.forward(x);
        x = cv_b_17.forward(x);
        x = torch::cat({x_before, x}, 1);

        // transition 2
        x = relu.forward(x);
        x = c2.forward(x);
        x = avg0.forward(x);


        // block 3
        x_before = x;
        x = relu.forward(x);
        x = cv_a_18.forward(x);
        x = relu.forward(x);
        x = cv_b_18.forward(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = relu.forward(x);
        x = cv_a_19.forward(x);
        x = relu.forward(x);
        x = cv_b_19(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = relu.forward(x);
        x = cv_a_20.forward(x);
        x = relu.forward(x);
        x = cv_b_20.forward(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = relu.forward(x);
        x = cv_a_21.forward(x);
        x = relu.forward(x);
        x = cv_b_21.forward(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = relu.forward(x);
        x = cv_a_22.forward(x);
        x = relu.forward(x);
        x = cv_b_22.forward(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = relu.forward(x);
        x = cv_a_23.forward(x);
        x = relu.forward(x);
        x = cv_b_23.forward(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = relu.forward(x);
        x = cv_a_24.forward(x);
        x = relu.forward(x);
        x = cv_b_24.forward(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = relu.forward(x);
        x = cv_a_25.forward(x);
        x = relu.forward(x);
        x = cv_b_25(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = relu.forward(x);
        x = cv_a_26.forward(x);
        x = relu.forward(x);
        x = cv_b_26.forward(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = relu.forward(x);
        x = cv_a_27.forward(x);
        x = relu.forward(x);
        x = cv_b_27.forward(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = relu.forward(x);
        x = cv_a_28.forward(x);
        x = relu.forward(x);
        x = cv_b_28.forward(x);
        x = torch::cat({x_before, x}, 1);

        x_before = x;
        x = relu.forward(x);
        x = cv_a_29.forward(x);
        x = relu.forward(x);
        x = cv_b_29.forward(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = relu.forward(x);
        x = cv_a_30.forward(x);
        x = relu.forward(x);
        x = cv_b_30.forward(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = relu.forward(x);
        x = cv_a_31.forward(x);
        x = relu.forward(x);
        x = cv_b_31.forward(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = relu.forward(x);
        x = cv_a_32.forward(x);
        x = relu.forward(x);
        x = cv_b_32(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = relu.forward(x);
        x = cv_a_33.forward(x);
        x = relu.forward(x);
        x = cv_b_33.forward(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = relu.forward(x);
        x = cv_a_34.forward(x);
        x = relu.forward(x);
        x = cv_b_34.forward(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = relu.forward(x);
        x = cv_a_35(x);
        x = relu.forward(x);
        x = cv_b_35.forward(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = relu.forward(x);
        x = cv_a_36.forward(x);
        x = relu.forward(x);
        x = cv_b_36.forward(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = relu.forward(x);
        x = cv_a_37(x);
        x = relu.forward(x);
        x = cv_b_37(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = relu.forward(x);
        x = cv_a_38.forward(x);
        x = relu.forward(x);
        x = cv_b_38.forward(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = relu.forward(x);
        x = cv_a_39.forward(x);
        x = relu.forward(x);
        x = cv_b_39.forward(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = relu.forward(x);
        x = cv_a_40.forward(x);
        x = relu.forward(x);
        x = cv_b_40.forward(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = relu.forward(x);
        x = cv_a_41.forward(x);
        x = relu.forward(x);
        x = cv_b_41.forward(x);
        x = torch::cat({x_before, x}, 1);

        //x = b3(x);
        x = relu.forward(x);
        x = c3(x);
        x = avg0.forward(x);

        //block 4
        x_before = x;
        x = relu.forward(x);
        x = cv_a_42.forward(x);
        x = relu.forward(x);
        x = cv_b_42.forward(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = relu.forward(x);
        x = cv_a_43(x);
        x = relu.forward(x);
        x = cv_b_43(x);
        x = torch::cat({x_before, x}, 1);

        x_before = x;
        x = relu.forward(x);
        x = cv_a_44.forward(x);
        x = relu.forward(x);
        x = cv_b_44.forward(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = relu.forward(x);
        x = cv_a_45.forward(x);
        x = relu.forward(x);
        x = cv_b_45.forward(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = relu.forward(x);
        x = cv_a_46.forward(x);
        x = relu.forward(x);
        x = cv_b_46.forward(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = relu.forward(x);
        x = cv_a_47.forward(x);
        x = relu.forward(x);
        x = cv_b_47.forward(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = relu.forward(x);
        x = cv_a_48.forward(x);
        x = relu.forward(x);
        x = cv_b_48.forward(x);
        x = torch::cat({x_before, x}, 1);

        x_before = x;
        x = relu.forward(x);
        x = cv_a_49(x);
        x = relu.forward(x);
        x = cv_b_49.forward(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = relu.forward(x);
        x = cv_a_50.forward(x);
        x = relu.forward(x);
        x = cv_b_50.forward(x);
        x = torch::cat({x_before, x}, 1);

        x_before = x;
        x = relu.forward(x);
        x = cv_a_51.forward(x);
        x = relu.forward(x);
        x = cv_b_51(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = relu.forward(x);
        x = cv_a_52.forward(x);
        x = relu.forward(x);
        x = cv_b_52.forward(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = relu.forward(x);
        x = cv_a_53.forward(x);
        x = relu.forward(x);
        x = cv_b_53(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = relu.forward(x);
        x = cv_a_54(x);
        x = relu.forward(x);
        x = cv_b_54.forward(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = relu.forward(x);
        x = cv_a_55.forward(x);
        x = relu.forward(x);
        x = cv_b_55(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = relu.forward(x);
        x = cv_a_56.forward(x);
        x = relu.forward(x);
        x = cv_b_56(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = relu.forward(x);
        x = cv_a_57.forward(x);
        x = relu.forward(x);
        x = cv_b_57.forward(x);
        x = torch::cat({x_before, x}, 1);

        //end
        x = relu.forward(x);
        x = admaxp2d0.forward(x);
        x = x.view({x.sizes()[0], -1});
        x = linear0.forward(x);
        return x;
    }
    void morphpara(){
        // nothing to do, inherit from virtual func
    }
};

class dense_gpu_part1 : public dense_part
{
public:
    torch::nn::Conv2d c0;
    //torch::nn::BatchNorm2d b0;
    operator4 mxp2d0;
    operator1 relu;

    operator3 cv_a_0;
    torch::nn::Conv2d cv_a_1;
    operator3 cv_a_2;
    operator3 cv_a_3;
    operator3 cv_a_4;
    operator3 cv_a_5;
    operator3 cv_a_6;
    operator3 cv_a_7;
    torch::nn::Conv2d cv_a_8;
    operator3 cv_a_9;
    operator3 cv_a_10;
    operator3 cv_a_11;
    operator3 cv_a_12;
    operator3 cv_a_13;
    operator3 cv_a_14;
    operator3 cv_a_15;
    operator3 cv_a_16;
    torch::nn::Conv2d cv_a_17;
    operator3 cv_a_18;
    operator3 cv_a_19;
    operator3 cv_a_20;
    operator3 cv_a_21;
    operator3 cv_a_22;
    operator3 cv_a_23;
    operator3 cv_a_24;
    operator3 cv_a_25;
    operator3 cv_a_26;
    operator3 cv_a_27;
    operator3 cv_a_28;
    operator3 cv_a_29;
    operator3 cv_a_30;
    operator3 cv_a_31;
    operator3 cv_a_32;
    operator3 cv_a_33;
    operator3 cv_a_34;
    torch::nn::Conv2d cv_a_35;
    operator3 cv_a_36;
    torch::nn::Conv2d cv_a_37;
    operator3 cv_a_38;
    operator3 cv_a_39;
    operator3 cv_a_40;
    operator3 cv_a_41;
    operator3 cv_a_42;
    torch::nn::Conv2d cv_a_43;
    operator3 cv_a_44;
    operator3 cv_a_45;
    operator3 cv_a_46;
    operator3 cv_a_47;
    operator3 cv_a_48;
    torch::nn::Conv2d cv_a_49;
    operator3 cv_a_50;
    operator3 cv_a_51;
    operator3 cv_a_52;
    operator3 cv_a_53;
    torch::nn::Conv2d cv_a_54;
    operator3 cv_a_55;
    operator3 cv_a_56;
    operator3 cv_a_57;
    operator2 cv_b_0;
    operator2 cv_b_1;
    operator2 cv_b_2;
    operator2 cv_b_3;
    operator2 cv_b_4;
    operator2 cv_b_5;
    torch::nn::Conv2d cv_b_6;
    operator2 cv_b_7;
    torch::nn::Conv2d cv_b_8;
    operator2 cv_b_9;
    operator2 cv_b_10;
    operator2 cv_b_11;
    operator2 cv_b_12;
    operator2 cv_b_13;
    operator2 cv_b_14;
    operator2 cv_b_15;
    operator2 cv_b_16;
    operator2 cv_b_17;
    operator2 cv_b_18;
    torch::nn::Conv2d cv_b_19;
    operator2 cv_b_20;
    operator2 cv_b_21;
    operator2 cv_b_22;
    operator2 cv_b_23;
    operator2 cv_b_24;
    torch::nn::Conv2d cv_b_25;
    operator2 cv_b_26;
    operator2 cv_b_27;
    operator2 cv_b_28;
    operator2 cv_b_29;
    operator2 cv_b_30;
    operator2 cv_b_31;
    torch::nn::Conv2d cv_b_32;
    operator2 cv_b_33;
    operator2 cv_b_34;
    operator2 cv_b_35;
    operator2 cv_b_36;
    torch::nn::Conv2d cv_b_37;
    operator2 cv_b_38;
    operator2 cv_b_39;
    operator2 cv_b_40;
    operator2 cv_b_41;
    operator2 cv_b_42;
    torch::nn::Conv2d cv_b_43;
    operator2 cv_b_44;
    operator2 cv_b_45;
    operator2 cv_b_46;
    operator2 cv_b_47;
    operator2 cv_b_48;
    operator2 cv_b_49;
    operator2 cv_b_50;
    torch::nn::Conv2d cv_b_51;
    operator2 cv_b_52;
    torch::nn::Conv2d cv_b_53;
    operator2 cv_b_54;
    torch::nn::Conv2d cv_b_55;
    torch::nn::Conv2d cv_b_56;
    operator2 cv_b_57;

    //torch::nn::BatchNorm2d b1;
    operator3 c1;
    //torch::nn::BatchNorm2d b2;
    operator3 c2;
    operator5 avg0;
    //torch::nn::BatchNorm2d b3;
    torch::nn::Conv2d c3;
    //torch::nn::BatchNorm2d b4;
    operator6 admaxp2d0;
    operator7 linear0;
    operator5 avg1;

    dense_gpu_part1():
        c0(torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 7).padding(3).stride(2))),
        //b0(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(64))),
        mxp2d0(3, 1, 2),
        cv_a_0(64, 128, 1, 1),
        cv_a_1(torch::nn::Conv2dOptions(96, 128, 1).stride(1)),
        cv_a_2(128, 128, 1, 1),
        cv_a_3(160, 128, 1, 1),
        cv_a_4(192, 128, 1, 1),
        cv_a_5(224, 128, 1, 1),
        cv_a_6(128, 128, 1, 1),
        cv_a_7(160, 128, 1, 1),
        cv_a_8(torch::nn::Conv2dOptions(192, 128, 1).stride(1)),
        cv_a_9(224, 128, 1, 1),
        cv_a_10(256, 128, 1, 1),
        cv_a_11(288, 128, 1, 1),
        cv_a_12(320, 128, 1, 1),
        cv_a_13(352, 128, 1, 1),
        cv_a_14(384, 128, 1, 1),
        cv_a_15(416, 128, 1, 1),
        cv_a_16(448, 128, 1, 1),
        cv_a_17(torch::nn::Conv2dOptions(480, 128, 1).stride(1)),
        cv_a_18(256, 128, 1, 1),
        cv_a_19(288, 128, 1, 1),
        cv_a_20(320, 128, 1, 1),
        cv_a_21(352, 128, 1, 1),
        cv_a_22(384, 128, 1, 1),
        cv_a_23(416, 128, 1, 1),
        cv_a_24(448, 128, 1, 1),
        cv_a_25(480, 128, 1, 1),
        cv_a_26(512, 128, 1, 1),
        cv_a_27(544, 128, 1, 1),
        cv_a_28(576, 128, 1, 1),
        cv_a_29(608, 128, 1, 1),
        cv_a_30(640, 128, 1, 1),
        cv_a_31(672, 128, 1, 1),
        cv_a_32(704, 128, 1, 1),
        cv_a_33(736, 128, 1, 1),
        cv_a_34(768, 128, 1, 1),
        cv_a_35(torch::nn::Conv2dOptions(800, 128, 1).stride(1)),
        cv_a_36(832, 128, 1, 1),
        cv_a_37(torch::nn::Conv2dOptions(864, 128, 1).stride(1)),
        cv_a_38(896, 128, 1, 1),
        cv_a_39(928, 128, 1, 1),
        cv_a_40(960, 128, 1, 1),
        cv_a_41(992, 128, 1, 1),
        cv_a_42(512, 128, 1, 1),
        cv_a_43(torch::nn::Conv2dOptions(544, 128, 1).stride(1)),
        cv_a_44(576, 128, 1, 1),
        cv_a_45(608, 128, 1, 1),
        cv_a_46(640, 128, 1, 1),
        cv_a_47(672, 128, 1, 1),
        cv_a_48(704, 128, 1, 1),
        cv_a_49(torch::nn::Conv2dOptions(736, 128, 1).stride(1)),
        cv_a_50(768, 128, 1, 1),
        cv_a_51(800, 128, 1, 1),
        cv_a_52(832, 128, 1, 1),
        cv_a_53(864, 128, 1, 1),
        cv_a_54(torch::nn::Conv2dOptions(896, 128, 1).stride(1)),
        cv_a_55(928, 128, 1, 1),
        cv_a_56(960, 128, 1, 1),
        cv_a_57(992, 128, 1, 1),
        cv_b_0(128, 32, 3, 1, 1),
        cv_b_1(128, 32, 3, 1, 1),
        cv_b_2(128, 32, 3, 1, 1),
        cv_b_3(128, 32, 3, 1, 1),
        cv_b_4(128, 32, 3, 1, 1),
        cv_b_5(128, 32, 3, 1, 1),
        cv_b_6(torch::nn::Conv2dOptions(128, 32, 3).padding(1).stride(1)),
        cv_b_7(128, 32, 3, 1, 1),
        cv_b_8(torch::nn::Conv2dOptions(128, 32, 3).padding(1).stride(1)),
        cv_b_9(128, 32, 3, 1, 1),
        cv_b_10(128, 32, 3, 1, 1),
        cv_b_11(128, 32, 3, 1, 1),
        cv_b_12(128, 32, 3, 1, 1),
        cv_b_13(128, 32, 3, 1, 1),
        cv_b_14(128, 32, 3, 1, 1),
        cv_b_15(128, 32, 3, 1, 1),
        cv_b_16(128, 32, 3, 1, 1),
        cv_b_17(128, 32, 3, 1, 1),
        cv_b_18(128, 32, 3, 1, 1),
        cv_b_19(torch::nn::Conv2dOptions(128, 32, 3).padding(1).stride(1)),
        cv_b_20(128, 32, 3, 1, 1),
        cv_b_21(128, 32, 3, 1, 1),
        cv_b_22(128, 32, 3, 1, 1),
        cv_b_23(128, 32, 3, 1, 1),
        cv_b_24(128, 32, 3, 1, 1),
        cv_b_25(torch::nn::Conv2dOptions(128, 32, 3).padding(1).stride(1)),
        cv_b_26(128, 32, 3, 1, 1),
        cv_b_27(128, 32, 3, 1, 1),
        cv_b_28(128, 32, 3, 1, 1),
        cv_b_29(128, 32, 3, 1, 1),
        cv_b_30(128, 32, 3, 1, 1),
        cv_b_31(128, 32, 3, 1, 1),
        cv_b_32(torch::nn::Conv2dOptions(128, 32, 3).padding(1).stride(1)),
        cv_b_33(128, 32, 3, 1, 1),
        cv_b_34(128, 32, 3, 1, 1),
        cv_b_35(128, 32, 3, 1, 1),
        cv_b_36(128, 32, 3, 1, 1),
        cv_b_37(torch::nn::Conv2dOptions(128, 32, 3).padding(1).stride(1)),
        cv_b_38(128, 32, 3, 1, 1),
        cv_b_39(128, 32, 3, 1, 1),
        cv_b_40(128, 32, 3, 1, 1),
        cv_b_41(128, 32, 3, 1, 1),
        cv_b_42(128, 32, 3, 1, 1),
        cv_b_43(torch::nn::Conv2dOptions(128, 32, 3).padding(1).stride(1)),
        cv_b_44(128, 32, 3, 1, 1),
        cv_b_45(128, 32, 3, 1, 1),
        cv_b_46(128, 32, 3, 1, 1),
        cv_b_47(128, 32, 3, 1, 1),
        cv_b_48(128, 32, 3, 1, 1),
        cv_b_49(128, 32, 3, 1, 1),
        cv_b_50(128, 32, 3, 1, 1),
        cv_b_51(torch::nn::Conv2dOptions(128, 32, 3).padding(1).stride(1)),
        cv_b_52(128, 32, 3, 1, 1),
        cv_b_53(torch::nn::Conv2dOptions(128, 32, 3).padding(1).stride(1)),
        cv_b_54(128, 32, 3, 1, 1),
        cv_b_55(torch::nn::Conv2dOptions(128, 32, 3).padding(1).stride(1)),
        cv_b_56(torch::nn::Conv2dOptions(128, 32, 3).padding(1).stride(1)),
        cv_b_57(128, 32, 3, 1, 1),
        c1(256, 128, 1, 1),
        c2(512,256, 1, 1),
        avg0(2, 2),
        //b3(1024),
        c3(torch::nn::Conv2d(torch::nn::Conv2dOptions(1024, 512, 1).stride(1))),
        //b4(1024),
        admaxp2d0(1, 1),
        linear0(1024, 1000),
        avg1(2, 2)
        {
            c0->to(at::kCUDA);
            mxp2d0.to(at::kCUDA);
            relu.to(at::kCUDA);
            cv_a_0.to(at::kCUDA);
            cv_a_1->to(at::kCUDA);
            cv_a_2.to(at::kCUDA);
            cv_a_3.to(at::kCUDA);
            cv_a_4.to(at::kCUDA);
            cv_a_5.to(at::kCUDA);
            cv_a_6.to(at::kCUDA);
            cv_a_7.to(at::kCUDA);
            cv_a_8->to(at::kCUDA);
            cv_a_9.to(at::kCUDA);
            cv_a_10.to(at::kCUDA);
            cv_a_11.to(at::kCUDA);
            cv_a_12.to(at::kCUDA);
            cv_a_13.to(at::kCUDA);
            cv_a_14.to(at::kCUDA);
            cv_a_15.to(at::kCUDA);
            cv_a_16.to(at::kCUDA);
            cv_a_17->to(at::kCUDA);
            cv_a_18.to(at::kCUDA);
            cv_a_19.to(at::kCUDA);
            cv_a_20.to(at::kCUDA);
            cv_a_21.to(at::kCUDA);
            cv_a_22.to(at::kCUDA);
            cv_a_23.to(at::kCUDA);
            cv_a_24.to(at::kCUDA);
            cv_a_25.to(at::kCUDA);
            cv_a_26.to(at::kCUDA);
            cv_a_27.to(at::kCUDA);
            cv_a_28.to(at::kCUDA);
            cv_a_29.to(at::kCUDA);
            cv_a_30.to(at::kCUDA);
            cv_a_31.to(at::kCUDA);
            cv_a_32.to(at::kCUDA);
            cv_a_33.to(at::kCUDA);
            cv_a_34.to(at::kCUDA);
            cv_a_35->to(at::kCUDA);
            cv_a_36.to(at::kCUDA);
            cv_a_37->to(at::kCUDA);
            cv_a_38.to(at::kCUDA);
            cv_a_39.to(at::kCUDA);
            cv_a_40.to(at::kCUDA);
            cv_a_41.to(at::kCUDA);
            cv_a_42.to(at::kCUDA);
            cv_a_43->to(at::kCUDA);
            cv_a_44.to(at::kCUDA);
            cv_a_45.to(at::kCUDA);
            cv_a_46.to(at::kCUDA);
            cv_a_47.to(at::kCUDA);
            cv_a_48.to(at::kCUDA);
            cv_a_49->to(at::kCUDA);
            cv_a_50.to(at::kCUDA);
            cv_a_51.to(at::kCUDA);
            cv_a_52.to(at::kCUDA);
            cv_a_53.to(at::kCUDA);
            cv_a_54->to(at::kCUDA);
            cv_a_55.to(at::kCUDA);
            cv_a_56.to(at::kCUDA);
            cv_a_57.to(at::kCUDA);
            cv_b_0.to(at::kCUDA);
            cv_b_1.to(at::kCUDA);
            cv_b_2.to(at::kCUDA);
            cv_b_3.to(at::kCUDA);
            cv_b_4.to(at::kCUDA);
            cv_b_5.to(at::kCUDA);
            cv_b_6->to(at::kCUDA);
            cv_b_7.to(at::kCUDA);
            cv_b_8->to(at::kCUDA);
            cv_b_9.to(at::kCUDA);
            cv_b_10.to(at::kCUDA);
            cv_b_11.to(at::kCUDA);
            cv_b_12.to(at::kCUDA);
            cv_b_13.to(at::kCUDA);
            cv_b_14.to(at::kCUDA);
            cv_b_15.to(at::kCUDA);
            cv_b_16.to(at::kCUDA);
            cv_b_17.to(at::kCUDA);
            cv_b_18.to(at::kCUDA);
            cv_b_19->to(at::kCUDA);
            cv_b_20.to(at::kCUDA);
            cv_b_21.to(at::kCUDA);
            cv_b_22.to(at::kCUDA);
            cv_b_23.to(at::kCUDA);
            cv_b_24.to(at::kCUDA);
            cv_b_25->to(at::kCUDA);
            cv_b_26.to(at::kCUDA);
            cv_b_27.to(at::kCUDA);
            cv_b_28.to(at::kCUDA);
            cv_b_29.to(at::kCUDA);
            cv_b_30.to(at::kCUDA);
            cv_b_31.to(at::kCUDA);
            cv_b_32->to(at::kCUDA);
            cv_b_33.to(at::kCUDA);
            cv_b_34.to(at::kCUDA);
            cv_b_35.to(at::kCUDA);
            cv_b_36.to(at::kCUDA);
            cv_b_37->to(at::kCUDA);
            cv_b_38.to(at::kCUDA);
            cv_b_39.to(at::kCUDA);
            cv_b_40.to(at::kCUDA);
            cv_b_41.to(at::kCUDA);
            cv_b_42.to(at::kCUDA);
            cv_b_43->to(at::kCUDA);
            cv_b_44.to(at::kCUDA);
            cv_b_45.to(at::kCUDA);
            cv_b_46.to(at::kCUDA);
            cv_b_47.to(at::kCUDA);
            cv_b_48.to(at::kCUDA);
            cv_b_49.to(at::kCUDA);
            cv_b_50.to(at::kCUDA);
            cv_b_51->to(at::kCUDA);
            cv_b_52.to(at::kCUDA);
            cv_b_53->to(at::kCUDA);
            cv_b_54.to(at::kCUDA);
            cv_b_55->to(at::kCUDA);
            cv_b_56->to(at::kCUDA);
            cv_b_57.to(at::kCUDA);
            c1.to(at::kCUDA);
            c2.to(at::kCUDA);
            avg0.to(at::kCUDA);
            c3->to(at::kCUDA);
            admaxp2d0.to(at::kCUDA);
            linear0.to(at::kCUDA);
        }
    
    torch::Tensor forward(torch::Tensor x)
    {
        torch::Tensor x_before;

        x = c0(x);
        x = relu.forward(x);
        x = mxp2d0.forward(x);

        // block 1
        x_before = x;
        x = relu.forward(x);
        x = cv_a_0.forward(x);
        x = relu.forward(x);
        x = cv_b_0.forward(x);
        x = torch::cat({x_before, x}, 1);
        
        x_before = x;
        x = relu.forward(x);
        x = cv_a_1(x);
        x = relu.forward(x);
        x = cv_b_1.forward(x);
        x = torch::cat({x_before, x}, 1);

        x_before = x;
        x = relu.forward(x);
        x = cv_a_2.forward(x);
        x = relu.forward(x);
        x = cv_b_2.forward(x);
        x = torch::cat({x_before, x}, 1);

        x_before = x;
        x = relu.forward(x);
        x = cv_a_3.forward(x);
        x = relu.forward(x);
        x = cv_b_3.forward(x);
        x = torch::cat({x_before, x}, 1);

        x_before = x;
        x = relu.forward(x);
        x = cv_a_4.forward(x);
        x = relu.forward(x);
        x = cv_b_4.forward(x);
        x = torch::cat({x_before, x}, 1);

        x_before = x;
        x = relu.forward(x);
        x = cv_a_5.forward(x);
        x = relu.forward(x);
        x = cv_b_5.forward(x);
        x = torch::cat({x_before, x}, 1);
        x = relu.forward(x);
        x = c1.forward(x);
        x = avg1.forward(x);

        // block 2
        x_before = x;
        x = relu.forward(x);
        x = cv_a_6.forward(x);
        x = relu.forward(x);
        x = cv_b_6(x);
        x = torch::cat({x_before, x}, 1);

        x_before = x;
        x = relu.forward(x);
        x = cv_a_7.forward(x);
        x = relu.forward(x);
        x = cv_b_7.forward(x);
        x = torch::cat({x_before, x}, 1);

        x_before = x;
        x = relu.forward(x);
        x = cv_a_8(x);
        x = relu.forward(x);
        x = cv_b_8(x);
        x = torch::cat({x_before, x}, 1);

        x_before = x;
        x = relu.forward(x);
        x = cv_a_9.forward(x);
        x = relu.forward(x);
        x = cv_b_9.forward(x);
        x = torch::cat({x_before, x}, 1);

        x_before = x;
        x = relu.forward(x);
        x = cv_a_10.forward(x);
        x = relu.forward(x);
        x = cv_b_10.forward(x);
        x = torch::cat({x_before, x}, 1);

        x_before = x;
        x = relu.forward(x);
        x = cv_a_11.forward(x);
        x = relu.forward(x);
        x = cv_b_11.forward(x);
        x = torch::cat({x_before, x}, 1);

        x_before = x;
        x = relu.forward(x);
        x = cv_a_12.forward(x);
        x = relu.forward(x);
        x = cv_b_12.forward(x);
        x = torch::cat({x_before, x}, 1);

        x_before = x;
        x = relu.forward(x);
        x = cv_a_13.forward(x);
        x = relu.forward(x);
        x = cv_b_13.forward(x);
        x = torch::cat({x_before, x}, 1);

        x_before = x;
        x = relu.forward(x);
        x = cv_a_14.forward(x);
        x = relu.forward(x);
        x = cv_b_14.forward(x);
        x = torch::cat({x_before, x}, 1);

        x_before = x;
        x = relu.forward(x);
        x = cv_a_15.forward(x);
        x = relu.forward(x);
        x = cv_b_15.forward(x);
        x = torch::cat({x_before, x}, 1);

        x_before = x;
        x = relu.forward(x);
        x = cv_a_16.forward(x);
        x = relu.forward(x);
        x = cv_b_16.forward(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = relu.forward(x);
        x = cv_a_17(x);
        x = relu.forward(x);
        x = cv_b_17.forward(x);
        x = torch::cat({x_before, x}, 1);

        // transition 2
        x = relu.forward(x);
        x = c2.forward(x);
        x = avg0.forward(x);


        // block 3
        x_before = x;
        x = relu.forward(x);
        x = cv_a_18.forward(x);
        x = relu.forward(x);
        x = cv_b_18.forward(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = relu.forward(x);
        x = cv_a_19.forward(x);
        x = relu.forward(x);
        x = cv_b_19(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = relu.forward(x);
        x = cv_a_20.forward(x);
        x = relu.forward(x);
        x = cv_b_20.forward(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = relu.forward(x);
        x = cv_a_21.forward(x);
        x = relu.forward(x);
        x = cv_b_21.forward(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = relu.forward(x);
        x = cv_a_22.forward(x);
        x = relu.forward(x);
        x = cv_b_22.forward(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = relu.forward(x);
        x = cv_a_23.forward(x);
        x = relu.forward(x);
        x = cv_b_23.forward(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = relu.forward(x);
        x = cv_a_24.forward(x);
        x = relu.forward(x);
        x = cv_b_24.forward(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = relu.forward(x);
        x = cv_a_25.forward(x);
        x = relu.forward(x);
        x = cv_b_25(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = relu.forward(x);
        x = cv_a_26.forward(x);
        x = relu.forward(x);
        x = cv_b_26.forward(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = relu.forward(x);
        x = cv_a_27.forward(x);
        x = relu.forward(x);
        x = cv_b_27.forward(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = relu.forward(x);
        x = cv_a_28.forward(x);
        x = relu.forward(x);
        x = cv_b_28.forward(x);
        x = torch::cat({x_before, x}, 1);

        x_before = x;
        x = relu.forward(x);
        x = cv_a_29.forward(x);
        x = relu.forward(x);
        x = cv_b_29.forward(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = relu.forward(x);
        x = cv_a_30.forward(x);
        x = relu.forward(x);
        x = cv_b_30.forward(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = relu.forward(x);
        x = cv_a_31.forward(x);
        x = relu.forward(x);
        x = cv_b_31.forward(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = relu.forward(x);
        x = cv_a_32.forward(x);
        x = relu.forward(x);
        x = cv_b_32(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = relu.forward(x);
        x = cv_a_33.forward(x);
        x = relu.forward(x);
        x = cv_b_33.forward(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = relu.forward(x);
        x = cv_a_34.forward(x);
        x = relu.forward(x);
        x = cv_b_34.forward(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = relu.forward(x);
        x = cv_a_35(x);
        x = relu.forward(x);
        x = cv_b_35.forward(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = relu.forward(x);
        x = cv_a_36.forward(x);
        x = relu.forward(x);
        x = cv_b_36.forward(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = relu.forward(x);
        x = cv_a_37(x);
        x = relu.forward(x);
        x = cv_b_37(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = relu.forward(x);
        x = cv_a_38.forward(x);
        x = relu.forward(x);
        x = cv_b_38.forward(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = relu.forward(x);
        x = cv_a_39.forward(x);
        x = relu.forward(x);
        x = cv_b_39.forward(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = relu.forward(x);
        x = cv_a_40.forward(x);
        x = relu.forward(x);
        x = cv_b_40.forward(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = relu.forward(x);
        x = cv_a_41.forward(x);
        x = relu.forward(x);
        x = cv_b_41.forward(x);
        x = torch::cat({x_before, x}, 1);

        //x = b3(x);
        x = relu.forward(x);
        x = c3(x);
        x = avg0.forward(x);

        //block 4
        x_before = x;
        x = relu.forward(x);
        x = cv_a_42.forward(x);
        x = relu.forward(x);
        x = cv_b_42.forward(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = relu.forward(x);
        x = cv_a_43(x);
        x = relu.forward(x);
        x = cv_b_43(x);
        x = torch::cat({x_before, x}, 1);

        x_before = x;
        x = relu.forward(x);
        x = cv_a_44.forward(x);
        x = relu.forward(x);
        x = cv_b_44.forward(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = relu.forward(x);
        x = cv_a_45.forward(x);
        x = relu.forward(x);
        x = cv_b_45.forward(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = relu.forward(x);
        x = cv_a_46.forward(x);
        x = relu.forward(x);
        x = cv_b_46.forward(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = relu.forward(x);
        x = cv_a_47.forward(x);
        x = relu.forward(x);
        x = cv_b_47.forward(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = relu.forward(x);
        x = cv_a_48.forward(x);
        x = relu.forward(x);
        x = cv_b_48.forward(x);
        x = torch::cat({x_before, x}, 1);

        x_before = x;
        x = relu.forward(x);
        x = cv_a_49(x);
        x = relu.forward(x);
        x = cv_b_49.forward(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = relu.forward(x);
        x = cv_a_50.forward(x);
        x = relu.forward(x);
        x = cv_b_50.forward(x);
        x = torch::cat({x_before, x}, 1);

        x_before = x;
        x = relu.forward(x);
        x = cv_a_51.forward(x);
        x = relu.forward(x);
        x = cv_b_51(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = relu.forward(x);
        x = cv_a_52.forward(x);
        x = relu.forward(x);
        x = cv_b_52.forward(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = relu.forward(x);
        x = cv_a_53.forward(x);
        x = relu.forward(x);
        x = cv_b_53(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = relu.forward(x);
        x = cv_a_54(x);
        x = relu.forward(x);
        x = cv_b_54.forward(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = relu.forward(x);
        x = cv_a_55.forward(x);
        x = relu.forward(x);
        x = cv_b_55(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = relu.forward(x);
        x = cv_a_56.forward(x);
        x = relu.forward(x);
        x = cv_b_56(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = relu.forward(x);
        x = cv_a_57.forward(x);
        x = relu.forward(x);
        x = cv_b_57.forward(x);
        x = torch::cat({x_before, x}, 1);

        // //end
        // x = relu.forward(x);
        // x = admaxp2d0.forward(x);
        // x = x.view({x.sizes()[0], -1});
        // x = linear0.forward(x);
        return x;
    }
    void morphpara(){
        // nothing to do, inherit from virtual func
    }
};

class dense_gpu_part2 : public dense_part
{
public:
    torch::nn::Conv2d c0;
    operator7 linear0;
    operator7 linear1;
    operator7 linear2;
    dense_gpu_part2():
        c0(torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 7).padding(3).stride(2))),
        linear0(1024, 32768),
        linear1(32768, 9000),
        linear2(9000, 1000)
        {
            c0->to(at::kCUDA);
            linear0.to(at::kCUDA);
            linear1.to(at::kCUDA);
            linear2.to(at::kCUDA);
        }
    torch::Tensor forward(torch::Tensor x)
    {
        x = linear0.forward(x);
        x = linear1.forward(x);
        // x = linear2.forward(x);
        return x;
    }
    void morphpara(){
        // nothing to do, inherit from virtual func
    }
};

class dense_gpu_part3 : public dense_part
{
public:
    torch::nn::Conv2d c0;
    dense_gpu_part3():
        c0(torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 7).padding(3).stride(2)))
        {
            c0->to(at::kCUDA);
        }
    torch::Tensor forward(torch::Tensor x)
    {
        return x;
    }
    void morphpara(){
        // nothing to do, inherit from virtual func
    }
};


dense_part* models[] = {
    new dense_warmup(),
    new dense_gpu_part1(),
    new dense_gpu_part2(),
    new dense_gpu_part3()
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
