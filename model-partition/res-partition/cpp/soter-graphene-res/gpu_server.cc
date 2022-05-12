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
int total_para = 0;
int scalar = 4;

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
        num_flat_features0(x);
        return x;   
    }
};
struct operator2 : public torch::nn::Module
{
    torch::nn::Conv2d conv;
    operator2(int a, int b, int c, int d, int e):
        conv(conv_options(a, b, c, d, e))
        {
            register_module("conv", conv);
        }
    torch::Tensor forward(torch::Tensor x)
    {
        x = conv(x);
        num_flat_features0(x);
        return x;
    }
    void morph(){
        conv->weight = conv->weight * scalar;
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
        num_flat_features0(x);
        return x;
    }
    void morph(){
        conv->weight = conv->weight * scalar;
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
    void morph(){
        linear->weight = linear->weight * scalar;
        linear->bias = linear->bias * scalar;
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
        num_flat_features0(x);
        return x;
    }
    void morph(){
        conv->weight = conv->weight * scalar;
    }
};

class resnet_part : public torch::nn::Module {
    public:
    virtual torch::Tensor forward(torch::Tensor x) = 0;
    virtual void morphpara() = 0;
};

class resnet_warmup : public resnet_part
{
public:
    torch::nn::Conv2d c0;
    operator4 mxp2d0;
    operator5 avg0;
    operator7 linear0;
    operator1 relu;
    operator2 cv_a_0;
    operator2 cv_a_1;
    operator2 cv_a_2;
    torch::nn::Conv2d cv_a_3;
    operator2 cv_a_4;
    operator2 cv_a_5;
    operator2 cv_a_6;
    operator2 cv_a_7;
    operator2 cv_a_8;
    operator2 cv_a_9;
    operator2 cv_a_10;
    operator2 cv_a_11;
    torch::nn::Conv2d cv_a_12;
    operator2 cv_a_13;
    operator2 cv_a_14;
    operator2 cv_a_15;
    operator2 cv_a_16;
    operator2 cv_a_17;
    operator2 cv_a_18;
    operator2 cv_a_19;
    operator2 cv_a_20;
    operator2 cv_a_21;
    operator2 cv_a_22;
    operator2 cv_a_23;
    torch::nn::Conv2d cv_a_24;
    operator2 cv_a_25;
    torch::nn::Conv2d cv_a_26;
    operator2 cv_a_27;
    torch::nn::Conv2d cv_a_28;
    operator2 cv_a_29;
    torch::nn::Conv2d cv_a_30;
    operator2 cv_a_31;
    torch::nn::Conv2d cv_a_32;
    operator2 cv_a_33;
    operator2 cv_a_34;
    operator2 cv_a_35;
    operator2 cv_a_36;
    operator2 cv_a_37;
    operator2 cv_a_38;
    operator2 cv_a_39;
    operator2 cv_a_40;
    operator2 cv_a_41;
    operator2 cv_a_42;
    operator2 cv_a_43;
    operator2 cv_a_44;
    operator2 cv_a_45;
    torch::nn::Conv2d cv_a_46;
    operator2 cv_a_47;
    operator2 cv_a_48;
    operator2 cv_a_49;
    operator2 cv_a_50;
    operator2 cv_a_51;
    operator2 cv_a_52;
    torch::nn::Conv2d cv_a_53;
    operator3 cv_b_0;
    operator3 cv_b_1;
    operator3 cv_b_2;
    operator3 cv_b_3;
    operator8 cv_c_0;
    operator8 cv_c_1;
    operator8 cv_c_2;
    operator8 cv_c_3;
    torch::nn::Conv2d cv_c_4;
    torch::nn::Conv2d cv_c_5;
    operator8 cv_c_6;
    operator8 cv_c_7;
    operator8 cv_c_8;
    operator8 cv_c_9;
    operator8 cv_c_10;
    operator8 cv_c_11;
    operator8 cv_c_12;
    torch::nn::Conv2d cv_c_13;
    operator8 cv_c_14;
    torch::nn::Conv2d cv_c_15;
    operator8 cv_c_16;
    torch::nn::Conv2d cv_c_17;
    torch::nn::Conv2d cv_c_18;
    operator8 cv_c_19;
    operator8 cv_c_20;
    operator8 cv_c_21;
    operator8 cv_c_22;
    torch::nn::Conv2d cv_c_23;
    operator8 cv_c_24;
    operator8 cv_c_25;
    operator8 cv_c_26;
    operator8 cv_c_27;
    torch::nn::Conv2d cv_c_28;
    operator8 cv_c_29;
    operator8 cv_c_30;
    operator8 cv_c_31;
    operator8 cv_c_32;
    operator8 cv_c_33;
    operator8 cv_c_34;
    operator8 cv_c_35;
    torch::nn::Conv2d cv_c_36;
    torch::nn::Conv2d cv_c_37;
    operator8 cv_c_38;
    operator8 cv_c_39;
    operator8 cv_c_40;
    torch::nn::Conv2d cv_c_41;
    operator8 cv_c_42;
    torch::nn::Conv2d cv_c_43;
    operator8 cv_c_44;
    operator8 cv_c_45;
    operator8 cv_c_46;
    torch::nn::Conv2d cv_c_47;
    torch::nn::Conv2d cv_c_48;
    operator8 cv_c_49;
    operator8 cv_c_50;
    operator8 cv_c_51;
    operator8 cv_c_52;
    operator8 cv_c_53;
    torch::nn::Conv2d cv_c_54;
    operator8 cv_c_55;
    operator8 cv_c_56;
    operator8 cv_c_57;
    operator8 cv_c_58;
    torch::nn::Conv2d cv_c_59;
    torch::nn::Conv2d cv_c_60;
    operator8 cv_c_61;
    operator8 cv_c_62;
    operator8 cv_c_63;
    torch::nn::Conv2d cv_c_64;
    operator8 cv_c_65;
    operator8 cv_c_66;
    torch::nn::Conv2d cv_c_67;
    operator8 cv_c_68;
    torch::nn::Conv2d cv_c_69;
    torch::nn::Conv2d cv_c_70;
    operator8 cv_c_71;
    torch::nn::Conv2d cv_c_72;
    operator8 cv_c_73;
    torch::nn::Conv2d cv_c_74;
    operator8 cv_c_75;
    torch::nn::Conv2d cv_c_76;
    operator8 cv_c_77;
    torch::nn::Conv2d cv_c_78;
    operator8 cv_c_79;
    operator8 cv_c_80;
    operator8 cv_c_81;
    operator8 cv_c_82;
    operator8 cv_c_83;
    operator8 cv_c_84;
    operator8 cv_c_85;
    operator8 cv_c_86;
    operator8 cv_c_87;
    operator8 cv_c_88;
    operator8 cv_c_89;
    operator8 cv_c_90;
    torch::nn::Conv2d cv_c_91;
    operator8 cv_c_92;
    operator8 cv_c_93;
    operator8 cv_c_94;
    operator8 cv_c_95;
    torch::nn::Conv2d cv_c_96;
    operator8 cv_c_97;
    operator8 cv_c_98;
    torch::nn::Conv2d cv_c_99;
    torch::nn::Conv2d cv_c_100;
    operator8 cv_c_101;
    operator8 cv_c_102;
    operator8 cv_c_103;
    operator8 cv_c_104;
    operator8 cv_c_105;
    torch::nn::Conv2d cv_c_106;
    torch::nn::Conv2d cv_c_107;
// tag1
    resnet_warmup():
        c0(conv_options(3, 64, 7, 2, 3)),
        mxp2d0(3, 1, 2),
        avg0(7, 1),
        linear0(2048, 1000),
        cv_a_0(64, 64, 3, 1, 1),
        cv_a_1(64, 64, 3, 1, 1),
        cv_a_2(64, 64, 3, 1, 1),
        cv_a_3(conv_options(64, 64, 3, 1, 1)),
        cv_a_4(128, 128, 3, 2, 1),
        cv_a_5(128, 128, 3, 1, 1),
        cv_a_6(128, 128, 3, 1, 1),
        cv_a_7(128, 128, 3, 1, 1),
        cv_a_8(128, 128, 3, 1, 1),
        cv_a_9(128, 128, 3, 1, 1),
        cv_a_10(128, 128, 3, 1, 1),
        cv_a_11(128, 128, 3, 1, 1),
        cv_a_12(conv_options(128, 128, 3, 1, 1)),
        cv_a_13(256, 256, 3, 2, 1),
        cv_a_14(256, 256, 3, 1, 1),
        cv_a_15(256, 256, 3, 1, 1),
        cv_a_16(256, 256, 3, 1, 1),
        cv_a_17(256, 256, 3, 1, 1),
        cv_a_18(256, 256, 3, 1, 1),
        cv_a_19(256, 256, 3, 1, 1),
        cv_a_20(256, 256, 3, 1, 1),
        cv_a_21(256, 256, 3, 1, 1),
        cv_a_22(256, 256, 3, 1, 1),
        cv_a_23(256, 256, 3, 1, 1),
        cv_a_24(conv_options(256, 256, 3, 1, 1)),
        cv_a_25(256, 256, 3, 1, 1),
        cv_a_26(conv_options(256, 256, 3, 1, 1)),
        cv_a_27(256, 256, 3, 1, 1),
        cv_a_28(conv_options(256, 256, 3, 1, 1)),
        cv_a_29(256, 256, 3, 1, 1),
        cv_a_30(conv_options(256, 256, 3, 1, 1)),
        cv_a_31(256, 256, 3, 1, 1),
        cv_a_32(conv_options(256, 256, 3, 1, 1)),
        cv_a_33(256, 256, 3, 1, 1),
        cv_a_34(256, 256, 3, 1, 1),
        cv_a_35(256, 256, 3, 1, 1),
        cv_a_36(256, 256, 3, 1, 1),
        cv_a_37(256, 256, 3, 1, 1),
        cv_a_38(256, 256, 3, 1, 1),
        cv_a_39(256, 256, 3, 1, 1),
        cv_a_40(256, 256, 3, 1, 1),
        cv_a_41(256, 256, 3, 1, 1),
        cv_a_42(256, 256, 3, 1, 1),
        cv_a_43(256, 256, 3, 1, 1),
        cv_a_44(256, 256, 3, 1, 1),
        cv_a_45(256, 256, 3, 1, 1),
        cv_a_46(conv_options(256, 256, 3, 1, 1)),
        cv_a_47(256, 256, 3, 1, 1),
        cv_a_48(256, 256, 3, 1, 1),
        cv_a_49(256, 256, 3, 1, 1),
        cv_a_50(512, 512, 3, 2, 1),
        cv_a_51(512, 512, 3, 1, 1),
        cv_a_52(512, 512, 3, 1, 1),
        cv_a_53(conv_options(512, 512, 3, 1, 1)),
        cv_b_0(64, 256, 1, 1),
        cv_b_1(256, 512, 1, 2),
        cv_b_2(512, 1024, 1, 2),
        cv_b_3(1024, 2048, 1, 2),
        cv_c_0(64, 64, 1),
        cv_c_1(64, 256, 1),
        cv_c_2(256, 64, 1),
        cv_c_3(64, 256, 1),
        cv_c_4(conv_options(256, 64, 1)),
        cv_c_5(conv_options(64, 256, 1)),
        cv_c_6(256, 64, 1),
        cv_c_7(64, 256, 1),
        cv_c_8(256, 128, 1),
        cv_c_9(128, 512, 1),
        cv_c_10(512, 128, 1),
        cv_c_11(128, 512, 1),
        cv_c_12(512, 128, 1),
        cv_c_13(conv_options(128, 512, 1)),
        cv_c_14(512, 128, 1),
        cv_c_15(conv_options(128, 512, 1)),
        cv_c_16(512, 128, 1),
        cv_c_17(conv_options(128, 512, 1)),
        cv_c_18(conv_options(512, 128, 1)),
        cv_c_19(128, 512, 1),
        cv_c_20(512, 128, 1),
        cv_c_21(128, 512, 1),
        cv_c_22(512, 128, 1),
        cv_c_23(conv_options(128, 512, 1)),
        cv_c_24(512, 128, 1),
        cv_c_25(128, 512, 1),
        cv_c_26(512, 256, 1),
        cv_c_27(256, 1024, 1),
        cv_c_28(conv_options(1024, 256, 1)),
        cv_c_29(256, 1024, 1),
        cv_c_30(1024, 256, 1),
        cv_c_31(256, 1024, 1),
        cv_c_32(1024, 256, 1),
        cv_c_33(256, 1024, 1),
        cv_c_34(1024, 256, 1),
        cv_c_35(256, 1024, 1),
        cv_c_36(conv_options(1024, 256, 1)),
        cv_c_37(conv_options(256, 1024, 1)),
        cv_c_38(1024, 256, 1),
        cv_c_39(256, 1024, 1),
        cv_c_40(1024, 256, 1),
        cv_c_41(conv_options(256, 1024, 1)),
        cv_c_42(1024, 256, 1),
        cv_c_43(conv_options(256, 1024, 1)),
        cv_c_44(1024, 256, 1),
        cv_c_45(256, 1024, 1),
        cv_c_46(1024, 256, 1),
        cv_c_47(conv_options(256, 1024, 1)),
        cv_c_48(conv_options(1024, 256, 1)),
        cv_c_49(256, 1024, 1),
        cv_c_50(1024, 256, 1),
        cv_c_51(256, 1024, 1),
        cv_c_52(1024, 256, 1),
        cv_c_53(256, 1024, 1),
        cv_c_54(conv_options(1024, 256, 1)),
        cv_c_55(256, 1024, 1),
        cv_c_56(1024, 256, 1),
        cv_c_57(256, 1024, 1),
        cv_c_58(1024, 256, 1),
        cv_c_59(conv_options(256, 1024, 1)),
        cv_c_60(conv_options(1024, 256, 1)),
        cv_c_61(256, 1024, 1),
        cv_c_62(1024, 256, 1),
        cv_c_63(256, 1024, 1),
        cv_c_64(conv_options(1024, 256, 1)),
        cv_c_65(256, 1024, 1),
        cv_c_66(1024, 256, 1),
        cv_c_67(conv_options(256, 1024, 1)),
        cv_c_68(1024, 256, 1),
        cv_c_69(conv_options(256, 1024, 1)),
        cv_c_70(conv_options(1024, 256, 1)),
        cv_c_71(256, 1024, 1),
        cv_c_72(conv_options(1024, 256, 1)),
        cv_c_73(256, 1024, 1),
        cv_c_74(conv_options(1024, 256, 1)),
        cv_c_75(256, 1024, 1),
        cv_c_76(conv_options(1024, 256, 1)),
        cv_c_77(256, 1024, 1),
        cv_c_78(conv_options(1024, 256, 1)),
        cv_c_79(256, 1024, 1),
        cv_c_80(1024, 256, 1),
        cv_c_81(256, 1024, 1),
        cv_c_82(1024, 256, 1),
        cv_c_83(256, 1024, 1),
        cv_c_84(1024, 256, 1),
        cv_c_85(256, 1024, 1),
        cv_c_86(1024, 256, 1),
        cv_c_87(256, 1024, 1),
        cv_c_88(1024, 256, 1),
        cv_c_89(256, 1024, 1),
        cv_c_90(1024, 256, 1),
        cv_c_91(conv_options(256, 1024, 1)),
        cv_c_92(1024, 256, 1),
        cv_c_93(256, 1024, 1),
        cv_c_94(1024, 256, 1),
        cv_c_95(256, 1024, 1),
        cv_c_96(conv_options(1024, 256, 1)),
        cv_c_97(256, 1024, 1),
        cv_c_98(1024, 256, 1),
        cv_c_99(conv_options(256, 1024, 1)),
        cv_c_100(conv_options(1024, 512, 1)),
        cv_c_101(512, 2048, 1),
        cv_c_102(2048, 512, 1),
        cv_c_103(512, 2048, 1),
        cv_c_104(2048, 512, 1),
        cv_c_105(512, 2048, 1),
        cv_c_106(conv_options(2048, 512, 1)),
        cv_c_107(conv_options(512, 2048, 1))
        {// tag2
            c0->to(at::kCUDA);
            mxp2d0.to(at::kCUDA);
            avg0.to(at::kCUDA);
            linear0.to(at::kCUDA);
            cv_a_0.to(at::kCUDA);
            cv_a_1.to(at::kCUDA);
            cv_a_2.to(at::kCUDA);
            cv_a_3->to(at::kCUDA);
            cv_a_4.to(at::kCUDA);
            cv_a_5.to(at::kCUDA);
            cv_a_6.to(at::kCUDA);
            cv_a_7.to(at::kCUDA);
            cv_a_8.to(at::kCUDA);
            cv_a_9.to(at::kCUDA);
            cv_a_10.to(at::kCUDA);
            cv_a_11.to(at::kCUDA);
            cv_a_12->to(at::kCUDA);
            cv_a_13.to(at::kCUDA);
            cv_a_14.to(at::kCUDA);
            cv_a_15.to(at::kCUDA);
            cv_a_16.to(at::kCUDA);
            cv_a_17.to(at::kCUDA);
            cv_a_18.to(at::kCUDA);
            cv_a_19.to(at::kCUDA);
            cv_a_20.to(at::kCUDA);
            cv_a_21.to(at::kCUDA);
            cv_a_22.to(at::kCUDA);
            cv_a_23.to(at::kCUDA);
            cv_a_24->to(at::kCUDA);
            cv_a_25.to(at::kCUDA);
            cv_a_26->to(at::kCUDA);
            cv_a_27.to(at::kCUDA);
            cv_a_28->to(at::kCUDA);
            cv_a_29.to(at::kCUDA);
            cv_a_30->to(at::kCUDA);
            cv_a_31.to(at::kCUDA);
            cv_a_32->to(at::kCUDA);
            cv_a_33.to(at::kCUDA);
            cv_a_34.to(at::kCUDA);
            cv_a_35.to(at::kCUDA);
            cv_a_36.to(at::kCUDA);
            cv_a_37.to(at::kCUDA);
            cv_a_38.to(at::kCUDA);
            cv_a_39.to(at::kCUDA);
            cv_a_40.to(at::kCUDA);
            cv_a_41.to(at::kCUDA);
            cv_a_42.to(at::kCUDA);
            cv_a_43.to(at::kCUDA);
            cv_a_44.to(at::kCUDA);
            cv_a_45.to(at::kCUDA);
            cv_a_46->to(at::kCUDA);
            cv_a_47.to(at::kCUDA);
            cv_a_48.to(at::kCUDA);
            cv_a_49.to(at::kCUDA);
            cv_a_50.to(at::kCUDA);
            cv_a_51.to(at::kCUDA);
            cv_a_52.to(at::kCUDA);
            cv_a_53->to(at::kCUDA);
            cv_b_0.to(at::kCUDA);
            cv_b_1.to(at::kCUDA);
            cv_b_2.to(at::kCUDA);
            cv_b_3.to(at::kCUDA);
            cv_c_0.to(at::kCUDA);
            cv_c_1.to(at::kCUDA);
            cv_c_2.to(at::kCUDA);
            cv_c_3.to(at::kCUDA);
            cv_c_4->to(at::kCUDA);
            cv_c_5->to(at::kCUDA);            
            cv_c_6.to(at::kCUDA);
            cv_c_7.to(at::kCUDA);
            cv_c_8.to(at::kCUDA);
            cv_c_9.to(at::kCUDA);
            cv_c_10.to(at::kCUDA);
            cv_c_11.to(at::kCUDA);
            cv_c_12.to(at::kCUDA);
            cv_c_13->to(at::kCUDA);
            cv_c_14.to(at::kCUDA);
            cv_c_15->to(at::kCUDA);
            cv_c_16.to(at::kCUDA);
            cv_c_17->to(at::kCUDA);
            cv_c_18->to(at::kCUDA);
            cv_c_19.to(at::kCUDA);
            cv_c_20.to(at::kCUDA);
            cv_c_21.to(at::kCUDA);
            cv_c_22.to(at::kCUDA);
            cv_c_23->to(at::kCUDA);
            cv_c_24.to(at::kCUDA);
            cv_c_25.to(at::kCUDA);
            cv_c_26.to(at::kCUDA);
            cv_c_27.to(at::kCUDA);
            cv_c_28->to(at::kCUDA);
            cv_c_29.to(at::kCUDA);
            cv_c_30.to(at::kCUDA);
            cv_c_31.to(at::kCUDA);
            cv_c_32.to(at::kCUDA);
            cv_c_33.to(at::kCUDA);
            cv_c_34.to(at::kCUDA);
            cv_c_35.to(at::kCUDA);
            cv_c_36->to(at::kCUDA);
            cv_c_37->to(at::kCUDA);
            cv_c_38.to(at::kCUDA);
            cv_c_39.to(at::kCUDA);
            cv_c_40.to(at::kCUDA);
            cv_c_41->to(at::kCUDA);
            cv_c_42.to(at::kCUDA);
            cv_c_43->to(at::kCUDA);
            cv_c_44.to(at::kCUDA);
            cv_c_45.to(at::kCUDA);
            cv_c_46.to(at::kCUDA);
            cv_c_47->to(at::kCUDA);
            cv_c_48->to(at::kCUDA);
            cv_c_49.to(at::kCUDA);
            cv_c_50.to(at::kCUDA);
            cv_c_51.to(at::kCUDA);
            cv_c_52.to(at::kCUDA);
            cv_c_53.to(at::kCUDA);
            cv_c_54->to(at::kCUDA);
            cv_c_55.to(at::kCUDA);
            cv_c_56.to(at::kCUDA);
            cv_c_57.to(at::kCUDA);
            cv_c_58.to(at::kCUDA);
            cv_c_59->to(at::kCUDA);
            cv_c_60->to(at::kCUDA);
            cv_c_61.to(at::kCUDA);
            cv_c_62.to(at::kCUDA);
            cv_c_63.to(at::kCUDA);
            cv_c_64->to(at::kCUDA);
            cv_c_65.to(at::kCUDA);
            cv_c_66.to(at::kCUDA);
            cv_c_67->to(at::kCUDA);
            cv_c_68.to(at::kCUDA);
            cv_c_69->to(at::kCUDA);
            cv_c_70->to(at::kCUDA);
            cv_c_71.to(at::kCUDA);
            cv_c_72->to(at::kCUDA);
            cv_c_73.to(at::kCUDA);
            cv_c_74->to(at::kCUDA);
            cv_c_75.to(at::kCUDA);
            cv_c_76->to(at::kCUDA);
            cv_c_77.to(at::kCUDA);
            cv_c_78->to(at::kCUDA);
            cv_c_79.to(at::kCUDA);
            cv_c_80.to(at::kCUDA);
            cv_c_81.to(at::kCUDA);
            cv_c_82.to(at::kCUDA);
            cv_c_83.to(at::kCUDA);
            cv_c_84.to(at::kCUDA);
            cv_c_85.to(at::kCUDA);
            cv_c_86.to(at::kCUDA);
            cv_c_87.to(at::kCUDA);
            cv_c_88.to(at::kCUDA);
            cv_c_89.to(at::kCUDA);
            cv_c_90.to(at::kCUDA);
            cv_c_91->to(at::kCUDA);
            cv_c_92.to(at::kCUDA);
            cv_c_93.to(at::kCUDA);
            cv_c_94.to(at::kCUDA);
            cv_c_95.to(at::kCUDA);
            cv_c_96->to(at::kCUDA);
            cv_c_97.to(at::kCUDA);
            cv_c_98.to(at::kCUDA);
            cv_c_99->to(at::kCUDA);
            cv_c_100->to(at::kCUDA);
            cv_c_101.to(at::kCUDA);
            cv_c_102.to(at::kCUDA);
            cv_c_103.to(at::kCUDA);
            cv_c_104.to(at::kCUDA);
            cv_c_105.to(at::kCUDA);
            cv_c_106->to(at::kCUDA);
            cv_c_107->to(at::kCUDA);
            relu.to(at::kCUDA);
        }

    torch::Tensor forward(torch::Tensor x)
    {// tag3
        torch::Tensor temp;
        x = c0(x);
        x = relu.forward(x);
        x = mxp2d0.forward(x);
 
        //ATTENTION: following is BLOCK 1
        at::Tensor residual0(x.clone());
        temp = residual0.to(at::kCUDA);
        x = cv_c_0.forward(x);
        x = relu.forward(x);
        x = cv_a_0.forward(x);
        x = relu.forward(x);
        x = cv_c_1.forward(x);
        temp = cv_b_0.forward(temp);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual1(x.clone());
        temp = residual1.to(at::kCUDA);
        x = cv_c_2.forward(x);
        x = relu.forward(x);
        x = cv_a_1.forward(x);
        x = relu.forward(x);
        x = cv_c_3.forward(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual2(x.clone());
        temp = residual2.to(at::kCUDA);
        x = cv_c_4(x);
        x = relu.forward(x);
        x = cv_a_2.forward(x);
        x = relu.forward(x);
        x = cv_c_5(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual3(x.clone());
        temp = residual3.to(at::kCUDA);
        x = cv_c_6.forward(x);
        x = relu.forward(x);
        x = cv_a_3(x);
        x = relu.forward(x);
        x = cv_c_7.forward(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual4(x.clone());
        temp = residual4.to(at::kCUDA);
        x = cv_c_8.forward(x);
        x = relu.forward(x);
        x = cv_a_4.forward(x);
        x = relu.forward(x);
        x = cv_c_9.forward(x);
        temp = cv_b_1.forward(temp);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual5(x.clone());
        temp = residual5.to(at::kCUDA);
        x = cv_c_10.forward(x);
        x = relu.forward(x);
        x = cv_a_5.forward(x);
        x = relu.forward(x);
        x = cv_c_11.forward(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual6(x.clone());
        temp = residual6.to(at::kCUDA);
        x = cv_c_12.forward(x);
        x = relu.forward(x);
        x = cv_a_6.forward(x);
        x = relu.forward(x);
        x = cv_c_13(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual7(x.clone());
        temp = residual7.to(at::kCUDA);
        x = cv_c_14.forward(x);
        x = relu.forward(x);
        x = cv_a_7.forward(x);
        x = relu.forward(x);
        x = cv_c_15(x);
        x += temp;
        x = relu.forward(x);


        at::Tensor residual8(x.clone());
        temp = residual8.to(at::kCUDA);
        x = cv_c_16.forward(x);
        x = relu.forward(x);
        x = cv_a_8.forward(x);
        x = relu.forward(x);
        x = cv_c_17(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual9(x.clone());
        temp = residual9.to(at::kCUDA);
        x = cv_c_18(x);
        x = relu.forward(x);
        x = cv_a_9.forward(x);
        x = relu.forward(x);
        x = cv_c_19.forward(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual10(x.clone());
        temp = residual10.to(at::kCUDA);
        x = cv_c_20.forward(x);
        x = relu.forward(x);
        x = cv_a_10.forward(x);
        x = relu.forward(x);
        x = cv_c_21.forward(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual11(x.clone());
        temp = residual11.to(at::kCUDA);
        x = cv_c_22.forward(x);
        x = relu.forward(x);
        x = cv_a_11.forward(x);
        x = relu.forward(x);
        x = cv_c_23(x);
        x += temp;
        x = relu.forward(x);
        

        at::Tensor residual12(x.clone());
        temp = residual12.to(at::kCUDA);
        x = cv_c_24.forward(x);
        x = relu.forward(x);
        x = cv_a_12(x);
        x = relu.forward(x);
        x = cv_c_25.forward(x);
        x += temp;
        x = relu.forward(x);
       

        //ATTENTION: following is BLOCK 3

        at::Tensor residual13(x.clone());
        temp = residual13.to(at::kCUDA);
        x = cv_c_26.forward(x);
        x = relu.forward(x);
        x = cv_a_13.forward(x);
        x = relu.forward(x);
        x = cv_c_27.forward(x);
        temp = cv_b_2.forward(temp);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual14(x.clone());
        temp = residual14.to(at::kCUDA);
        x = cv_c_28(x);
        x = relu.forward(x);
        x = cv_a_14.forward(x);
        x = relu.forward(x);
        x = cv_c_29.forward(x);
        x += temp;
        x = relu.forward(x);


        at::Tensor residual15(x.clone());
        temp = residual15.to(at::kCUDA);
        x = cv_c_30.forward(x);
        x = relu.forward(x);
        x = cv_a_15.forward(x);
        x = relu.forward(x);
        x = cv_c_31.forward(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual16(x.clone());
        temp = residual16.to(at::kCUDA);
        x = cv_c_32.forward(x);
        x = relu.forward(x);
        x = cv_a_16.forward(x);
        x = relu.forward(x);
        x = cv_c_33.forward(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual17(x.clone());
        temp = residual17.to(at::kCUDA);
        x = cv_c_34.forward(x);
        x = relu.forward(x);
        x = cv_a_17.forward(x);
        x = relu.forward(x);
        x = cv_c_35.forward(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual18(x.clone());
        temp = residual18.to(at::kCUDA);
        x = cv_c_36(x);
        x = relu.forward(x);
        x = cv_a_18.forward(x);
        x = relu.forward(x);
        x = cv_c_37(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual19(x.clone());
        temp = residual19.to(at::kCUDA);
        x = cv_c_38.forward(x);
        x = relu.forward(x);
        x = cv_a_19.forward(x);
        x = relu.forward(x);
        x = cv_c_39.forward(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual20(x.clone());
        temp = residual20.to(at::kCUDA);
        x = cv_c_40.forward(x);
        x = relu.forward(x);
        x = cv_a_20.forward(x);
        x = relu.forward(x);
        x = cv_c_41(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual21(x.clone());
        temp = residual21.to(at::kCUDA);
        x = cv_c_42.forward(x);
        x = relu.forward(x);
        x = cv_a_21.forward(x);
        x = relu.forward(x);
        x = cv_c_43(x);
        x += temp;
        x = relu.forward(x);


        at::Tensor residual22(x.clone());
        temp = residual22.to(at::kCUDA);
        x = cv_c_44.forward(x);
        x = relu.forward(x);
        x = cv_a_22.forward(x);
        x = relu.forward(x);
        x = cv_c_45.forward(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual23(x.clone());
        temp = residual23.to(at::kCUDA);
        x = cv_c_46.forward(x);
        x = relu.forward(x);
        x = cv_a_23.forward(x);
        x = relu.forward(x);
        x = cv_c_47(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual24(x.clone());
        temp = residual24.to(at::kCUDA);
        x = cv_c_48(x);
        x = relu.forward(x);
        x = cv_a_24(x);
        x = relu.forward(x);
        x = cv_c_49.forward(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual25(x.clone());
        temp = residual25.to(at::kCUDA);
        x = cv_c_50.forward(x);
        x = relu.forward(x);
        x = cv_a_25.forward(x);
        x = relu.forward(x);
        x = cv_c_51.forward(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual26(x.clone());
        temp = residual26.to(at::kCUDA);
        x = cv_c_52.forward(x);
        x = relu.forward(x);
        x = cv_a_26(x);
        x = relu.forward(x);
        x = cv_c_53.forward(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual27(x.clone());
        temp = residual27.to(at::kCUDA);
        x = cv_c_54(x);
        x = relu.forward(x);
        x = cv_a_27.forward(x);
        x = relu.forward(x);
        x = cv_c_55.forward(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual28(x.clone());
        temp = residual28.to(at::kCUDA);
        x = cv_c_56.forward(x);
        x = relu.forward(x);
        x = cv_a_28(x);
        x = relu.forward(x);
        x = cv_c_57.forward(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual29(x.clone());
        temp = residual29.to(at::kCUDA);
        x = cv_c_58.forward(x);
        x = relu.forward(x);
        x = cv_a_29.forward(x);
        x = relu.forward(x);
        x = cv_c_59(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual30(x.clone());
        temp = residual30.to(at::kCUDA);
        x = cv_c_60(x);
        x = relu.forward(x);
        x = cv_a_30(x);
        x = relu.forward(x);
        x = cv_c_61.forward(x);
        x += temp;
        x = relu.forward(x);
 
        at::Tensor residual31(x.clone());
        temp = residual31.to(at::kCUDA);
        x = cv_c_62.forward(x);
        x = relu.forward(x);
        x = cv_a_31.forward(x);
        x = relu.forward(x);
        x = cv_c_63.forward(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual32(x.clone());
        temp = residual32.to(at::kCUDA);
        x = cv_c_64(x);
        x = relu.forward(x);
        x = cv_a_32(x);
        x = relu.forward(x);
        x = cv_c_65.forward(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual33(x.clone());
        temp = residual33.to(at::kCUDA);
        x = cv_c_66.forward(x);
        x = relu.forward(x);
        x = cv_a_33.forward(x);
        x = relu.forward(x);
        x = cv_c_67(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual34(x.clone());
        temp = residual34.to(at::kCUDA);
        x = cv_c_68.forward(x);
        x = relu.forward(x);
        x = cv_a_34.forward(x);
        x = relu.forward(x);
        x = cv_c_69(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual35(x.clone());
        temp = residual35.to(at::kCUDA);
        x = cv_c_70(x);
        x = relu.forward(x);
        x = cv_a_35.forward(x);
        x = relu.forward(x);
        x = cv_c_71.forward(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual36(x.clone());
        temp = residual36.to(at::kCUDA);
        x = cv_c_72(x);
        x = relu.forward(x);
        x = cv_a_36.forward(x);
        x = relu.forward(x);
        x = cv_c_73.forward(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual37(x.clone());
        temp = residual37.to(at::kCUDA);
        x = cv_c_74(x);
        x = relu.forward(x);
        x = cv_a_37.forward(x);
        x = relu.forward(x);
        x = cv_c_75.forward(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual38(x.clone());
        temp = residual38.to(at::kCUDA);
        x = cv_c_76(x);
        x = relu.forward(x);
        x = cv_a_38.forward(x);
        x = relu.forward(x);
        x = cv_c_77.forward(x);
        x += temp;
        x = relu.forward(x);


        at::Tensor residual39(x.clone());
        temp = residual39.to(at::kCUDA);
        x = cv_c_78(x);
        x = relu.forward(x);
        x = cv_a_39.forward(x);
        x = relu.forward(x);
        x = cv_c_79.forward(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual40(x.clone());
        temp = residual40.to(at::kCUDA);
        x = cv_c_80.forward(x);
        x = relu.forward(x);
        x = cv_a_40.forward(x);
        x = relu.forward(x);
        x = cv_c_81.forward(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual41(x.clone());
        temp = residual41.to(at::kCUDA);
        x = cv_c_82.forward(x);
        x = relu.forward(x);
        x = cv_a_41.forward(x);
        x = relu.forward(x);
        x = cv_c_83.forward(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual42(x.clone());
        temp = residual42.to(at::kCUDA);
        x = cv_c_84.forward(x);
        x = relu.forward(x);
        x = cv_a_42.forward(x);
        x = relu.forward(x);
        x = cv_c_85.forward(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual43(x.clone());
        temp = residual43.to(at::kCUDA);
        x = cv_c_86.forward(x);
        x = relu.forward(x);
        x = cv_a_43.forward(x);
        x = relu.forward(x);
        x = cv_c_87.forward(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual44(x.clone());
        temp = residual44.to(at::kCUDA);
        x = cv_c_88.forward(x);
        x = relu.forward(x);
        x = cv_a_44.forward(x);
        x = relu.forward(x);
        x = cv_c_89.forward(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual45(x.clone());
        temp = residual45.to(at::kCUDA);
        x = cv_c_90.forward(x);
        x = relu.forward(x);
        x = cv_a_45.forward(x);
        x = relu.forward(x);
        x = cv_c_91(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual46(x.clone());
        temp = residual46.to(at::kCUDA);
        x = cv_c_92.forward(x);
        x = relu.forward(x);
        x = cv_a_46(x);
        x = relu.forward(x);
        x = cv_c_93.forward(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual47(x.clone());
        temp = residual47.to(at::kCUDA);
        x = cv_c_94.forward(x);
        x = relu.forward(x);
        x = cv_a_47.forward(x);
        x = relu.forward(x);
        x = cv_c_95.forward(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual48(x.clone());
        temp = residual48.to(at::kCUDA);
        x = cv_c_96(x);
        x = relu.forward(x);
        x = cv_a_48.forward(x);
        x = relu.forward(x);
        x = cv_c_97.forward(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual49(x.clone());
        temp = residual49.to(at::kCUDA);
        x = cv_c_98.forward(x);
        x = relu.forward(x);
        x = cv_a_49.forward(x);
        x = relu.forward(x);
        x = cv_c_99(x);
        x += temp;
        x = relu.forward(x);
       
        //ATTENTION: following is BLOCK 4

        at::Tensor residual50(x.clone());
        temp = residual50.to(at::kCUDA);
        x = cv_c_100(x);
        x = relu.forward(x);
        x = cv_a_50.forward(x);
        x = relu.forward(x);
        x = cv_c_101.forward(x);
        temp = cv_b_3.forward(temp);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual51(x.clone());
        temp = residual51.to(at::kCUDA);
        x = cv_c_102.forward(x);
        x = relu.forward(x);
        x = cv_a_51.forward(x);
        x = relu.forward(x);
        x = cv_c_103.forward(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual52(x.clone());
        temp = residual52.to(at::kCUDA);
        x = cv_c_104.forward(x);
        x = relu.forward(x);
        x = cv_a_52.forward(x);
        x = relu.forward(x);
        x = cv_c_105.forward(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual53(x.clone());
        temp = residual53.to(at::kCUDA);
        x = cv_c_106(x);
        x = relu.forward(x);
        x = cv_a_53(x);
        x = relu.forward(x);
        x = cv_c_107(x);
        x += temp;
        x = relu.forward(x);

        x = avg0.forward(x);
        x = x.view({x.sizes()[0], -1});
        x = linear0.forward(x);
        return x;
    }
    void morphpara(){
        // nothing to do in warm up
    }
};
//tag4

class resnet_gpu_part1 : public resnet_part
{
public:
    torch::nn::Conv2d c0;
    operator4 mxp2d0;
    operator1 relu;
    operator8 cv_c_0;
    operator2 cv_a_0;
    operator8 cv_c_1;
    operator3 cv_b_0;
// tag1
    resnet_gpu_part1():
        c0(conv_options(3, 64, 7, 2, 3)),
        mxp2d0(3, 1, 2),
        cv_a_0(64, 64, 3, 1, 1),
        cv_b_0(64, 256, 1, 1),
        cv_c_0(64, 64, 1),
        cv_c_1(64, 256, 1)
        {// tag2
            c0->to(at::kCUDA);
            mxp2d0.to(at::kCUDA);
            cv_a_0.to(at::kCUDA);
            cv_b_0.to(at::kCUDA);
            cv_c_0.to(at::kCUDA);
            cv_c_1.to(at::kCUDA);
            relu.to(at::kCUDA);
        }
    torch::Tensor forward(torch::Tensor x)
    {// tag3
        torch::Tensor temp;
        x = c0(x);
        x = relu.forward(x);
        x = mxp2d0.forward(x);
        at::Tensor residual0(x.clone());
        temp = residual0.to(at::kCUDA);
        x = cv_c_0.forward(x);
        x = relu.forward(x);
        x = cv_a_0.forward(x);
        x = relu.forward(x);
        x = cv_c_1.forward(x);
        temp = cv_b_0.forward(temp);     
        x += temp;
        x = relu.forward(x);
        // std::cout<<"p1 size "<<x.sizes()<<std::endl;
        return x;
    }
    void morphpara(){
        c0->weight = c0->weight * scalar;
        cv_a_0.morph(); // struct use inner func
        cv_b_0.morph();
        cv_c_0.morph();
        cv_c_1.morph();
    }
};
//tag5

class resnet_gpu_part2 : public resnet_part
{
public:
    torch::nn::Conv2d c0;
    operator4 mxp2d0;
    operator5 avg0;
    operator7 linear0;
    operator7 linear1;
    operator7 linear2;
    operator1 relu;
    operator2 cv_a_0;
    operator2 cv_a_1;
    operator2 cv_a_2;
    torch::nn::Conv2d cv_a_3;
    operator2 cv_a_4;
    operator2 cv_a_5;
    operator2 cv_a_6;
    operator2 cv_a_7;
    operator2 cv_a_8;
    operator2 cv_a_9;
    operator2 cv_a_10;
    operator2 cv_a_11;
    torch::nn::Conv2d cv_a_12;
    operator2 cv_a_13;
    operator2 cv_a_14;
    operator2 cv_a_15;
    operator2 cv_a_16;
    operator2 cv_a_17;
    operator2 cv_a_18;
    operator2 cv_a_19;
    operator2 cv_a_20;
    operator2 cv_a_21;
    operator2 cv_a_22;
    operator2 cv_a_23;
    torch::nn::Conv2d cv_a_24;
    operator2 cv_a_25;
    torch::nn::Conv2d cv_a_26;
    operator2 cv_a_27;
    torch::nn::Conv2d cv_a_28;
    operator2 cv_a_29;
    torch::nn::Conv2d cv_a_30;
    operator2 cv_a_31;
    torch::nn::Conv2d cv_a_32;
    operator2 cv_a_33;
    operator2 cv_a_34;
    operator2 cv_a_35;
    operator2 cv_a_36;
    operator2 cv_a_37;
    operator2 cv_a_38;
    operator2 cv_a_39;
    operator2 cv_a_40;
    operator2 cv_a_41;
    operator2 cv_a_42;
    operator2 cv_a_43;
    operator2 cv_a_44;
    operator2 cv_a_45;
    torch::nn::Conv2d cv_a_46;
    operator2 cv_a_47;
    operator2 cv_a_48;
    operator2 cv_a_49;
    operator2 cv_a_50;
    operator2 cv_a_51;
    operator2 cv_a_52;
    torch::nn::Conv2d cv_a_53;
    operator3 cv_b_0;
    operator3 cv_b_1;
    operator3 cv_b_2;
    operator3 cv_b_3;
    operator8 cv_c_0;
    operator8 cv_c_1;
    operator8 cv_c_2;
    operator8 cv_c_3;
    torch::nn::Conv2d cv_c_4;
    torch::nn::Conv2d cv_c_5;
    operator8 cv_c_6;
    operator8 cv_c_7;
    operator8 cv_c_8;
    operator8 cv_c_9;
    operator8 cv_c_10;
    operator8 cv_c_11;
    operator8 cv_c_12;
    torch::nn::Conv2d cv_c_13;
    operator8 cv_c_14;
    torch::nn::Conv2d cv_c_15;
    operator8 cv_c_16;
    torch::nn::Conv2d cv_c_17;
    torch::nn::Conv2d cv_c_18;
    operator8 cv_c_19;
    operator8 cv_c_20;
    operator8 cv_c_21;
    operator8 cv_c_22;
    torch::nn::Conv2d cv_c_23;
    operator8 cv_c_24;
    operator8 cv_c_25;
    operator8 cv_c_26;
    operator8 cv_c_27;
    torch::nn::Conv2d cv_c_28;
    operator8 cv_c_29;
    operator8 cv_c_30;
    operator8 cv_c_31;
    operator8 cv_c_32;
    operator8 cv_c_33;
    operator8 cv_c_34;
    operator8 cv_c_35;
    torch::nn::Conv2d cv_c_36;
    torch::nn::Conv2d cv_c_37;
    operator8 cv_c_38;
    operator8 cv_c_39;
    operator8 cv_c_40;
    torch::nn::Conv2d cv_c_41;
    operator8 cv_c_42;
    torch::nn::Conv2d cv_c_43;
    operator8 cv_c_44;
    operator8 cv_c_45;
    operator8 cv_c_46;
    torch::nn::Conv2d cv_c_47;
    torch::nn::Conv2d cv_c_48;
    operator8 cv_c_49;
    operator8 cv_c_50;
    operator8 cv_c_51;
    operator8 cv_c_52;
    operator8 cv_c_53;
    torch::nn::Conv2d cv_c_54;
    operator8 cv_c_55;
    operator8 cv_c_56;
    operator8 cv_c_57;
    operator8 cv_c_58;
    torch::nn::Conv2d cv_c_59;
    torch::nn::Conv2d cv_c_60;
    operator8 cv_c_61;
    operator8 cv_c_62;
    operator8 cv_c_63;
    torch::nn::Conv2d cv_c_64;
    operator8 cv_c_65;
    operator8 cv_c_66;
    torch::nn::Conv2d cv_c_67;
    operator8 cv_c_68;
    torch::nn::Conv2d cv_c_69;
    torch::nn::Conv2d cv_c_70;
    operator8 cv_c_71;
    torch::nn::Conv2d cv_c_72;
    operator8 cv_c_73;
    torch::nn::Conv2d cv_c_74;
    operator8 cv_c_75;
    torch::nn::Conv2d cv_c_76;
    operator8 cv_c_77;
    torch::nn::Conv2d cv_c_78;
    operator8 cv_c_79;
    operator8 cv_c_80;
    operator8 cv_c_81;
    operator8 cv_c_82;
    operator8 cv_c_83;
    operator8 cv_c_84;
    operator8 cv_c_85;
    operator8 cv_c_86;
    operator8 cv_c_87;
    operator8 cv_c_88;
    operator8 cv_c_89;
    operator8 cv_c_90;
    torch::nn::Conv2d cv_c_91;
    operator8 cv_c_92;
    operator8 cv_c_93;
    operator8 cv_c_94;
    operator8 cv_c_95;
    torch::nn::Conv2d cv_c_96;
    operator8 cv_c_97;
    operator8 cv_c_98;
    torch::nn::Conv2d cv_c_99;
    torch::nn::Conv2d cv_c_100;
    operator8 cv_c_101;
    operator8 cv_c_102;
    operator8 cv_c_103;
    operator8 cv_c_104;
    operator8 cv_c_105;
    torch::nn::Conv2d cv_c_106;
    torch::nn::Conv2d cv_c_107;
// tag1
    resnet_gpu_part2():
        c0(conv_options(3, 64, 7, 2, 3)),
        mxp2d0(3, 1, 2),
        avg0(7, 1),
        linear0(2048, 32768),
        linear1(32768, 16384),
        linear2(16384, 1000),
        cv_a_0(64, 64, 3, 1, 1),
        cv_a_1(64, 64, 3, 1, 1),
        cv_a_2(64, 64, 3, 1, 1),
        cv_a_3(conv_options(64, 64, 3, 1, 1)),
        cv_a_4(128, 128, 3, 2, 1),
        cv_a_5(128, 128, 3, 1, 1),
        cv_a_6(128, 128, 3, 1, 1),
        cv_a_7(128, 128, 3, 1, 1),
        cv_a_8(128, 128, 3, 1, 1),
        cv_a_9(128, 128, 3, 1, 1),
        cv_a_10(128, 128, 3, 1, 1),
        cv_a_11(128, 128, 3, 1, 1),
        cv_a_12(conv_options(128, 128, 3, 1, 1)),
        cv_a_13(256, 256, 3, 2, 1),
        cv_a_14(256, 256, 3, 1, 1),
        cv_a_15(256, 256, 3, 1, 1),
        cv_a_16(256, 256, 3, 1, 1),
        cv_a_17(256, 256, 3, 1, 1),
        cv_a_18(256, 256, 3, 1, 1),
        cv_a_19(256, 256, 3, 1, 1),
        cv_a_20(256, 256, 3, 1, 1),
        cv_a_21(256, 256, 3, 1, 1),
        cv_a_22(256, 256, 3, 1, 1),
        cv_a_23(256, 256, 3, 1, 1),
        cv_a_24(conv_options(256, 256, 3, 1, 1)),
        cv_a_25(256, 256, 3, 1, 1),
        cv_a_26(conv_options(256, 256, 3, 1, 1)),
        cv_a_27(256, 256, 3, 1, 1),
        cv_a_28(conv_options(256, 256, 3, 1, 1)),
        cv_a_29(256, 256, 3, 1, 1),
        cv_a_30(conv_options(256, 256, 3, 1, 1)),
        cv_a_31(256, 256, 3, 1, 1),
        cv_a_32(conv_options(256, 256, 3, 1, 1)),
        cv_a_33(256, 256, 3, 1, 1),
        cv_a_34(256, 256, 3, 1, 1),
        cv_a_35(256, 256, 3, 1, 1),
        cv_a_36(256, 256, 3, 1, 1),
        cv_a_37(256, 256, 3, 1, 1),
        cv_a_38(256, 256, 3, 1, 1),
        cv_a_39(256, 256, 3, 1, 1),
        cv_a_40(256, 256, 3, 1, 1),
        cv_a_41(256, 256, 3, 1, 1),
        cv_a_42(256, 256, 3, 1, 1),
        cv_a_43(256, 256, 3, 1, 1),
        cv_a_44(256, 256, 3, 1, 1),
        cv_a_45(256, 256, 3, 1, 1),
        cv_a_46(conv_options(256, 256, 3, 1, 1)),
        cv_a_47(256, 256, 3, 1, 1),
        cv_a_48(256, 256, 3, 1, 1),
        cv_a_49(256, 256, 3, 1, 1),
        cv_a_50(512, 512, 3, 2, 1),
        cv_a_51(512, 512, 3, 1, 1),
        cv_a_52(512, 512, 3, 1, 1),
        cv_a_53(conv_options(512, 512, 3, 1, 1)),
        cv_b_0(64, 256, 1, 1),
        cv_b_1(256, 512, 1, 2),
        cv_b_2(512, 1024, 1, 2),
        cv_b_3(1024, 2048, 1, 2),
        cv_c_0(64, 64, 1),
        cv_c_1(64, 256, 1),
        cv_c_2(256, 64, 1),
        cv_c_3(64, 256, 1),
        cv_c_4(conv_options(256, 64, 1)),
        cv_c_5(conv_options(64, 256, 1)),
        cv_c_6(256, 64, 1),
        cv_c_7(64, 256, 1),
        cv_c_8(256, 128, 1),
        cv_c_9(128, 512, 1),
        cv_c_10(512, 128, 1),
        cv_c_11(128, 512, 1),
        cv_c_12(512, 128, 1),
        cv_c_13(conv_options(128, 512, 1)),
        cv_c_14(512, 128, 1),
        cv_c_15(conv_options(128, 512, 1)),
        cv_c_16(512, 128, 1),
        cv_c_17(conv_options(128, 512, 1)),
        cv_c_18(conv_options(512, 128, 1)),
        cv_c_19(128, 512, 1),
        cv_c_20(512, 128, 1),
        cv_c_21(128, 512, 1),
        cv_c_22(512, 128, 1),
        cv_c_23(conv_options(128, 512, 1)),
        cv_c_24(512, 128, 1),
        cv_c_25(128, 512, 1),
        cv_c_26(512, 256, 1),
        cv_c_27(256, 1024, 1),
        cv_c_28(conv_options(1024, 256, 1)),
        cv_c_29(256, 1024, 1),
        cv_c_30(1024, 256, 1),
        cv_c_31(256, 1024, 1),
        cv_c_32(1024, 256, 1),
        cv_c_33(256, 1024, 1),
        cv_c_34(1024, 256, 1),
        cv_c_35(256, 1024, 1),
        cv_c_36(conv_options(1024, 256, 1)),
        cv_c_37(conv_options(256, 1024, 1)),
        cv_c_38(1024, 256, 1),
        cv_c_39(256, 1024, 1),
        cv_c_40(1024, 256, 1),
        cv_c_41(conv_options(256, 1024, 1)),
        cv_c_42(1024, 256, 1),
        cv_c_43(conv_options(256, 1024, 1)),
        cv_c_44(1024, 256, 1),
        cv_c_45(256, 1024, 1),
        cv_c_46(1024, 256, 1),
        cv_c_47(conv_options(256, 1024, 1)),
        cv_c_48(conv_options(1024, 256, 1)),
        cv_c_49(256, 1024, 1),
        cv_c_50(1024, 256, 1),
        cv_c_51(256, 1024, 1),
        cv_c_52(1024, 256, 1),
        cv_c_53(256, 1024, 1),
        cv_c_54(conv_options(1024, 256, 1)),
        cv_c_55(256, 1024, 1),
        cv_c_56(1024, 256, 1),
        cv_c_57(256, 1024, 1),
        cv_c_58(1024, 256, 1),
        cv_c_59(conv_options(256, 1024, 1)),
        cv_c_60(conv_options(1024, 256, 1)),
        cv_c_61(256, 1024, 1),
        cv_c_62(1024, 256, 1),
        cv_c_63(256, 1024, 1),
        cv_c_64(conv_options(1024, 256, 1)),
        cv_c_65(256, 1024, 1),
        cv_c_66(1024, 256, 1),
        cv_c_67(conv_options(256, 1024, 1)),
        cv_c_68(1024, 256, 1),
        cv_c_69(conv_options(256, 1024, 1)),
        cv_c_70(conv_options(1024, 256, 1)),
        cv_c_71(256, 1024, 1),
        cv_c_72(conv_options(1024, 256, 1)),
        cv_c_73(256, 1024, 1),
        cv_c_74(conv_options(1024, 256, 1)),
        cv_c_75(256, 1024, 1),
        cv_c_76(conv_options(1024, 256, 1)),
        cv_c_77(256, 1024, 1),
        cv_c_78(conv_options(1024, 256, 1)),
        cv_c_79(256, 1024, 1),
        cv_c_80(1024, 256, 1),
        cv_c_81(256, 1024, 1),
        cv_c_82(1024, 256, 1),
        cv_c_83(256, 1024, 1),
        cv_c_84(1024, 256, 1),
        cv_c_85(256, 1024, 1),
        cv_c_86(1024, 256, 1),
        cv_c_87(256, 1024, 1),
        cv_c_88(1024, 256, 1),
        cv_c_89(256, 1024, 1),
        cv_c_90(1024, 256, 1),
        cv_c_91(conv_options(256, 1024, 1)),
        cv_c_92(1024, 256, 1),
        cv_c_93(256, 1024, 1),
        cv_c_94(1024, 256, 1),
        cv_c_95(256, 1024, 1),
        cv_c_96(conv_options(1024, 256, 1)),
        cv_c_97(256, 1024, 1),
        cv_c_98(1024, 256, 1),
        cv_c_99(conv_options(256, 1024, 1)),
        cv_c_100(conv_options(1024, 512, 1)),
        cv_c_101(512, 2048, 1),
        cv_c_102(2048, 512, 1),
        cv_c_103(512, 2048, 1),
        cv_c_104(2048, 512, 1),
        cv_c_105(512, 2048, 1),
        cv_c_106(conv_options(2048, 512, 1)),
        cv_c_107(conv_options(512, 2048, 1))
        {// tag2
            c0->to(at::kCUDA);
            mxp2d0.to(at::kCUDA);
            avg0.to(at::kCUDA);
            linear0.to(at::kCUDA);
            linear1.to(at::kCUDA);
            linear2.to(at::kCUDA);
            cv_a_0.to(at::kCUDA);
            cv_a_1.to(at::kCUDA);
            cv_a_2.to(at::kCUDA);
            cv_a_3->to(at::kCUDA);
            cv_a_4.to(at::kCUDA);
            cv_a_5.to(at::kCUDA);
            cv_a_6.to(at::kCUDA);
            cv_a_7.to(at::kCUDA);
            cv_a_8.to(at::kCUDA);
            cv_a_9.to(at::kCUDA);
            cv_a_10.to(at::kCUDA);
            cv_a_11.to(at::kCUDA);
            cv_a_12->to(at::kCUDA);
            cv_a_13.to(at::kCUDA);
            cv_a_14.to(at::kCUDA);
            cv_a_15.to(at::kCUDA);
            cv_a_16.to(at::kCUDA);
            cv_a_17.to(at::kCUDA);
            cv_a_18.to(at::kCUDA);
            cv_a_19.to(at::kCUDA);
            cv_a_20.to(at::kCUDA);
            cv_a_21.to(at::kCUDA);
            cv_a_22.to(at::kCUDA);
            cv_a_23.to(at::kCUDA);
            cv_a_24->to(at::kCUDA);
            cv_a_25.to(at::kCUDA);
            cv_a_26->to(at::kCUDA);
            cv_a_27.to(at::kCUDA);
            cv_a_28->to(at::kCUDA);
            cv_a_29.to(at::kCUDA);
            cv_a_30->to(at::kCUDA);
            cv_a_31.to(at::kCUDA);
            cv_a_32->to(at::kCUDA);
            cv_a_33.to(at::kCUDA);
            cv_a_34.to(at::kCUDA);
            cv_a_35.to(at::kCUDA);
            cv_a_36.to(at::kCUDA);
            cv_a_37.to(at::kCUDA);
            cv_a_38.to(at::kCUDA);
            cv_a_39.to(at::kCUDA);
            cv_a_40.to(at::kCUDA);
            cv_a_41.to(at::kCUDA);
            cv_a_42.to(at::kCUDA);
            cv_a_43.to(at::kCUDA);
            cv_a_44.to(at::kCUDA);
            cv_a_45.to(at::kCUDA);
            cv_a_46->to(at::kCUDA);
            cv_a_47.to(at::kCUDA);
            cv_a_48.to(at::kCUDA);
            cv_a_49.to(at::kCUDA);
            cv_a_50.to(at::kCUDA);
            cv_a_51.to(at::kCUDA);
            cv_a_52.to(at::kCUDA);
            cv_a_53->to(at::kCUDA);
            cv_b_0.to(at::kCUDA);
            cv_b_1.to(at::kCUDA);
            cv_b_2.to(at::kCUDA);
            cv_b_3.to(at::kCUDA);
            cv_c_0.to(at::kCUDA);
            cv_c_1.to(at::kCUDA);
            cv_c_2.to(at::kCUDA);
            cv_c_3.to(at::kCUDA);
            cv_c_4->to(at::kCUDA);
            cv_c_5->to(at::kCUDA);            
            cv_c_6.to(at::kCUDA);
            cv_c_7.to(at::kCUDA);
            cv_c_8.to(at::kCUDA);
            cv_c_9.to(at::kCUDA);
            cv_c_10.to(at::kCUDA);
            cv_c_11.to(at::kCUDA);
            cv_c_12.to(at::kCUDA);
            cv_c_13->to(at::kCUDA);
            cv_c_14.to(at::kCUDA);
            cv_c_15->to(at::kCUDA);
            cv_c_16.to(at::kCUDA);
            cv_c_17->to(at::kCUDA);
            cv_c_18->to(at::kCUDA);
            cv_c_19.to(at::kCUDA);
            cv_c_20.to(at::kCUDA);
            cv_c_21.to(at::kCUDA);
            cv_c_22.to(at::kCUDA);
            cv_c_23->to(at::kCUDA);
            cv_c_24.to(at::kCUDA);
            cv_c_25.to(at::kCUDA);
            cv_c_26.to(at::kCUDA);
            cv_c_27.to(at::kCUDA);
            cv_c_28->to(at::kCUDA);
            cv_c_29.to(at::kCUDA);
            cv_c_30.to(at::kCUDA);
            cv_c_31.to(at::kCUDA);
            cv_c_32.to(at::kCUDA);
            cv_c_33.to(at::kCUDA);
            cv_c_34.to(at::kCUDA);
            cv_c_35.to(at::kCUDA);
            cv_c_36->to(at::kCUDA);
            cv_c_37->to(at::kCUDA);
            cv_c_38.to(at::kCUDA);
            cv_c_39.to(at::kCUDA);
            cv_c_40.to(at::kCUDA);
            cv_c_41->to(at::kCUDA);
            cv_c_42.to(at::kCUDA);
            cv_c_43->to(at::kCUDA);
            cv_c_44.to(at::kCUDA);
            cv_c_45.to(at::kCUDA);
            cv_c_46.to(at::kCUDA);
            cv_c_47->to(at::kCUDA);
            cv_c_48->to(at::kCUDA);
            cv_c_49.to(at::kCUDA);
            cv_c_50.to(at::kCUDA);
            cv_c_51.to(at::kCUDA);
            cv_c_52.to(at::kCUDA);
            cv_c_53.to(at::kCUDA);
            cv_c_54->to(at::kCUDA);
            cv_c_55.to(at::kCUDA);
            cv_c_56.to(at::kCUDA);
            cv_c_57.to(at::kCUDA);
            cv_c_58.to(at::kCUDA);
            cv_c_59->to(at::kCUDA);
            cv_c_60->to(at::kCUDA);
            cv_c_61.to(at::kCUDA);
            cv_c_62.to(at::kCUDA);
            cv_c_63.to(at::kCUDA);
            cv_c_64->to(at::kCUDA);
            cv_c_65.to(at::kCUDA);
            cv_c_66.to(at::kCUDA);
            cv_c_67->to(at::kCUDA);
            cv_c_68.to(at::kCUDA);
            cv_c_69->to(at::kCUDA);
            cv_c_70->to(at::kCUDA);
            cv_c_71.to(at::kCUDA);
            cv_c_72->to(at::kCUDA);
            cv_c_73.to(at::kCUDA);
            cv_c_74->to(at::kCUDA);
            cv_c_75.to(at::kCUDA);
            cv_c_76->to(at::kCUDA);
            cv_c_77.to(at::kCUDA);
            cv_c_78->to(at::kCUDA);
            cv_c_79.to(at::kCUDA);
            cv_c_80.to(at::kCUDA);
            cv_c_81.to(at::kCUDA);
            cv_c_82.to(at::kCUDA);
            cv_c_83.to(at::kCUDA);
            cv_c_84.to(at::kCUDA);
            cv_c_85.to(at::kCUDA);
            cv_c_86.to(at::kCUDA);
            cv_c_87.to(at::kCUDA);
            cv_c_88.to(at::kCUDA);
            cv_c_89.to(at::kCUDA);
            cv_c_90.to(at::kCUDA);
            cv_c_91->to(at::kCUDA);
            cv_c_92.to(at::kCUDA);
            cv_c_93.to(at::kCUDA);
            cv_c_94.to(at::kCUDA);
            cv_c_95.to(at::kCUDA);
            cv_c_96->to(at::kCUDA);
            cv_c_97.to(at::kCUDA);
            cv_c_98.to(at::kCUDA);
            cv_c_99->to(at::kCUDA);
            cv_c_100->to(at::kCUDA);
            cv_c_101.to(at::kCUDA);
            cv_c_102.to(at::kCUDA);
            cv_c_103.to(at::kCUDA);
            cv_c_104.to(at::kCUDA);
            cv_c_105.to(at::kCUDA);
            cv_c_106->to(at::kCUDA);
            cv_c_107->to(at::kCUDA);
            relu.to(at::kCUDA);
        }

    torch::Tensor forward(torch::Tensor x)
    {// tag3
        torch::Tensor temp;
        at::Tensor residual2(x.clone());
        temp = residual2.to(at::kCUDA);
        x = cv_c_4(x);
        x = relu.forward(x);
        x = cv_a_2.forward(x);
        x = relu.forward(x);
        x = cv_c_5(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual3(x.clone());
        temp = residual3.to(at::kCUDA);
        x = cv_c_6.forward(x);
        x = relu.forward(x);
        x = cv_a_3(x);
        x = relu.forward(x);
        x = cv_c_7.forward(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual4(x.clone());
        temp = residual4.to(at::kCUDA);
        x = cv_c_8.forward(x);
        x = relu.forward(x);
        x = cv_a_4.forward(x);
        x = relu.forward(x);
        x = cv_c_9.forward(x);
        temp = cv_b_1.forward(temp);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual5(x.clone());
        temp = residual5.to(at::kCUDA);
        x = cv_c_10.forward(x);
        x = relu.forward(x);
        x = cv_a_5.forward(x);
        x = relu.forward(x);
        x = cv_c_11.forward(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual6(x.clone());
        temp = residual6.to(at::kCUDA);
        x = cv_c_12.forward(x);
        x = relu.forward(x);
        x = cv_a_6.forward(x);
        x = relu.forward(x);
        x = cv_c_13(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual7(x.clone());
        temp = residual7.to(at::kCUDA);
        x = cv_c_14.forward(x);
        x = relu.forward(x);
        x = cv_a_7.forward(x);
        x = relu.forward(x);
        x = cv_c_15(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual8(x.clone());
        temp = residual8.to(at::kCUDA);
        x = cv_c_16.forward(x);
        x = relu.forward(x);
        x = cv_a_8.forward(x);
        x = relu.forward(x);
        x = cv_c_17(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual9(x.clone());
        temp = residual9.to(at::kCUDA);
        x = cv_c_18(x);
        x = relu.forward(x);
        x = cv_a_9.forward(x);
        x = relu.forward(x);
        x = cv_c_19.forward(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual10(x.clone());
        temp = residual10.to(at::kCUDA);
        x = cv_c_20.forward(x);
        x = relu.forward(x);
        x = cv_a_10.forward(x);
        x = relu.forward(x);
        x = cv_c_21.forward(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual11(x.clone());
        temp = residual11.to(at::kCUDA);
        x = cv_c_22.forward(x);
        x = relu.forward(x);
        x = cv_a_11.forward(x);
        x = relu.forward(x);
        x = cv_c_23(x);
        x += temp;
        x = relu.forward(x);
        
        at::Tensor residual12(x.clone());
        temp = residual12.to(at::kCUDA);
        x = cv_c_24.forward(x);
        x = relu.forward(x);
        x = cv_a_12(x);
        x = relu.forward(x);
        x = cv_c_25.forward(x);
        x += temp;
        x = relu.forward(x);
       
        //ATTENTION: following is BLOCK 3

        at::Tensor residual13(x.clone());
        temp = residual13.to(at::kCUDA);
        x = cv_c_26.forward(x);
        x = relu.forward(x);
        x = cv_a_13.forward(x);
        x = relu.forward(x);
        x = cv_c_27.forward(x);
        temp = cv_b_2.forward(temp);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual14(x.clone());
        temp = residual14.to(at::kCUDA);
        x = cv_c_28(x);
        x = relu.forward(x);
        x = cv_a_14.forward(x);
        x = relu.forward(x);
        x = cv_c_29.forward(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual15(x.clone());
        temp = residual15.to(at::kCUDA);
        x = cv_c_30.forward(x);
        x = relu.forward(x);
        x = cv_a_15.forward(x);
        x = relu.forward(x);
        x = cv_c_31.forward(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual16(x.clone());
        temp = residual16.to(at::kCUDA);
        x = cv_c_32.forward(x);
        x = relu.forward(x);
        x = cv_a_16.forward(x);
        x = relu.forward(x);
        x = cv_c_33.forward(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual17(x.clone());
        temp = residual17.to(at::kCUDA);
        x = cv_c_34.forward(x);
        x = relu.forward(x);
        x = cv_a_17.forward(x);
        x = relu.forward(x);
        x = cv_c_35.forward(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual18(x.clone());
        temp = residual18.to(at::kCUDA);
        x = cv_c_36(x);
        x = relu.forward(x);
        x = cv_a_18.forward(x);
        x = relu.forward(x);
        x = cv_c_37(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual19(x.clone());
        temp = residual19.to(at::kCUDA);
        x = cv_c_38.forward(x);
        x = relu.forward(x);
        x = cv_a_19.forward(x);
        x = relu.forward(x);
        x = cv_c_39.forward(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual20(x.clone());
        temp = residual20.to(at::kCUDA);
        x = cv_c_40.forward(x);
        x = relu.forward(x);
        x = cv_a_20.forward(x);
        x = relu.forward(x);
        x = cv_c_41(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual21(x.clone());
        temp = residual21.to(at::kCUDA);
        x = cv_c_42.forward(x);
        x = relu.forward(x);
        x = cv_a_21.forward(x);
        x = relu.forward(x);
        x = cv_c_43(x);
        x += temp;
        x = relu.forward(x);


        at::Tensor residual22(x.clone());
        temp = residual22.to(at::kCUDA);
        x = cv_c_44.forward(x);
        x = relu.forward(x);
        x = cv_a_22.forward(x);
        x = relu.forward(x);
        x = cv_c_45.forward(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual23(x.clone());
        temp = residual23.to(at::kCUDA);
        x = cv_c_46.forward(x);
        x = relu.forward(x);
        x = cv_a_23.forward(x);
        x = relu.forward(x);
        x = cv_c_47(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual24(x.clone());
        temp = residual24.to(at::kCUDA);
        x = cv_c_48(x);
        x = relu.forward(x);
        x = cv_a_24(x);
        x = relu.forward(x);
        x = cv_c_49.forward(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual25(x.clone());
        temp = residual25.to(at::kCUDA);
        x = cv_c_50.forward(x);
        x = relu.forward(x);
        x = cv_a_25.forward(x);
        x = relu.forward(x);
        x = cv_c_51.forward(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual26(x.clone());
        temp = residual26.to(at::kCUDA);
        x = cv_c_52.forward(x);
        x = relu.forward(x);
        x = cv_a_26(x);
        x = relu.forward(x);
        x = cv_c_53.forward(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual27(x.clone());
        temp = residual27.to(at::kCUDA);
        x = cv_c_54(x);
        x = relu.forward(x);
        x = cv_a_27.forward(x);
        x = relu.forward(x);
        x = cv_c_55.forward(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual28(x.clone());
        temp = residual28.to(at::kCUDA);
        x = cv_c_56.forward(x);
        x = relu.forward(x);
        x = cv_a_28(x);
        x = relu.forward(x);
        x = cv_c_57.forward(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual29(x.clone());
        temp = residual29.to(at::kCUDA);
        x = cv_c_58.forward(x);
        x = relu.forward(x);
        x = cv_a_29.forward(x);
        x = relu.forward(x);
        x = cv_c_59(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual30(x.clone());
        temp = residual30.to(at::kCUDA);
        x = cv_c_60(x);
        x = relu.forward(x);
        x = cv_a_30(x);
        x = relu.forward(x);
        x = cv_c_61.forward(x);
        x += temp;
        x = relu.forward(x);
 
        at::Tensor residual31(x.clone());
        temp = residual31.to(at::kCUDA);
        x = cv_c_62.forward(x);
        x = relu.forward(x);
        x = cv_a_31.forward(x);
        x = relu.forward(x);
        x = cv_c_63.forward(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual32(x.clone());
        temp = residual32.to(at::kCUDA);
        x = cv_c_64(x);
        x = relu.forward(x);
        x = cv_a_32(x);
        x = relu.forward(x);
        x = cv_c_65.forward(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual33(x.clone());
        temp = residual33.to(at::kCUDA);
        x = cv_c_66.forward(x);
        x = relu.forward(x);
        x = cv_a_33.forward(x);
        x = relu.forward(x);
        x = cv_c_67(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual34(x.clone());
        temp = residual34.to(at::kCUDA);
        x = cv_c_68.forward(x);
        x = relu.forward(x);
        x = cv_a_34.forward(x);
        x = relu.forward(x);
        x = cv_c_69(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual35(x.clone());
        temp = residual35.to(at::kCUDA);
        x = cv_c_70(x);
        x = relu.forward(x);
        x = cv_a_35.forward(x);
        x = relu.forward(x);
        x = cv_c_71.forward(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual36(x.clone());
        temp = residual36.to(at::kCUDA);
        x = cv_c_72(x);
        x = relu.forward(x);
        x = cv_a_36.forward(x);
        x = relu.forward(x);
        x = cv_c_73.forward(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual37(x.clone());
        temp = residual37.to(at::kCUDA);
        x = cv_c_74(x);
        x = relu.forward(x);
        x = cv_a_37.forward(x);
        x = relu.forward(x);
        x = cv_c_75.forward(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual38(x.clone());
        temp = residual38.to(at::kCUDA);
        x = cv_c_76(x);
        x = relu.forward(x);
        x = cv_a_38.forward(x);
        x = relu.forward(x);
        x = cv_c_77.forward(x);
        x += temp;
        x = relu.forward(x);


        at::Tensor residual39(x.clone());
        temp = residual39.to(at::kCUDA);
        x = cv_c_78(x);
        x = relu.forward(x);
        x = cv_a_39.forward(x);
        x = relu.forward(x);
        x = cv_c_79.forward(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual40(x.clone());
        temp = residual40.to(at::kCUDA);
        x = cv_c_80.forward(x);
        x = relu.forward(x);
        x = cv_a_40.forward(x);
        x = relu.forward(x);
        x = cv_c_81.forward(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual41(x.clone());
        temp = residual41.to(at::kCUDA);
        x = cv_c_82.forward(x);
        x = relu.forward(x);
        x = cv_a_41.forward(x);
        x = relu.forward(x);
        x = cv_c_83.forward(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual42(x.clone());
        temp = residual42.to(at::kCUDA);
        x = cv_c_84.forward(x);
        x = relu.forward(x);
        x = cv_a_42.forward(x);
        x = relu.forward(x);
        x = cv_c_85.forward(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual43(x.clone());
        temp = residual43.to(at::kCUDA);
        x = cv_c_86.forward(x);
        x = relu.forward(x);
        x = cv_a_43.forward(x);
        x = relu.forward(x);
        x = cv_c_87.forward(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual44(x.clone());
        temp = residual44.to(at::kCUDA);
        x = cv_c_88.forward(x);
        x = relu.forward(x);
        x = cv_a_44.forward(x);
        x = relu.forward(x);
        x = cv_c_89.forward(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual45(x.clone());
        temp = residual45.to(at::kCUDA);
        x = cv_c_90.forward(x);
        x = relu.forward(x);
        x = cv_a_45.forward(x);
        x = relu.forward(x);
        x = cv_c_91(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual46(x.clone());
        temp = residual46.to(at::kCUDA);
        x = cv_c_92.forward(x);
        x = relu.forward(x);
        x = cv_a_46(x);
        x = relu.forward(x);
        x = cv_c_93.forward(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual47(x.clone());
        temp = residual47.to(at::kCUDA);
        x = cv_c_94.forward(x);
        x = relu.forward(x);
        x = cv_a_47.forward(x);
        x = relu.forward(x);
        x = cv_c_95.forward(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual48(x.clone());
        temp = residual48.to(at::kCUDA);
        x = cv_c_96(x);
        x = relu.forward(x);
        x = cv_a_48.forward(x);
        x = relu.forward(x);
        x = cv_c_97.forward(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual49(x.clone());
        temp = residual49.to(at::kCUDA);
        x = cv_c_98.forward(x);
        x = relu.forward(x);
        x = cv_a_49.forward(x);
        x = relu.forward(x);
        x = cv_c_99(x);
        x += temp;
        x = relu.forward(x);
       
        //ATTENTION: following is BLOCK 4

        at::Tensor residual50(x.clone());
        temp = residual50.to(at::kCUDA);
        x = cv_c_100(x);
        x = relu.forward(x);
        x = cv_a_50.forward(x);
        x = relu.forward(x);
        x = cv_c_101.forward(x);
        temp = cv_b_3.forward(temp);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual51(x.clone());
        temp = residual51.to(at::kCUDA);
        x = cv_c_102.forward(x);
        x = relu.forward(x);
        x = cv_a_51.forward(x);
        x = relu.forward(x);
        x = cv_c_103.forward(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual52(x.clone());
        temp = residual52.to(at::kCUDA);
        x = cv_c_104.forward(x);
        x = relu.forward(x);
        x = cv_a_52.forward(x);
        x = relu.forward(x);
        x = cv_c_105.forward(x);
        x += temp;
        x = relu.forward(x);

        at::Tensor residual53(x.clone());
        temp = residual53.to(at::kCUDA);
        x = cv_c_106(x);
        x = relu.forward(x);
        x = cv_a_53(x);
        x = relu.forward(x);
        x = cv_c_107(x);
        x += temp;
        x = relu.forward(x);
        x = avg0.forward(x);
        x = x.view({x.sizes()[0], -1});
        x = linear0.forward(x);
        x = linear1.forward(x);
        x = linear2.forward(x);
        // std::cout<<"p2 size "<<x.sizes()<<std::endl; 
        return x;
    }
    void morphpara(){
        // nothing to do in warm up
    }
};
//tag6

resnet_part* models[] = {
    new resnet_warmup(),
    new resnet_gpu_part1(),
    new resnet_gpu_part2()
};

// Logic and data behind the server's behavior.
class GreeterServiceImpl final : public Greeter::Service {
    std::string prefix;

    Status SayHello(ServerContext* context, const HelloRequest* request,
                    HelloReply* reply) override {
        int tag = request->tag();
        if (tag == 0) {
            std::cout << "[Debug] Receive warming up ..."<< std::endl;
            for (size_t i = 0; i < 10; i++) {
                output = models[tag]->forward(input);
            }
            std::cout << "[Preprocessing phase] Receive TEE signal, GPU is now warming up ..."<< std::endl;
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
    RunServer();
    return 0;
}
