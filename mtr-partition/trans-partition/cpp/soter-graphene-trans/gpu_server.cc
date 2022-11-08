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
int count = 100;
int scalar = 4;
int d_model = 1024;
int nheads = 8;
int d_k = d_model / nheads;
int d_v = d_k;
int sqrt_dk = (int)floor(sqrt((double)d_v));
int d_ff = 2048;
torch::Tensor src = torch::rand( {32, 10, d_model} ).to(at::kCUDA);
torch::Tensor tgt = torch::rand( {32, 20, d_model} ).to(at::kCUDA);
torch::Tensor out;

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
    operator2(int a, int b, int c, int d, int e):
        conv(conv_options(a, b, c, d, e))
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
        mxp2d(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(a).padding(b).stride(c)))
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
        return x;
    }
};


class trans_part : public torch::nn::Module {
    public:
    virtual torch::Tensor forward(torch::Tensor x, torch::Tensor y) = 0;
    virtual void morphpara() = 0;
};

class trans_warmup : public trans_part
{
public:
    operator1 relu;
    operator7 fc0;
    operator7 fc1;
    operator7 fc2;
    operator7 fc3;
    torch::nn::Linear fc4;
    operator7 fc5;
    operator7 fc6;
    operator7 fc7;
    operator7 fc8;
    torch::nn::Linear fc9;
    operator7 fc10;
    operator7 fc11;
    operator7 fc12;
    operator7 fc13;
    torch::nn::Linear fc14;
    operator7 fc15;
    operator7 fc16;
    operator7 fc17;
    torch::nn::Linear fc18;
    operator7 fc19;
    operator7 fc20;
    operator7 fc21;
    operator7 fc22;
    operator7 fc23;
    operator7 fc24;
    torch::nn::Linear fc25;
    operator7 fc26;
    operator7 fc27;
    operator7 fc28;
    operator7 fc29;
    torch::nn::Linear fc30;
    torch::nn::Linear fc31;
    torch::nn::Linear fc32;
    torch::nn::Linear fc33;
    operator7 fc34;
    operator7 fc35;
    operator7 fc36;
    torch::nn::Linear fc37;
    operator7 fc38;
    operator7 fc39;
    operator7 fc40;
    torch::nn::Linear fc41;
    operator7 fc42;
    torch::nn::Linear fc43;
    operator7 fc44;
    torch::nn::Linear fc45;
    operator7 fc46;
    operator7 fc47;
    operator7 fc48;
    torch::nn::Linear fc49;
    operator7 fc50;
    operator7 fc51;
    operator7 fc52;
    operator7 fc53;
    torch::nn::Linear fc54;
    operator7 fc55;
    torch::nn::Linear fc56;
    operator7 fc57;
    operator7 fc58;
    torch::nn::Linear fc59;
    operator7 fc60;
    operator7 fc61;
    operator7 fc62;
    operator7 fc63;
    operator7 fc64;
    operator7 fc65;
    torch::nn::Linear fc66;
    operator7 fc67;
    operator7 fc68;
    operator7 fc69;
    torch::nn::Linear fc70;
    operator7 fc71;
    operator7 fc72;
    operator7 fc73;
    operator7 fc74;
    operator7 fc75;
    operator7 fc76;
    operator7 fc77;
    operator7 fc78;
    operator7 fc79;
    torch::nn::Linear fc80;
    operator7 fc81;
    operator7 fc82;
    operator7 fc83;
    operator7 fc84;
    operator7 fc85;
    operator7 fc86;
    torch::nn::Linear fc87;
    operator7 fc88;
    operator7 fc89;
    torch::nn::Linear fc90;
    operator7 fc91;
    operator7 fc92;
    operator7 fc93;
    torch::nn::Linear fc94;
    torch::nn::Linear fc95;
    operator7 fc96;
    operator7 fc97;
    operator7 fc98;
    operator7 fc99;
    operator7 fc100;
    torch::nn::Linear fc101;
    torch::nn::Linear fc102;
    operator7 fc103;
    operator7 fc104;
    operator7 fc105;
    operator7 fc106;
    operator7 fc107;
    operator7 fc108;
    operator7 fc109;
    operator7 fc110;
    operator7 fc111;
    operator7 fc112;
    operator7 fc113;
    operator7 fc114;
    operator7 fc115;
    operator7 fc116;
    operator7 fc117;
    operator7 fc118;
    operator7 fc119;
    operator7 fc120;
    operator7 fc121;
    operator7 fc122;
    operator7 fc123;
    operator7 fc124;
    operator7 fc125;
    operator7 fc126;
    torch::nn::Linear fc127;
    torch::nn::Linear fc128;
    torch::nn::Linear fc129;
    operator7 fc130;
    operator7 fc131;
    operator7 fc132;
    operator7 fc133;
    operator7 fc134;
    operator7 fc135;
    operator7 fc136;
    operator7 fc137;
    operator7 fc138;
    operator7 fc139;
    operator7 fc140;
    torch::nn::Linear fc141;
    torch::nn::Linear fc142;
    operator7 fc143;
    operator7 fc144;
    torch::nn::Linear fc145;
    operator7 fc146;
    operator7 fc147;
    operator7 fc148;
    torch::nn::Linear fc149;
    operator7 fc150;
    operator7 fc151;
    operator7 fc152;
    operator7 fc153;
    operator7 fc154;
    operator7 fc155;
    operator7 fc156;
    operator7 fc157;
    operator7 fc158;
    operator7 fc159;
    operator7 fc160;
    operator7 fc161;
    torch::nn::Linear fc162;
    operator7 fc163;
    torch::nn::Linear fc164;
    operator7 fc165;
    operator7 fc166;
    operator7 fc167;
    operator7 fc168;
    operator7 fc169;
    torch::nn::Linear fc170;
    operator7 fc171;
    operator7 fc172;
    torch::nn::Linear fc173;
    operator7 fc174;
    operator7 fc175;
    operator7 fc176;
    operator7 fc177;
    operator7 fc178;
    torch::nn::Linear fc179;
    operator7 fc180;
    operator7 fc181;
    operator7 fc182;
    operator7 fc183;
    operator7 fc184;
    operator7 fc185;
    operator7 fc186;
    operator7 fc187;
    operator7 fc188;
    operator7 fc189;
    operator7 fc190;
    operator7 fc191;
    operator7 fc192;
    operator7 fc193;
    operator7 fc194;
    operator7 fc195;
    operator7 fc196;
    operator7 fc197;
    operator7 fc198;
    operator7 fc199;
    operator7 fc200;
    operator7 fc201;
    operator7 fc202;
    torch::nn::Linear fc203;
    operator7 fc204;
    operator7 fc205;
    torch::nn::Linear fc206;
    operator7 fc207;
    operator7 fc208;
    torch::nn::Linear fc209;
    operator7 fc210;
    operator7 fc211;
    operator7 fc212;
    operator7 fc213;
    operator7 fc214;
    operator7 fc215;
    operator7 fc216;
    operator7 fc217;
    operator7 fc218;
    torch::nn::Linear fc219;
    operator7 fc220;
    operator7 fc221;
    operator7 fc222;
    operator7 fc223;
    operator7 fc224;
    operator7 fc225;
    torch::nn::Linear fc226;
    operator7 fc227;
    operator7 fc228;
    operator7 fc229;
    operator7 fc230;
    operator7 fc231;
    operator7 fc232;
    operator7 fc233;
    operator7 fc234;
    operator7 fc235;
    operator7 fc236;
    operator7 fc237;
    operator7 fc238;
    operator7 fc239;
    operator7 fc240;
    operator7 fc241;
    operator7 fc242;
    operator7 fc243;
    operator7 fc244;
    operator7 fc245;
    operator7 fc246;
    operator7 fc247;
    operator7 fc248;
    operator7 fc249;
    torch::nn::Linear fc250;
    operator7 fc251;
    operator7 fc252;
    operator7 fc253;
    operator7 fc254;
    operator7 fc255;
    operator7 fc256;
    operator7 fc257;
    operator7 fc258;
    torch::nn::Linear fc259;
    operator7 fc260;
    operator7 fc261;
    torch::nn::Linear fc262;
    operator7 fc263;
    operator7 fc264;
    operator7 fc265;
    torch::nn::Linear fc266;
    operator7 fc267;
    operator7 fc268;
    operator7 fc269;
    operator7 fc270;
    operator7 fc271;
    operator7 fc272;
    operator7 fc273;
    operator7 fc274;
    operator7 fc275;
    operator7 fc276;
    operator7 fc277;
    operator7 fc278;
    torch::nn::Linear fc279;
    operator7 fc280;
    operator7 fc281;
    operator7 fc282;
    operator7 fc283;
    operator7 fc284;
    operator7 fc285;
    torch::nn::Linear fc286;
    operator7 fc287;
    operator7 fc288;
    operator7 fc289;
    torch::nn::Linear fc290;
    operator7 fc291;
    operator7 fc292;
    operator7 fc293;
    operator7 fc294;
    operator7 fc295;
    operator7 fc296;
    operator7 fc297;
    operator7 fc298;
    operator7 fc299;
    operator7 fc300;
    operator7 fc301;
    operator7 fc302;
    torch::nn::Linear fc303;
    operator7 fc304;
    operator7 fc305;
    operator7 fc306;
    operator7 fc307;
    operator7 fc308;
    torch::nn::Linear fc309;
    operator7 fc310;
    torch::nn::Linear fc311;

    trans_warmup():
        fc0(d_model, d_model),
        fc1(d_model, d_model),
        fc2(d_model, d_model),
        fc3(d_model, d_ff),
        fc4(d_ff, d_model),
        fc5(d_model, d_model),
        fc6(d_model, d_model),
        fc7(d_model, d_model),
        fc8(d_model, d_ff),
        fc9(d_ff, d_model),
        fc10(d_model, d_model),
        fc11(d_model, d_model),
        fc12(d_model, d_model),
        fc13(d_model, d_ff),
        fc14(d_ff, d_model),
        fc15(d_model, d_model),
        fc16(d_model, d_model),
        fc17(d_model, d_model),
        fc18(d_model, d_ff),
        fc19(d_ff, d_model),
        fc20(d_model, d_model),
        fc21(d_model, d_model),
        fc22(d_model, d_model),
        fc23(d_model, d_ff),
        fc24(d_ff, d_model),
        fc25(d_model, d_model),
        fc26(d_model, d_model),
        fc27(d_model, d_model),
        fc28(d_model, d_ff),
        fc29(d_ff, d_model),
        fc30(d_model, d_model),
        fc31(d_model, d_model),
        fc32(d_model, d_model),
        fc33(d_model, d_ff),
        fc34(d_ff, d_model),
        fc35(d_model, d_model),
        fc36(d_model, d_model),
        fc37(d_model, d_model),
        fc38(d_model, d_ff),
        fc39(d_ff, d_model),
        fc40(d_model, d_model),
        fc41(d_model, d_model),
        fc42(d_model, d_model),
        fc43(d_model, d_ff),
        fc44(d_ff, d_model),
        fc45(d_model, d_model),
        fc46(d_model, d_model),
        fc47(d_model, d_model),
        fc48(d_model, d_ff),
        fc49(d_ff, d_model),
        fc50(d_model, d_model),
        fc51(d_model, d_model),
        fc52(d_model, d_model),
        fc53(d_model, d_ff),
        fc54(d_ff, d_model),
        fc55(d_model, d_model),
        fc56(d_model, d_model),
        fc57(d_model, d_model),
        fc58(d_model, d_ff),
        fc59(d_ff, d_model),
        fc60(d_model, d_model),
        fc61(d_model, d_model),
        fc62(d_model, d_model),
        fc63(d_model, d_ff),
        fc64(d_ff, d_model),
        fc65(d_model, d_model),
        fc66(d_model, d_model),
        fc67(d_model, d_model),
        fc68(d_model, d_ff),
        fc69(d_ff, d_model),
        fc70(d_model, d_model),
        fc71(d_model, d_model),
        fc72(d_model, d_model),
        fc73(d_model, d_ff),
        fc74(d_ff, d_model),
        fc75(d_model, d_model),
        fc76(d_model, d_model),
        fc77(d_model, d_model),
        fc78(d_model, d_ff),
        fc79(d_ff, d_model),
        fc80(d_model, d_model),
        fc81(d_model, d_model),
        fc82(d_model, d_model),
        fc83(d_model, d_ff),
        fc84(d_ff, d_model),
        fc85(d_model, d_model),
        fc86(d_model, d_model),
        fc87(d_model, d_model),
        fc88(d_model, d_ff),
        fc89(d_ff, d_model),
        fc90(d_model, d_model),
        fc91(d_model, d_model),
        fc92(d_model, d_model),
        fc93(d_model, d_ff),
        fc94(d_ff, d_model),
        fc95(d_model, d_model),
        fc96(d_model, d_model),
        fc97(d_model, d_model),
        fc98(d_model, d_ff),
        fc99(d_ff, d_model),
        fc100(d_model, d_model),
        fc101(d_model, d_model),
        fc102(d_model, d_model),
        fc103(d_model, d_ff),
        fc104(d_ff, d_model),
        fc105(d_model, d_model),
        fc106(d_model, d_model),
        fc107(d_model, d_model),
        fc108(d_model, d_ff),
        fc109(d_ff, d_model),
        fc110(d_model, d_model),
        fc111(d_model, d_model),
        fc112(d_model, d_model),
        fc113(d_model, d_ff),
        fc114(d_ff, d_model),
        fc115(d_model, d_model),
        fc116(d_model, d_model),
        fc117(d_model, d_model),
        fc118(d_model, d_ff),
        fc119(d_ff, d_model),
        fc120(d_model, d_model),
        fc121(d_model, d_model),
        fc122(d_model, d_model),
        fc123(d_model, d_model),
        fc124(d_model, d_model),
        fc125(d_model, d_model),
        fc126(d_model, d_ff),
        fc127(d_ff, d_model),
        fc128(d_model, d_model),
        fc129(d_model, d_model),
        fc130(d_model, d_model),
        fc131(d_model, d_model),
        fc132(d_model, d_model),
        fc133(d_model, d_model),
        fc134(d_model, d_ff),
        fc135(d_ff, d_model),
        fc136(d_model, d_model),
        fc137(d_model, d_model),
        fc138(d_model, d_model),
        fc139(d_model, d_model),
        fc140(d_model, d_model),
        fc141(d_model, d_model),
        fc142(d_model, d_ff),
        fc143(d_ff, d_model),
        fc144(d_model, d_model),
        fc145(d_model, d_model),
        fc146(d_model, d_model),
        fc147(d_model, d_model),
        fc148(d_model, d_model),
        fc149(d_model, d_model),
        fc150(d_model, d_ff),
        fc151(d_ff, d_model),
        fc152(d_model, d_model),
        fc153(d_model, d_model),
        fc154(d_model, d_model),
        fc155(d_model, d_model),
        fc156(d_model, d_model),
        fc157(d_model, d_model),
        fc158(d_model, d_ff),
        fc159(d_ff, d_model),
        fc160(d_model, d_model),
        fc161(d_model, d_model),
        fc162(d_model, d_model),
        fc163(d_model, d_model),
        fc164(d_model, d_model),
        fc165(d_model, d_model),
        fc166(d_model, d_ff),
        fc167(d_ff, d_model),
        fc168(d_model, d_model),
        fc169(d_model, d_model),
        fc170(d_model, d_model),
        fc171(d_model, d_model),
        fc172(d_model, d_model),
        fc173(d_model, d_model),
        fc174(d_model, d_ff),
        fc175(d_ff, d_model),
        fc176(d_model, d_model),
        fc177(d_model, d_model),
        fc178(d_model, d_model),
        fc179(d_model, d_model),
        fc180(d_model, d_model),
        fc181(d_model, d_model),
        fc182(d_model, d_ff),
        fc183(d_ff, d_model),
        fc184(d_model, d_model),
        fc185(d_model, d_model),
        fc186(d_model, d_model),
        fc187(d_model, d_model),
        fc188(d_model, d_model),
        fc189(d_model, d_model),
        fc190(d_model, d_ff),
        fc191(d_ff, d_model),
        fc192(d_model, d_model),
        fc193(d_model, d_model),
        fc194(d_model, d_model),
        fc195(d_model, d_model),
        fc196(d_model, d_model),
        fc197(d_model, d_model),
        fc198(d_model, d_ff),
        fc199(d_ff, d_model),
        fc200(d_model, d_model),
        fc201(d_model, d_model),
        fc202(d_model, d_model),
        fc203(d_model, d_model),
        fc204(d_model, d_model),
        fc205(d_model, d_model),
        fc206(d_model, d_ff),
        fc207(d_ff, d_model),
        fc208(d_model, d_model),
        fc209(d_model, d_model),
        fc210(d_model, d_model),
        fc211(d_model, d_model),
        fc212(d_model, d_model),
        fc213(d_model, d_model),
        fc214(d_model, d_ff),
        fc215(d_ff, d_model),
        fc216(d_model, d_model),
        fc217(d_model, d_model),
        fc218(d_model, d_model),
        fc219(d_model, d_model),
        fc220(d_model, d_model),
        fc221(d_model, d_model),
        fc222(d_model, d_ff),
        fc223(d_ff, d_model),
        fc224(d_model, d_model),
        fc225(d_model, d_model),
        fc226(d_model, d_model),
        fc227(d_model, d_model),
        fc228(d_model, d_model),
        fc229(d_model, d_model),
        fc230(d_model, d_ff),
        fc231(d_ff, d_model),
        fc232(d_model, d_model),
        fc233(d_model, d_model),
        fc234(d_model, d_model),
        fc235(d_model, d_model),
        fc236(d_model, d_model),
        fc237(d_model, d_model),
        fc238(d_model, d_ff),
        fc239(d_ff, d_model),
        fc240(d_model, d_model),
        fc241(d_model, d_model),
        fc242(d_model, d_model),
        fc243(d_model, d_model),
        fc244(d_model, d_model),
        fc245(d_model, d_model),
        fc246(d_model, d_ff),
        fc247(d_ff, d_model),
        fc248(d_model, d_model),
        fc249(d_model, d_model),
        fc250(d_model, d_model),
        fc251(d_model, d_model),
        fc252(d_model, d_model),
        fc253(d_model, d_model),
        fc254(d_model, d_ff),
        fc255(d_ff, d_model),
        fc256(d_model, d_model),
        fc257(d_model, d_model),
        fc258(d_model, d_model),
        fc259(d_model, d_model),
        fc260(d_model, d_model),
        fc261(d_model, d_model),
        fc262(d_model, d_ff),
        fc263(d_ff, d_model),
        fc264(d_model, d_model),
        fc265(d_model, d_model),
        fc266(d_model, d_model),
        fc267(d_model, d_model),
        fc268(d_model, d_model),
        fc269(d_model, d_model),
        fc270(d_model, d_ff),
        fc271(d_ff, d_model),
        fc272(d_model, d_model),
        fc273(d_model, d_model),
        fc274(d_model, d_model),
        fc275(d_model, d_model),
        fc276(d_model, d_model),
        fc277(d_model, d_model),
        fc278(d_model, d_ff),
        fc279(d_ff, d_model),
        fc280(d_model, d_model),
        fc281(d_model, d_model),
        fc282(d_model, d_model),
        fc283(d_model, d_model),
        fc284(d_model, d_model),
        fc285(d_model, d_model),
        fc286(d_model, d_ff),
        fc287(d_ff, d_model),
        fc288(d_model, d_model),
        fc289(d_model, d_model),
        fc290(d_model, d_model),
        fc291(d_model, d_model),
        fc292(d_model, d_model),
        fc293(d_model, d_model),
        fc294(d_model, d_ff),
        fc295(d_ff, d_model),
        fc296(d_model, d_model),
        fc297(d_model, d_model),
        fc298(d_model, d_model),
        fc299(d_model, d_model),
        fc300(d_model, d_model),
        fc301(d_model, d_model),
        fc302(d_model, d_ff),
        fc303(d_ff, d_model),
        fc304(d_model, d_model),
        fc305(d_model, d_model),
        fc306(d_model, d_model),
        fc307(d_model, d_model),
        fc308(d_model, d_model),
        fc309(d_model, d_model),
        fc310(d_model, d_ff),
        fc311(d_ff, d_model)
        {
            relu.to(at::kCUDA);
            fc0.to(at::kCUDA);
            fc1.to(at::kCUDA);
            fc2.to(at::kCUDA);
            fc3.to(at::kCUDA);
            fc4->to(at::kCUDA);
            fc5.to(at::kCUDA);
            fc6.to(at::kCUDA);
            fc7.to(at::kCUDA);
            fc8.to(at::kCUDA);
            fc9->to(at::kCUDA);
            fc10.to(at::kCUDA);
            fc11.to(at::kCUDA);
            fc12.to(at::kCUDA);
            fc13.to(at::kCUDA);
            fc14->to(at::kCUDA);
            fc15.to(at::kCUDA);
            fc16.to(at::kCUDA);
            fc17.to(at::kCUDA);
            fc18->to(at::kCUDA);
            fc19.to(at::kCUDA);
            fc20.to(at::kCUDA);
            fc21.to(at::kCUDA);
            fc22.to(at::kCUDA);
            fc23.to(at::kCUDA);
            fc24.to(at::kCUDA);
            fc25->to(at::kCUDA);
            fc26.to(at::kCUDA);
            fc27.to(at::kCUDA);
            fc28.to(at::kCUDA);
            fc29.to(at::kCUDA);
            fc30->to(at::kCUDA);
            fc31->to(at::kCUDA);
            fc32->to(at::kCUDA);
            fc33->to(at::kCUDA);
            fc34.to(at::kCUDA);
            fc35.to(at::kCUDA);
            fc36.to(at::kCUDA);
            fc37->to(at::kCUDA);
            fc38.to(at::kCUDA);
            fc39.to(at::kCUDA);
            fc40.to(at::kCUDA);
            fc41->to(at::kCUDA);
            fc42.to(at::kCUDA);
            fc43->to(at::kCUDA);
            fc44.to(at::kCUDA);
            fc45->to(at::kCUDA);
            fc46.to(at::kCUDA);
            fc47.to(at::kCUDA);
            fc48.to(at::kCUDA);
            fc49->to(at::kCUDA);
            fc50.to(at::kCUDA);
            fc51.to(at::kCUDA);
            fc52.to(at::kCUDA);
            fc53.to(at::kCUDA);
            fc54->to(at::kCUDA);
            fc55.to(at::kCUDA);
            fc56->to(at::kCUDA);
            fc57.to(at::kCUDA);
            fc58.to(at::kCUDA);
            fc59->to(at::kCUDA);
            fc60.to(at::kCUDA);
            fc61.to(at::kCUDA);
            fc62.to(at::kCUDA);
            fc63.to(at::kCUDA);
            fc64.to(at::kCUDA);
            fc65.to(at::kCUDA);
            fc66->to(at::kCUDA);
            fc67.to(at::kCUDA);
            fc68.to(at::kCUDA);
            fc69.to(at::kCUDA);
            fc70->to(at::kCUDA);
            fc71.to(at::kCUDA);
            fc72.to(at::kCUDA);
            fc73.to(at::kCUDA);
            fc74.to(at::kCUDA);
            fc75.to(at::kCUDA);
            fc76.to(at::kCUDA);
            fc77.to(at::kCUDA);
            fc78.to(at::kCUDA);
            fc79.to(at::kCUDA);
            fc80->to(at::kCUDA);
            fc81.to(at::kCUDA);
            fc82.to(at::kCUDA);
            fc83.to(at::kCUDA);
            fc84.to(at::kCUDA);
            fc85.to(at::kCUDA);
            fc86.to(at::kCUDA);
            fc87->to(at::kCUDA);
            fc88.to(at::kCUDA);
            fc89.to(at::kCUDA);
            fc90->to(at::kCUDA);
            fc91.to(at::kCUDA);
            fc92.to(at::kCUDA);
            fc93.to(at::kCUDA);
            fc94->to(at::kCUDA);
            fc95->to(at::kCUDA);
            fc96.to(at::kCUDA);
            fc97.to(at::kCUDA);
            fc98.to(at::kCUDA);
            fc99.to(at::kCUDA);
            fc100.to(at::kCUDA);
            fc101->to(at::kCUDA);
            fc102->to(at::kCUDA);
            fc103.to(at::kCUDA);
            fc104.to(at::kCUDA);
            fc105.to(at::kCUDA);
            fc106.to(at::kCUDA);
            fc107.to(at::kCUDA);
            fc108.to(at::kCUDA);
            fc109.to(at::kCUDA);
            fc110.to(at::kCUDA);
            fc111.to(at::kCUDA);
            fc112.to(at::kCUDA);
            fc113.to(at::kCUDA);
            fc114.to(at::kCUDA);
            fc115.to(at::kCUDA);
            fc116.to(at::kCUDA);
            fc117.to(at::kCUDA);
            fc118.to(at::kCUDA);
            fc119.to(at::kCUDA);
            fc120.to(at::kCUDA);
            fc121.to(at::kCUDA);
            fc122.to(at::kCUDA);
            fc123.to(at::kCUDA);
            fc124.to(at::kCUDA);
            fc125.to(at::kCUDA);
            fc126.to(at::kCUDA);
            fc127->to(at::kCUDA);
            fc128->to(at::kCUDA);
            fc129->to(at::kCUDA);
            fc130.to(at::kCUDA);
            fc131.to(at::kCUDA);
            fc132.to(at::kCUDA);
            fc133.to(at::kCUDA);
            fc134.to(at::kCUDA);
            fc135.to(at::kCUDA);
            fc136.to(at::kCUDA);
            fc137.to(at::kCUDA);
            fc138.to(at::kCUDA);
            fc139.to(at::kCUDA);
            fc140.to(at::kCUDA);
            fc141->to(at::kCUDA);
            fc142->to(at::kCUDA);
            fc143.to(at::kCUDA);
            fc144.to(at::kCUDA);
            fc145->to(at::kCUDA);
            fc146.to(at::kCUDA);
            fc147.to(at::kCUDA);
            fc148.to(at::kCUDA);
            fc149->to(at::kCUDA);
            fc150.to(at::kCUDA);
            fc151.to(at::kCUDA);
            fc152.to(at::kCUDA);
            fc153.to(at::kCUDA);
            fc154.to(at::kCUDA);
            fc155.to(at::kCUDA);
            fc156.to(at::kCUDA);
            fc157.to(at::kCUDA);
            fc158.to(at::kCUDA);
            fc159.to(at::kCUDA);
            fc160.to(at::kCUDA);
            fc161.to(at::kCUDA);
            fc162->to(at::kCUDA);
            fc163.to(at::kCUDA);
            fc164->to(at::kCUDA);
            fc165.to(at::kCUDA);
            fc166.to(at::kCUDA);
            fc167.to(at::kCUDA);
            fc168.to(at::kCUDA);
            fc169.to(at::kCUDA);
            fc170->to(at::kCUDA);
            fc171.to(at::kCUDA);
            fc172.to(at::kCUDA);
            fc173->to(at::kCUDA);
            fc174.to(at::kCUDA);
            fc175.to(at::kCUDA);
            fc176.to(at::kCUDA);
            fc177.to(at::kCUDA);
            fc178.to(at::kCUDA);
            fc179->to(at::kCUDA);
            fc180.to(at::kCUDA);
            fc181.to(at::kCUDA);
            fc182.to(at::kCUDA);
            fc183.to(at::kCUDA);
            fc184.to(at::kCUDA);
            fc185.to(at::kCUDA);
            fc186.to(at::kCUDA);
            fc187.to(at::kCUDA);
            fc188.to(at::kCUDA);
            fc189.to(at::kCUDA);
            fc190.to(at::kCUDA);
            fc191.to(at::kCUDA);
            fc192.to(at::kCUDA);
            fc193.to(at::kCUDA);
            fc194.to(at::kCUDA);
            fc195.to(at::kCUDA);
            fc196.to(at::kCUDA);
            fc197.to(at::kCUDA);
            fc198.to(at::kCUDA);
            fc199.to(at::kCUDA);
            fc200.to(at::kCUDA);
            fc201.to(at::kCUDA);
            fc202.to(at::kCUDA);
            fc203->to(at::kCUDA);
            fc204.to(at::kCUDA);
            fc205.to(at::kCUDA);
            fc206->to(at::kCUDA);
            fc207.to(at::kCUDA);
            fc208.to(at::kCUDA);
            fc209->to(at::kCUDA);
            fc210.to(at::kCUDA);
            fc211.to(at::kCUDA);
            fc212.to(at::kCUDA);
            fc213.to(at::kCUDA);
            fc214.to(at::kCUDA);
            fc215.to(at::kCUDA);
            fc216.to(at::kCUDA);
            fc217.to(at::kCUDA);
            fc218.to(at::kCUDA);
            fc219->to(at::kCUDA);
            fc220.to(at::kCUDA);
            fc221.to(at::kCUDA);
            fc222.to(at::kCUDA);
            fc223.to(at::kCUDA);
            fc224.to(at::kCUDA);
            fc225.to(at::kCUDA);
            fc226->to(at::kCUDA);
            fc227.to(at::kCUDA);
            fc228.to(at::kCUDA);
            fc229.to(at::kCUDA);
            fc230.to(at::kCUDA);
            fc231.to(at::kCUDA);
            fc232.to(at::kCUDA);
            fc233.to(at::kCUDA);
            fc234.to(at::kCUDA);
            fc235.to(at::kCUDA);
            fc236.to(at::kCUDA);
            fc237.to(at::kCUDA);
            fc238.to(at::kCUDA);
            fc239.to(at::kCUDA);
            fc240.to(at::kCUDA);
            fc241.to(at::kCUDA);
            fc242.to(at::kCUDA);
            fc243.to(at::kCUDA);
            fc244.to(at::kCUDA);
            fc245.to(at::kCUDA);
            fc246.to(at::kCUDA);
            fc247.to(at::kCUDA);
            fc248.to(at::kCUDA);
            fc249.to(at::kCUDA);
            fc250->to(at::kCUDA);
            fc251.to(at::kCUDA);
            fc252.to(at::kCUDA);
            fc253.to(at::kCUDA);
            fc254.to(at::kCUDA);
            fc255.to(at::kCUDA);
            fc256.to(at::kCUDA);
            fc257.to(at::kCUDA);
            fc258.to(at::kCUDA);
            fc259->to(at::kCUDA);
            fc260.to(at::kCUDA);
            fc261.to(at::kCUDA);
            fc262->to(at::kCUDA);
            fc263.to(at::kCUDA);
            fc264.to(at::kCUDA);
            fc265.to(at::kCUDA);
            fc266->to(at::kCUDA);
            fc267.to(at::kCUDA);
            fc268.to(at::kCUDA);
            fc269.to(at::kCUDA);
            fc270.to(at::kCUDA);
            fc271.to(at::kCUDA);
            fc272.to(at::kCUDA);
            fc273.to(at::kCUDA);
            fc274.to(at::kCUDA);
            fc275.to(at::kCUDA);
            fc276.to(at::kCUDA);
            fc277.to(at::kCUDA);
            fc278.to(at::kCUDA);
            fc279->to(at::kCUDA);
            fc280.to(at::kCUDA);
            fc281.to(at::kCUDA);
            fc282.to(at::kCUDA);
            fc283.to(at::kCUDA);
            fc284.to(at::kCUDA);
            fc285.to(at::kCUDA);
            fc286->to(at::kCUDA);
            fc287.to(at::kCUDA);
            fc288.to(at::kCUDA);
            fc289.to(at::kCUDA);
            fc290->to(at::kCUDA);
            fc291.to(at::kCUDA);
            fc292.to(at::kCUDA);
            fc293.to(at::kCUDA);
            fc294.to(at::kCUDA);
            fc295.to(at::kCUDA);
            fc296.to(at::kCUDA);
            fc297.to(at::kCUDA);
            fc298.to(at::kCUDA);
            fc299.to(at::kCUDA);
            fc300.to(at::kCUDA);
            fc301.to(at::kCUDA);
            fc302.to(at::kCUDA);
            fc303->to(at::kCUDA);
            fc304.to(at::kCUDA);
            fc305.to(at::kCUDA);
            fc306.to(at::kCUDA);
            fc307.to(at::kCUDA);
            fc308.to(at::kCUDA);
            fc309->to(at::kCUDA);
            fc310.to(at::kCUDA);
            fc311->to(at::kCUDA);
        }
    
    torch::Tensor forward(torch::Tensor src, torch::Tensor tgt)
    {
       
        int nbatches;
        torch::Tensor temp;
        torch::Tensor temp0;
        torch::Tensor temp1;
        torch::Tensor temp2;
        torch::Tensor temp3;
        torch::Tensor src0;
        torch::Tensor tgt0;

        //encoder-layer-0
        src0 = src;
        nbatches = src0.size(0);
        temp = fc0.forward(src0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = fc1.forward(src0);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = fc2.forward(src0);
        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        src0 = temp.div(sqrt_dk);
        src0 = src0.softmax(3);
        src0 = src0.matmul(temp3);
        src0 = src0.transpose(1, 2);
        src0 = src0.contiguous();
        src0 = src0.view( {nbatches, -1, nheads * d_k} )[0];

        src = src + src0;
        temp = fc3.forward(src);
        temp = relu.forward(temp);
        temp = fc4(temp);
        src = src + temp;

        //encoder-layer-1
        src0 = src;
        nbatches = src0.size(0);
        temp = fc5.forward(src0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = fc6.forward(src0);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = fc7.forward(src0);
        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        src0 = temp.div(sqrt_dk);
        src0 = src0.softmax(3);
        src0 = src0.matmul(temp3);
        src0 = src0.transpose(1, 2);
        src0 = src0.contiguous();
        src0 = src0.view( {nbatches, -1, nheads * d_k} )[0];
        
        src = src + src0;
        temp = fc8.forward(src);
        temp = relu.forward(temp);
        temp = fc9(temp);
        src = src + temp;
       

        //encoder-layer-2
        src0 = src;
        nbatches = src0.size(0);
        temp = fc10.forward(src0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
       
        temp = fc11.forward(src0);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc12.forward(src0);
        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        src0 = temp.div(sqrt_dk);
        src0 = src0.softmax(3);
        src0 = src0.matmul(temp3);
        src0 = src0.transpose(1, 2);
        src0 = src0.contiguous();
        src0 = src0.view( {nbatches, -1, nheads * d_k} )[0];
        
        src = src + src0;
        temp = fc13.forward(src);
        temp = relu.forward(temp);
        temp = fc14(temp);
        src = src + temp;

        //encoder-layer-3


        src0 = src;
        nbatches = src0.size(0);
        temp = fc15.forward(src0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
       
        temp = fc16.forward(src0);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
   
        temp = fc17.forward(src0);
        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        src0 = temp.div(sqrt_dk);
        src0 = src0.softmax(3);
        src0 = src0.matmul(temp3);
        src0 = src0.transpose(1, 2);
        src0 = src0.contiguous();
        src0 = src0.view( {nbatches, -1, nheads * d_k} )[0];
        
        src = src + src0;
        temp = fc18(src);
        temp = relu.forward(temp);
        temp = fc19.forward(temp);
        src = src + temp;
 


        //encoder-layer-4


        src0 = src;
        nbatches = src0.size(0);
        temp = fc20.forward(src0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
    
        temp = fc21.forward(src0);
   
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc22.forward(src0);

        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        src0 = temp.div(sqrt_dk);
        src0 = src0.softmax(3);
        src0 = src0.matmul(temp3);
        src0 = src0.transpose(1, 2);
        src0 = src0.contiguous();
        src0 = src0.view( {nbatches, -1, nheads * d_k} )[0];
        
        src = src + src0;
  
        temp = fc23.forward(src);
        temp = relu.forward(temp);
        temp = fc24.forward(temp);
        src = src + temp;

        //encoder-layer-5


        src0 = src;
        nbatches = src0.size(0);
        temp1 = fc25(src0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
   
        temp = fc26.forward(src0);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
     
        temp = fc27.forward(src0);

        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        src0 = temp.div(sqrt_dk);
        src0 = src0.softmax(3);
        src0 = src0.matmul(temp3);
        src0 = src0.transpose(1, 2);
        src0 = src0.contiguous();
        src0 = src0.view( {nbatches, -1, nheads * d_k} )[0];
        
        src = src + src0;
        temp = fc28.forward(src);
        temp = relu.forward(temp);
        temp = fc29.forward(temp);
        src = src + temp;


        //encoder-layer-6


        src0 = src;
        nbatches = src0.size(0);
        temp1 = fc30(src0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc31(src0);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc32(src0);
        temp3 = temp3.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        src0 = temp.div(sqrt_dk);
        src0 = src0.softmax(3);
        src0 = src0.matmul(temp3);
        src0 = src0.transpose(1, 2);
        src0 = src0.contiguous();
        src0 = src0.view( {nbatches, -1, nheads * d_k} )[0];
        
        src = src + src0;
        temp = fc33(src);
        temp = F::relu(temp);
        temp = fc34.forward(temp);
        src = src + temp;


        //encoder-layer-7


        src0 = src;
        nbatches = src0.size(0);

        temp = fc35.forward(src0);
  
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc36.forward(src0);

        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc37(src0);
        temp3 = temp3.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        src0 = temp.div(sqrt_dk);
        src0 = src0.softmax(3);
        src0 = src0.matmul(temp3);
        src0 = src0.transpose(1, 2);
        src0 = src0.contiguous();
        src0 = src0.view( {nbatches, -1, nheads * d_k} )[0];
        
        src = src + src0;
    
        temp = fc38.forward(src);
        temp = relu.forward(temp);
        temp = fc39.forward(temp);
        src = src + temp;


        //encoder-layer-8


        src0 = src;
        nbatches = src0.size(0);
        temp = fc40.forward(src0);
   
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc41(src0);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc42.forward(src0);

        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        src0 = temp.div(sqrt_dk);
        src0 = src0.softmax(3);
        src0 = src0.matmul(temp3);
        src0 = src0.transpose(1, 2);
        src0 = src0.contiguous();
        src0 = src0.view( {nbatches, -1, nheads * d_k} )[0];
        
        src = src + src0;
        temp = fc43(src);
        temp = relu.forward(temp);
        temp = fc44.forward(temp);
        src = src + temp;


        //encoder-layer-9


        src0 = src;
        nbatches = src0.size(0);
        temp1 = fc45(src0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
   
        temp = fc46.forward(src0);

        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc47.forward(src0);

        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        src0 = temp.div(sqrt_dk);
        src0 = src0.softmax(3);
        src0 = src0.matmul(temp3);
        src0 = src0.transpose(1, 2);
        src0 = src0.contiguous();
        src0 = src0.view( {nbatches, -1, nheads * d_k} )[0];
        
        src = src + src0;
   
        temp = fc48.forward(src);
        temp = relu.forward(temp);
        temp = fc49(temp);
        src = src + temp;



        //encoder-layer-10


        src0 = src;
        nbatches = src0.size(0);
        temp = fc50.forward(src0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
  
        temp = fc51.forward(src0);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
      
        temp = fc52.forward(src0);

        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        src0 = temp.div(sqrt_dk);
        src0 = src0.softmax(3);
        src0 = src0.matmul(temp3);
        src0 = src0.transpose(1, 2);
        src0 = src0.contiguous();
        src0 = src0.view( {nbatches, -1, nheads * d_k} )[0];
        
        src = src + src0;

        temp = fc53.forward(src);
        temp = relu.forward(temp);
        temp = fc54(temp);
        src = src + temp;


        //encoder-layer-11


        src0 = src;
        nbatches = src0.size(0);
        temp = fc55.forward(src0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc56(src0);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
     
        temp = fc57.forward(src0);
    
        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        src0 = temp.div(sqrt_dk);
        src0 = src0.softmax(3);
        src0 = src0.matmul(temp3);
        src0 = src0.transpose(1, 2);
        src0 = src0.contiguous();
        src0 = src0.view( {nbatches, -1, nheads * d_k} )[0];
        
        src = src + src0;
        temp = fc58.forward(src);
        temp = F::relu(temp);
        temp = fc59(temp);
        src = src + temp;


        //encoder-layer-12


        src0 = src;
        nbatches = src0.size(0);
        temp = fc60.forward(src0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
    
        temp = fc61.forward(src0);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
    
        temp = fc62.forward(src0);

        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        src0 = temp.div(sqrt_dk);
        src0 = src0.softmax(3);
        src0 = src0.matmul(temp3);
        src0 = src0.transpose(1, 2);
        src0 = src0.contiguous();
        src0 = src0.view( {nbatches, -1, nheads * d_k} )[0];
        
        src = src + src0;
        temp = fc63.forward(src);
        temp = relu.forward(temp);
        temp = fc64.forward(temp);
        src = src + temp;


        //encoder-layer-13


        src0 = src;
        nbatches = src0.size(0);
        temp = fc65.forward(src0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc66(src0);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
     
        temp = fc67.forward(src0);
        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        src0 = temp.div(sqrt_dk);
        src0 = src0.softmax(3);
        src0 = src0.matmul(temp3);
        src0 = src0.transpose(1, 2);
        src0 = src0.contiguous();
        src0 = src0.view( {nbatches, -1, nheads * d_k} )[0];
        
        src = src + src0;
        temp = fc68.forward(src);
        temp = relu.forward(temp);
        temp = fc69.forward(temp);
        src = src + temp;


        //encoder-layer-14


        src0 = src;
        nbatches = src0.size(0);
        temp1 = fc70(src0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
     
        temp = fc71.forward(src0);
   
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
 
        temp = fc72.forward(src0);

        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        src0 = temp.div(sqrt_dk);
        src0 = src0.softmax(3);
        src0 = src0.matmul(temp3);
        src0 = src0.transpose(1, 2);
        src0 = src0.contiguous();
        src0 = src0.view( {nbatches, -1, nheads * d_k} )[0];
        
        src = src + src0;

        temp = fc73.forward(src);
        temp = relu.forward(temp);
        temp = fc74.forward(temp);
        src = src + temp;

        //encoder-layer-15


        src0 = src;
        nbatches = src0.size(0);
        temp = fc75.forward(src0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc76.forward(src0);

        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc77.forward(src0);

        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        src0 = temp.div(sqrt_dk);
        src0 = src0.softmax(3);
        src0 = src0.matmul(temp3);
        src0 = src0.transpose(1, 2);
        src0 = src0.contiguous();
        src0 = src0.view( {nbatches, -1, nheads * d_k} )[0];
        
        src = src + src0;

        temp = fc78.forward(src);
        temp = relu.forward(temp);
        temp = fc79.forward(temp);
        src = src + temp;


        //encoder-layer-16


        src0 = src;
        nbatches = src0.size(0);
        temp1 = fc80(src0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc81.forward(src0);

        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
  
        temp = fc82.forward(src0);
  
        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        src0 = temp.div(sqrt_dk);
        src0 = src0.softmax(3);
        src0 = src0.matmul(temp3);
        src0 = src0.transpose(1, 2);
        src0 = src0.contiguous();
        src0 = src0.view( {nbatches, -1, nheads * d_k} )[0];
        
        src = src + src0;
        temp = fc83.forward(src);
        temp = relu.forward(temp);
        temp = fc84.forward(temp);
        src = src + temp;


        //encoder-layer-17


        src0 = src;
        nbatches = src0.size(0);
        temp = fc85.forward(src0);

        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
 
        temp = fc86.forward(src0);
  
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc87(src0);
        temp3 = temp3.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        src0 = temp.div(sqrt_dk);
        src0 = src0.softmax(3);
        src0 = src0.matmul(temp3);
        src0 = src0.transpose(1, 2);
        src0 = src0.contiguous();
        src0 = src0.view( {nbatches, -1, nheads * d_k} )[0];
        
        src = src + src0;
        temp = fc88.forward(src);
        temp = F::relu(temp);
        temp = fc89.forward(temp);
        src = src + temp;


        //encoder-layer-18


        src0 = src;
        nbatches = src0.size(0);
        temp1 = fc90(src0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
   
        temp = fc91.forward(src0);
  
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc92.forward(src0);

        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        src0 = temp.div(sqrt_dk);
        src0 = src0.softmax(3);
        src0 = src0.matmul(temp3);
        src0 = src0.transpose(1, 2);
        src0 = src0.contiguous();
        src0 = src0.view( {nbatches, -1, nheads * d_k} )[0];
        
        src = src + src0;

        temp = fc93.forward(src);
        temp = F::relu(temp);
        temp = fc94(temp);
        src = src + temp;


        //encoder-layer-19


        src0 = src;
        nbatches = src0.size(0);
        temp1 = fc95(src0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
   
        temp = fc96.forward(src0);

        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc97.forward(src0);

        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        src0 = temp.div(sqrt_dk);
        src0 = src0.softmax(3);
        src0 = src0.matmul(temp3);
        src0 = src0.transpose(1, 2);
        src0 = src0.contiguous();
        src0 = src0.view( {nbatches, -1, nheads * d_k} )[0];
        
        src = src + src0;

        temp = fc98.forward(src);
        temp = relu.forward(temp);
        temp = fc99.forward(temp);
        src = src + temp;


        //encoder-layer-20


        src0 = src;
        nbatches = src0.size(0);
        temp = fc100.forward(src0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc101(src0);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc102(src0);
        temp3 = temp3.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        src0 = temp.div(sqrt_dk);
        src0 = src0.softmax(3);
        src0 = src0.matmul(temp3);
        src0 = src0.transpose(1, 2);
        src0 = src0.contiguous();
        src0 = src0.view( {nbatches, -1, nheads * d_k} )[0];
        
        src = src + src0;
        temp = fc103.forward(src);
        temp = relu.forward(temp);
        temp = fc104.forward(temp);
        src = src + temp;


        //encoder-layer-21


        src0 = src;
        nbatches = src0.size(0);
        temp = fc105.forward(src0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc106.forward(src0);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
    
        temp = fc107.forward(src0);
        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        src0 = temp.div(sqrt_dk);
        src0 = src0.softmax(3);
        src0 = src0.matmul(temp3);
        src0 = src0.transpose(1, 2);
        src0 = src0.contiguous();
        src0 = src0.view( {nbatches, -1, nheads * d_k} )[0];
        
        src = src + src0;
        temp = fc108.forward(src);
        temp = relu.forward(temp);
        temp = fc109.forward(temp);
        src = src + temp;


        //encoder-layer-22


        src0 = src;
        nbatches = src0.size(0);
        temp = fc110.forward(src0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc111.forward(src0);

        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc112.forward(src0);

        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        src0 = temp.div(sqrt_dk);
        src0 = src0.softmax(3);
        src0 = src0.matmul(temp3);
        src0 = src0.transpose(1, 2);
        src0 = src0.contiguous();
        src0 = src0.view( {nbatches, -1, nheads * d_k} )[0];
        
        src = src + src0;

        temp = fc113.forward(src);
        temp = F::relu(temp);
        temp = fc114.forward(temp);
        src = src + temp;


        //encoder-layer-23


        src0 = src;
        nbatches = src0.size(0);

        temp = fc115.forward(src0);
  
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc116.forward(src0);

        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc117.forward(src0);

        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        src0 = temp.div(sqrt_dk);
        src0 = src0.softmax(3);
        src0 = src0.matmul(temp3);
        src0 = src0.transpose(1, 2);
        src0 = src0.contiguous();
        src0 = src0.view( {nbatches, -1, nheads * d_k} )[0];
        
        src = src + src0;
        temp = fc118.forward(src);
        temp = relu.forward(temp);
        temp = fc119.forward(temp);
        src = src + temp;



        //decoder-layer-0


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc120.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc121.forward(tgt0);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc122.forward(tgt0);
        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt = tgt + tgt0;
        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc123.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
  
        temp = fc124.forward(src);

        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
   
        temp = fc125.forward(src);

        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt0 = tgt + tgt0;
        
        temp = fc126.forward(tgt);
        temp = relu.forward(temp);
        temp = fc127(temp);
        tgt = tgt + temp;

        //decoder-layer-1


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp1 = fc128(tgt0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc129(tgt0);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc130.forward(tgt0);

        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt = tgt + tgt0;
        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc131.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
     
        temp = fc132.forward(src);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
   
        temp = fc133.forward(src);

        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt0 = tgt + tgt0;
  
        temp = fc134.forward(tgt);
        temp = F::relu(temp);
        temp = fc135.forward(temp);
        tgt = tgt + temp;


        //decoder-layer-2


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc136.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc137.forward(tgt0);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc138.forward(tgt0);
        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt = tgt + tgt0;
        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc139.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc140.forward(src);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc141(src);
        temp3 = temp3.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt0 = tgt + tgt0;
        
        temp = fc142(tgt);
        temp = relu.forward(temp);
        temp = fc143.forward(temp);
        tgt = tgt + temp;


        //decoder-layer-3


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc144.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc145(tgt0);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc146.forward(tgt0);

        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt = tgt + tgt0;
        tgt0 = tgt;
        nbatches = tgt0.size(0);

        temp = fc147.forward(tgt0);

        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc148.forward(src);
   
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc149(src);
        temp3 = temp3.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt0 = tgt + tgt0;
        
        temp = fc150.forward(tgt);
        temp = relu.forward(temp);
        temp = fc151.forward(temp);
        tgt = tgt + temp;


        //decoder-layer-4


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc152.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc153.forward(tgt0);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc154.forward(tgt0);

        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt = tgt + tgt0;
        tgt0 = tgt;
        nbatches = tgt0.size(0);

        temp = fc155.forward(tgt0);

        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
 
        temp = fc156.forward(src);

        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc157.forward(src);

        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt0 = tgt + tgt0;

        temp = fc158.forward(tgt);
        temp = relu.forward(temp);
        temp = fc159.forward(temp);
        tgt = tgt + temp;


        //decoder-layer-5


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc160.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc161.forward(tgt0);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc162(tgt0);
        temp3 = temp3.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt = tgt + tgt0;
        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc163.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc164(src);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc165.forward(src);
        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt0 = tgt + tgt0;
        
        temp = fc166.forward(tgt);
        temp = F::relu(temp);
        temp = fc167.forward(temp);
        tgt = tgt + temp;


        //decoder-layer-6


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc168.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
    
        temp = fc169.forward(tgt0);
    
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc170(tgt0);
        temp3 = temp3.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt = tgt + tgt0;
        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc171.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
      
        temp = fc172.forward(src);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc173(src);
        temp3 = temp3.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt0 = tgt + tgt0;
        
        temp = fc174.forward(tgt);
        temp = relu.forward(temp);
        temp = fc175.forward(temp);
        tgt = tgt + temp;


        //decoder-layer-7


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc176.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc177.forward(tgt0);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc178.forward(tgt0);
        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt = tgt + tgt0;
        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp1 = fc179(tgt0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
  
        temp = fc180.forward(src);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc181.forward(src);
        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt0 = tgt + tgt0;
  
        temp = fc182.forward(tgt);
        temp = relu.forward(temp);
        temp = fc183.forward(temp);
        tgt = tgt + temp;


        //decoder-layer-8


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc184.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
 
        temp = fc185.forward(tgt0);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
   
        temp = fc186.forward(tgt0);
        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt = tgt + tgt0;
        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc187.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
  
        temp = fc188.forward(src);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc189.forward(src);
        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt0 = tgt + tgt0;
        
        temp = fc190.forward(tgt);
        temp = relu.forward(temp);
        temp = fc191.forward(temp);
        tgt = tgt + temp;


        //decoder-layer-9


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc192.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc193.forward(tgt0);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
   
        temp = fc194.forward(tgt0);
        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt = tgt + tgt0;
        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc195.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
   
        temp = fc196.forward(src);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc197.forward(src);
        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt0 = tgt + tgt0;
 
        temp = fc198.forward(tgt);
        temp = relu.forward(temp);
        temp = fc199.forward(temp);
        tgt = tgt + temp;


        //decoder-layer-10


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc200.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc201.forward(tgt0);

        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
   
        temp = fc202.forward(tgt0);
  
        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt = tgt + tgt0;
        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp1 = fc203(tgt0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc204.forward(src);
     
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc205.forward(src);

        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt0 = tgt + tgt0;
        
        temp = fc206(tgt);
        temp = relu.forward(temp);
        temp = fc207.forward(temp);
        tgt = tgt + temp;


        //decoder-layer-11


        tgt0 = tgt;
        nbatches = tgt0.size(0);
 
        temp = fc208.forward(tgt0);
   
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc209(tgt0);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc210.forward(tgt0);

        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt = tgt + tgt0;
        tgt0 = tgt;
        nbatches = tgt0.size(0);
 
        temp = fc211.forward(tgt0);

        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc212.forward(src);

        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc213.forward(src);

        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt0 = tgt + tgt0;
        

        temp = fc214.forward(tgt);
        temp = F::relu(temp);
        temp = fc215.forward(temp);
        tgt = tgt + temp;


        //decoder-layer-12


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc216.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
   
        temp = fc217.forward(tgt0);

        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc218.forward(tgt0);

        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt = tgt + tgt0;
        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp1 = fc219(tgt0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc220.forward(src);

        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc221.forward(src);

        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt0 = tgt + tgt0;
        

        temp = fc222.forward(tgt);
        temp = relu.forward(temp);
        temp = fc223.forward(temp);
        tgt = tgt + temp;


        //decoder-layer-13


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc224.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
 
        temp = fc225.forward(tgt0);
 
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc226(tgt0);
        temp3 = temp3.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt = tgt + tgt0;
        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc227.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc228.forward(src);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc229.forward(src);
        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt0 = tgt + tgt0;
        
        temp = fc230.forward(tgt);
        temp = relu.forward(temp);
        temp = fc231.forward(temp);
        tgt = tgt + temp;


        //decoder-layer-14


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc232.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
      
        temp = fc233.forward(tgt0);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc234.forward(tgt0);
        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt = tgt + tgt0;
        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc235.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc236.forward(src);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
    
        temp = fc237.forward(src);
        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt0 = tgt + tgt0;
        
        temp = fc238.forward(tgt);
        temp = relu.forward(temp);
        temp = fc239.forward(temp);
        tgt = tgt + temp;


        //decoder-layer-15


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc240.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
    
        temp = fc241.forward(tgt0);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
    
        temp = fc242.forward(tgt0);
        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt = tgt + tgt0;
        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc243.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
 
        temp = fc244.forward(src);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
  
        temp = fc245.forward(src);
        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt0 = tgt + tgt0;
        
        temp = fc246.forward(tgt);
        temp = relu.forward(temp);
        temp = fc247.forward(temp);
        tgt = tgt + temp;


        //decoder-layer-16


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc248.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
 
        temp = fc249.forward(tgt0);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc250(tgt0);
        temp3 = temp3.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt = tgt + tgt0;
        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc251.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
 
        temp = fc252.forward(src);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc253.forward(src);
        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt0 = tgt + tgt0;
        
        temp = fc254.forward(tgt);
        temp = relu.forward(temp);
        temp = fc255.forward(temp);
        tgt = tgt + temp;


        //decoder-layer-17


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc256.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
   
        temp = fc257.forward(tgt0);

        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
    
        temp = fc258.forward(tgt0);
        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt = tgt + tgt0;
        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp1 = fc259(tgt0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc260.forward(src);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
    
        temp = fc261.forward(src);

        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt0 = tgt + tgt0;
        
        temp = fc262(tgt);
        temp = F::relu(temp);
        temp = fc263.forward(temp);
        tgt = tgt + temp;


        //decoder-layer-18


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc264.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        
        temp = fc265.forward(tgt0);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc266(tgt0);
        temp3 = temp3.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt = tgt + tgt0;
        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc267.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
       
        temp = fc268.forward(src);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
      
        temp = fc269.forward(src);
        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt0 = tgt + tgt0;
        
        temp = fc270.forward(tgt);
        temp = relu.forward(temp);
        temp = fc271.forward(temp);
        tgt = tgt + temp;


        //decoder-layer-19


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc272.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
 
        temp = fc273.forward(tgt0);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
    
        temp = fc274.forward(tgt0);
        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt = tgt + tgt0;
        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc275.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
       
        temp = fc276.forward(src);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
      
        temp = fc277.forward(src);
        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt0 = tgt + tgt0;
        
        temp = fc278.forward(tgt);
        temp = relu.forward(temp);
        temp = fc279(temp);
        tgt = tgt + temp;


        //decoder-layer-20


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc280.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
     
        temp = fc281.forward(tgt0);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
     
        temp = fc282.forward(tgt0);

        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt = tgt + tgt0;
        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc283.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
    
        temp = fc284.forward(src);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
      
        temp = fc285.forward(src);
        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt0 = tgt + tgt0;
        
        temp = fc286(tgt);
        temp = relu.forward(temp);
        temp = fc287.forward(temp);
        tgt = tgt + temp;


        //decoder-layer-21


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc288.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
       
        temp = fc289.forward(tgt0);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc290(tgt0);
        temp3 = temp3.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt = tgt + tgt0;
        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc291.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
 
        temp = fc292.forward(src);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
      
        temp = fc293.forward(src);
        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt0 = tgt + tgt0;
        
        temp = fc294.forward(tgt);
        temp = F::relu(temp);
        temp = fc295.forward(temp);
        tgt = tgt + temp;

        //decoder-layer-22


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc296.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
      
        temp = fc297.forward(tgt0);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
     
        temp = fc298.forward(tgt0);
        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt = tgt + tgt0;
        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc299.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
  
        temp = fc300.forward(src);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
      
        temp = fc301.forward(src);
        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt0 = tgt + tgt0;
        
        temp = fc302.forward(tgt);
        temp = relu.forward(temp);
        temp = fc303(temp);
        tgt = tgt + temp;


        //decoder-layer-23


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc304.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
       
        temp = fc305.forward(tgt0);
    
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
  
        temp = fc306.forward(tgt0);
        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt = tgt + tgt0;
        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc307.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
    
        temp = fc308.forward(src);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc309(src);
        temp3 = temp3.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt0 = tgt + tgt0;
        
        temp = fc310.forward(tgt);
        temp = F::relu(temp);
        temp = fc311(temp);
        tgt = tgt + temp;

        return tgt;
    }

    void morphpara(){
        // nothing to do, inherit from virtual func
    }
};


class trans_gpu_part1 : public trans_part
{
public:
    operator1 relu;
    operator7 fc0;
    operator7 fc1;
    operator7 fc2;
    operator7 fc3;
    torch::nn::Linear fc4;
    operator7 fc5;
    operator7 fc6;
    operator7 fc7;
    operator7 fc8;
    torch::nn::Linear fc9;
    trans_gpu_part1():
        fc0(d_model, d_model),
        fc1(d_model, d_model),
        fc2(d_model, d_model),
        fc3(d_model, d_ff),
        fc4(d_ff, d_model),
        fc5(d_model, d_model),
        fc6(d_model, d_model),
        fc7(d_model, d_model),
        fc8(d_model, d_ff),
        fc9(d_ff, d_model)
        {
            relu.to(at::kCUDA);
            fc0.to(at::kCUDA);
            fc1.to(at::kCUDA);
            fc2.to(at::kCUDA);
            fc3.to(at::kCUDA);
            fc4->to(at::kCUDA);
            fc5.to(at::kCUDA);
            fc6.to(at::kCUDA);
            fc7.to(at::kCUDA);
            fc8.to(at::kCUDA);
            fc9->to(at::kCUDA);
        }
    
    torch::Tensor forward(torch::Tensor src, torch::Tensor tgt)
    {
        // std::cout<<"trans_gpu_part1 forward"<<std::endl;
        int nbatches;
        torch::Tensor temp;
        torch::Tensor temp0;
        torch::Tensor temp1;
        torch::Tensor temp2;
        torch::Tensor temp3;
        torch::Tensor src0;
        torch::Tensor tgt0;

        // temp = fc3.forward(src);
        // temp = relu.forward(temp);
        // temp = fc4(temp);
        // src = src + temp;

        //encoder-layer-1
        src0 = src;
        nbatches = src0.size(0);
        temp = fc5.forward(src0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = fc6.forward(src0);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = fc7.forward(src0);
        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        src0 = temp.div(sqrt_dk);
        src0 = src0.softmax(3);
        src0 = src0.matmul(temp3);
        src0 = src0.transpose(1, 2);
        src0 = src0.contiguous();
        src0 = src0.view( {nbatches, -1, nheads * d_k} )[0];
        
        src = src + src0;
        temp = fc8.forward(src);
        temp = relu.forward(temp);
        temp = fc9(temp);
        src = src + temp;

        return src;
    }
    void morphpara(){
        fc0.morph();
        fc1.morph();
        fc2.morph();
        fc3.morph();
        fc4->weight = fc4->weight * scalar;
        fc4->bias = fc4->bias * scalar;
        fc5.morph();
        fc6.morph();
        fc7.morph();
        fc8.morph();
        fc9->weight = fc9->weight * scalar;
        fc9->bias = fc9->bias * scalar;
    }
};

class trans_gpu_part2 : public trans_part
{
public:
    operator1 relu;
    operator7 fc0;
    operator7 fc1;
    operator7 fc2;
    operator7 fc3;
    torch::nn::Linear fc4;
    operator7 fc5;
    operator7 fc6;
    operator7 fc7;
    operator7 fc8;
    torch::nn::Linear fc9;
    operator7 fc10;
    operator7 fc11;
    operator7 fc12;
    operator7 fc13;
    torch::nn::Linear fc14;
    operator7 fc15;
    operator7 fc16;
    operator7 fc17;
    torch::nn::Linear fc18;
    operator7 fc19;
    operator7 fc20;
    operator7 fc21;
    operator7 fc22;
    operator7 fc23;
    operator7 fc24;
    torch::nn::Linear fc25;
    operator7 fc26;
    operator7 fc27;
    operator7 fc28;
    operator7 fc29;
    torch::nn::Linear fc30;
    torch::nn::Linear fc31;
    torch::nn::Linear fc32;
    torch::nn::Linear fc33;
    operator7 fc34;
    operator7 fc35;
    operator7 fc36;
    torch::nn::Linear fc37;
    operator7 fc38;
    operator7 fc39;
    operator7 fc40;
    torch::nn::Linear fc41;
    operator7 fc42;
    torch::nn::Linear fc43;
    operator7 fc44;
    torch::nn::Linear fc45;
    operator7 fc46;
    operator7 fc47;
    operator7 fc48;
    torch::nn::Linear fc49;
    operator7 fc50;
    operator7 fc51;
    operator7 fc52;
    operator7 fc53;
    torch::nn::Linear fc54;
    operator7 fc55;
    torch::nn::Linear fc56;
    operator7 fc57;
    operator7 fc58;
    torch::nn::Linear fc59;
    operator7 fc60;
    operator7 fc61;
    operator7 fc62;
    operator7 fc63;
    operator7 fc64;
    operator7 fc65;
    torch::nn::Linear fc66;
    operator7 fc67;
    operator7 fc68;
    operator7 fc69;
    torch::nn::Linear fc70;
    operator7 fc71;
    operator7 fc72;
    operator7 fc73;
    operator7 fc74;
    operator7 fc75;
    operator7 fc76;
    operator7 fc77;
    operator7 fc78;
    operator7 fc79;
    torch::nn::Linear fc80;
    operator7 fc81;
    operator7 fc82;
    operator7 fc83;
    operator7 fc84;
    operator7 fc85;
    operator7 fc86;
    torch::nn::Linear fc87;
    operator7 fc88;
    operator7 fc89;
    torch::nn::Linear fc90;
    operator7 fc91;
    operator7 fc92;
    operator7 fc93;
    torch::nn::Linear fc94;
    torch::nn::Linear fc95;
    operator7 fc96;
    operator7 fc97;
    operator7 fc98;
    operator7 fc99;
    operator7 fc100;
    torch::nn::Linear fc101;
    torch::nn::Linear fc102;
    operator7 fc103;
    operator7 fc104;
    operator7 fc105;
    operator7 fc106;
    operator7 fc107;
    operator7 fc108;
    operator7 fc109;
    operator7 fc110;
    operator7 fc111;
    operator7 fc112;
    operator7 fc113;
    operator7 fc114;
    operator7 fc115;
    operator7 fc116;
    operator7 fc117;
    operator7 fc118;
    operator7 fc119;
    operator7 fc120;
    operator7 fc121;
    operator7 fc122;
    operator7 fc123;
    operator7 fc124;
    operator7 fc125;
    operator7 fc126;
    torch::nn::Linear fc127;
    torch::nn::Linear fc128;
    torch::nn::Linear fc129;
    operator7 fc130;
    operator7 fc131;
    operator7 fc132;
    operator7 fc133;
    operator7 fc134;
    operator7 fc135;
    operator7 fc136;
    operator7 fc137;
    operator7 fc138;
    operator7 fc139;
    operator7 fc140;
    torch::nn::Linear fc141;
    torch::nn::Linear fc142;
    operator7 fc143;
    operator7 fc144;
    torch::nn::Linear fc145;
    operator7 fc146;
    operator7 fc147;
    operator7 fc148;
    torch::nn::Linear fc149;
    operator7 fc150;
    operator7 fc151;
    operator7 fc152;
    operator7 fc153;
    operator7 fc154;
    operator7 fc155;
    operator7 fc156;
    operator7 fc157;
    operator7 fc158;
    operator7 fc159;
    operator7 fc160;
    operator7 fc161;
    torch::nn::Linear fc162;
    operator7 fc163;
    torch::nn::Linear fc164;
    operator7 fc165;
    operator7 fc166;
    operator7 fc167;
    operator7 fc168;
    operator7 fc169;
    torch::nn::Linear fc170;
    operator7 fc171;
    operator7 fc172;
    torch::nn::Linear fc173;
    operator7 fc174;
    operator7 fc175;
    operator7 fc176;
    operator7 fc177;
    operator7 fc178;
    torch::nn::Linear fc179;
    operator7 fc180;
    operator7 fc181;
    operator7 fc182;
    operator7 fc183;
    operator7 fc184;
    operator7 fc185;
    operator7 fc186;
    operator7 fc187;
    operator7 fc188;
    operator7 fc189;
    operator7 fc190;
    operator7 fc191;
    operator7 fc192;
    operator7 fc193;
    operator7 fc194;
    operator7 fc195;
    operator7 fc196;
    operator7 fc197;
    operator7 fc198;
    operator7 fc199;
    operator7 fc200;
    operator7 fc201;
    operator7 fc202;
    torch::nn::Linear fc203;
    operator7 fc204;
    operator7 fc205;
    torch::nn::Linear fc206;
    operator7 fc207;
    operator7 fc208;
    torch::nn::Linear fc209;
    operator7 fc210;
    operator7 fc211;
    operator7 fc212;
    operator7 fc213;
    operator7 fc214;
    operator7 fc215;
    operator7 fc216;
    operator7 fc217;
    operator7 fc218;
    torch::nn::Linear fc219;
    operator7 fc220;
    operator7 fc221;
    operator7 fc222;
    operator7 fc223;
    operator7 fc224;
    operator7 fc225;
    torch::nn::Linear fc226;
    operator7 fc227;
    operator7 fc228;
    operator7 fc229;
    operator7 fc230;
    operator7 fc231;
    operator7 fc232;
    operator7 fc233;
    operator7 fc234;
    operator7 fc235;
    operator7 fc236;
    operator7 fc237;
    operator7 fc238;
    operator7 fc239;
    operator7 fc240;
    operator7 fc241;
    operator7 fc242;
    operator7 fc243;
    operator7 fc244;
    operator7 fc245;
    operator7 fc246;
    operator7 fc247;
    operator7 fc248;
    operator7 fc249;
    torch::nn::Linear fc250;
    operator7 fc251;
    operator7 fc252;
    operator7 fc253;
    operator7 fc254;
    operator7 fc255;
    operator7 fc256;
    operator7 fc257;
    operator7 fc258;
    torch::nn::Linear fc259;
    operator7 fc260;
    operator7 fc261;
    torch::nn::Linear fc262;
    operator7 fc263;
    operator7 fc264;
    operator7 fc265;
    torch::nn::Linear fc266;
    operator7 fc267;
    operator7 fc268;
    operator7 fc269;
    operator7 fc270;
    operator7 fc271;
    operator7 fc272;
    operator7 fc273;
    operator7 fc274;
    operator7 fc275;
    operator7 fc276;
    operator7 fc277;
    operator7 fc278;
    torch::nn::Linear fc279;
    operator7 fc280;
    operator7 fc281;
    operator7 fc282;
    operator7 fc283;
    operator7 fc284;
    operator7 fc285;
    torch::nn::Linear fc286;
    operator7 fc287;
    operator7 fc288;
    operator7 fc289;
    torch::nn::Linear fc290;
    operator7 fc291;
    operator7 fc292;
    operator7 fc293;
    operator7 fc294;
    operator7 fc295;
    operator7 fc296;
    operator7 fc297;
    operator7 fc298;
    operator7 fc299;
    operator7 fc300;
    operator7 fc301;
    operator7 fc302;
    torch::nn::Linear fc303;
    operator7 fc304;
    operator7 fc305;
    operator7 fc306;
    operator7 fc307;
    operator7 fc308;
    torch::nn::Linear fc309;
    operator7 fc310;
    torch::nn::Linear fc311;

    trans_gpu_part2():
        fc0(d_model, d_model),
        fc1(d_model, d_model),
        fc2(d_model, d_model),
        fc3(d_model, d_ff),
        fc4(d_ff, d_model),
        fc5(d_model, d_model),
        fc6(d_model, d_model),
        fc7(d_model, d_model),
        fc8(d_model, d_ff),
        fc9(d_ff, d_model),
        fc10(d_model, d_model),
        fc11(d_model, d_model),
        fc12(d_model, d_model),
        fc13(d_model, d_ff),
        fc14(d_ff, d_model),
        fc15(d_model, d_model),
        fc16(d_model, d_model),
        fc17(d_model, d_model),
        fc18(d_model, d_ff),
        fc19(d_ff, d_model),
        fc20(d_model, d_model),
        fc21(d_model, d_model),
        fc22(d_model, d_model),
        fc23(d_model, d_ff),
        fc24(d_ff, d_model),
        fc25(d_model, d_model),
        fc26(d_model, d_model),
        fc27(d_model, d_model),
        fc28(d_model, d_ff),
        fc29(d_ff, d_model),
        fc30(d_model, d_model),
        fc31(d_model, d_model),
        fc32(d_model, d_model),
        fc33(d_model, d_ff),
        fc34(d_ff, d_model),
        fc35(d_model, d_model),
        fc36(d_model, d_model),
        fc37(d_model, d_model),
        fc38(d_model, d_ff),
        fc39(d_ff, d_model),
        fc40(d_model, d_model),
        fc41(d_model, d_model),
        fc42(d_model, d_model),
        fc43(d_model, d_ff),
        fc44(d_ff, d_model),
        fc45(d_model, d_model),
        fc46(d_model, d_model),
        fc47(d_model, d_model),
        fc48(d_model, d_ff),
        fc49(d_ff, d_model),
        fc50(d_model, d_model),
        fc51(d_model, d_model),
        fc52(d_model, d_model),
        fc53(d_model, d_ff),
        fc54(d_ff, d_model),
        fc55(d_model, d_model),
        fc56(d_model, d_model),
        fc57(d_model, d_model),
        fc58(d_model, d_ff),
        fc59(d_ff, d_model),
        fc60(d_model, d_model),
        fc61(d_model, d_model),
        fc62(d_model, d_model),
        fc63(d_model, d_ff),
        fc64(d_ff, d_model),
        fc65(d_model, d_model),
        fc66(d_model, d_model),
        fc67(d_model, d_model),
        fc68(d_model, d_ff),
        fc69(d_ff, d_model),
        fc70(d_model, d_model),
        fc71(d_model, d_model),
        fc72(d_model, d_model),
        fc73(d_model, d_ff),
        fc74(d_ff, d_model),
        fc75(d_model, d_model),
        fc76(d_model, d_model),
        fc77(d_model, d_model),
        fc78(d_model, d_ff),
        fc79(d_ff, d_model),
        fc80(d_model, d_model),
        fc81(d_model, d_model),
        fc82(d_model, d_model),
        fc83(d_model, d_ff),
        fc84(d_ff, d_model),
        fc85(d_model, d_model),
        fc86(d_model, d_model),
        fc87(d_model, d_model),
        fc88(d_model, d_ff),
        fc89(d_ff, d_model),
        fc90(d_model, d_model),
        fc91(d_model, d_model),
        fc92(d_model, d_model),
        fc93(d_model, d_ff),
        fc94(d_ff, d_model),
        fc95(d_model, d_model),
        fc96(d_model, d_model),
        fc97(d_model, d_model),
        fc98(d_model, d_ff),
        fc99(d_ff, d_model),
        fc100(d_model, d_model),
        fc101(d_model, d_model),
        fc102(d_model, d_model),
        fc103(d_model, d_ff),
        fc104(d_ff, d_model),
        fc105(d_model, d_model),
        fc106(d_model, d_model),
        fc107(d_model, d_model),
        fc108(d_model, d_ff),
        fc109(d_ff, d_model),
        fc110(d_model, d_model),
        fc111(d_model, d_model),
        fc112(d_model, d_model),
        fc113(d_model, d_ff),
        fc114(d_ff, d_model),
        fc115(d_model, d_model),
        fc116(d_model, d_model),
        fc117(d_model, d_model),
        fc118(d_model, d_ff),
        fc119(d_ff, d_model),
        fc120(d_model, d_model),
        fc121(d_model, d_model),
        fc122(d_model, d_model),
        fc123(d_model, d_model),
        fc124(d_model, d_model),
        fc125(d_model, d_model),
        fc126(d_model, d_ff),
        fc127(d_ff, d_model),
        fc128(d_model, d_model),
        fc129(d_model, d_model),
        fc130(d_model, d_model),
        fc131(d_model, d_model),
        fc132(d_model, d_model),
        fc133(d_model, d_model),
        fc134(d_model, d_ff),
        fc135(d_ff, d_model),
        fc136(d_model, d_model),
        fc137(d_model, d_model),
        fc138(d_model, d_model),
        fc139(d_model, d_model),
        fc140(d_model, d_model),
        fc141(d_model, d_model),
        fc142(d_model, d_ff),
        fc143(d_ff, d_model),
        fc144(d_model, d_model),
        fc145(d_model, d_model),
        fc146(d_model, d_model),
        fc147(d_model, d_model),
        fc148(d_model, d_model),
        fc149(d_model, d_model),
        fc150(d_model, d_ff),
        fc151(d_ff, d_model),
        fc152(d_model, d_model),
        fc153(d_model, d_model),
        fc154(d_model, d_model),
        fc155(d_model, d_model),
        fc156(d_model, d_model),
        fc157(d_model, d_model),
        fc158(d_model, d_ff),
        fc159(d_ff, d_model),
        fc160(d_model, d_model),
        fc161(d_model, d_model),
        fc162(d_model, d_model),
        fc163(d_model, d_model),
        fc164(d_model, d_model),
        fc165(d_model, d_model),
        fc166(d_model, d_ff),
        fc167(d_ff, d_model),
        fc168(d_model, d_model),
        fc169(d_model, d_model),
        fc170(d_model, d_model),
        fc171(d_model, d_model),
        fc172(d_model, d_model),
        fc173(d_model, d_model),
        fc174(d_model, d_ff),
        fc175(d_ff, d_model),
        fc176(d_model, d_model),
        fc177(d_model, d_model),
        fc178(d_model, d_model),
        fc179(d_model, d_model),
        fc180(d_model, d_model),
        fc181(d_model, d_model),
        fc182(d_model, d_ff),
        fc183(d_ff, d_model),
        fc184(d_model, d_model),
        fc185(d_model, d_model),
        fc186(d_model, d_model),
        fc187(d_model, d_model),
        fc188(d_model, d_model),
        fc189(d_model, d_model),
        fc190(d_model, d_ff),
        fc191(d_ff, d_model),
        fc192(d_model, d_model),
        fc193(d_model, d_model),
        fc194(d_model, d_model),
        fc195(d_model, d_model),
        fc196(d_model, d_model),
        fc197(d_model, d_model),
        fc198(d_model, d_ff),
        fc199(d_ff, d_model),
        fc200(d_model, d_model),
        fc201(d_model, d_model),
        fc202(d_model, d_model),
        fc203(d_model, d_model),
        fc204(d_model, d_model),
        fc205(d_model, d_model),
        fc206(d_model, d_ff),
        fc207(d_ff, d_model),
        fc208(d_model, d_model),
        fc209(d_model, d_model),
        fc210(d_model, d_model),
        fc211(d_model, d_model),
        fc212(d_model, d_model),
        fc213(d_model, d_model),
        fc214(d_model, d_ff),
        fc215(d_ff, d_model),
        fc216(d_model, d_model),
        fc217(d_model, d_model),
        fc218(d_model, d_model),
        fc219(d_model, d_model),
        fc220(d_model, d_model),
        fc221(d_model, d_model),
        fc222(d_model, d_ff),
        fc223(d_ff, d_model),
        fc224(d_model, d_model),
        fc225(d_model, d_model),
        fc226(d_model, d_model),
        fc227(d_model, d_model),
        fc228(d_model, d_model),
        fc229(d_model, d_model),
        fc230(d_model, d_ff),
        fc231(d_ff, d_model),
        fc232(d_model, d_model),
        fc233(d_model, d_model),
        fc234(d_model, d_model),
        fc235(d_model, d_model),
        fc236(d_model, d_model),
        fc237(d_model, d_model),
        fc238(d_model, d_ff),
        fc239(d_ff, d_model),
        fc240(d_model, d_model),
        fc241(d_model, d_model),
        fc242(d_model, d_model),
        fc243(d_model, d_model),
        fc244(d_model, d_model),
        fc245(d_model, d_model),
        fc246(d_model, d_ff),
        fc247(d_ff, d_model),
        fc248(d_model, d_model),
        fc249(d_model, d_model),
        fc250(d_model, d_model),
        fc251(d_model, d_model),
        fc252(d_model, d_model),
        fc253(d_model, d_model),
        fc254(d_model, d_ff),
        fc255(d_ff, d_model),
        fc256(d_model, d_model),
        fc257(d_model, d_model),
        fc258(d_model, d_model),
        fc259(d_model, d_model),
        fc260(d_model, d_model),
        fc261(d_model, d_model),
        fc262(d_model, d_ff),
        fc263(d_ff, d_model),
        fc264(d_model, d_model),
        fc265(d_model, d_model),
        fc266(d_model, d_model),
        fc267(d_model, d_model),
        fc268(d_model, d_model),
        fc269(d_model, d_model),
        fc270(d_model, d_ff),
        fc271(d_ff, d_model),
        fc272(d_model, d_model),
        fc273(d_model, d_model),
        fc274(d_model, d_model),
        fc275(d_model, d_model),
        fc276(d_model, d_model),
        fc277(d_model, d_model),
        fc278(d_model, d_ff),
        fc279(d_ff, d_model),
        fc280(d_model, d_model),
        fc281(d_model, d_model),
        fc282(d_model, d_model),
        fc283(d_model, d_model),
        fc284(d_model, d_model),
        fc285(d_model, d_model),
        fc286(d_model, d_ff),
        fc287(d_ff, d_model),
        fc288(d_model, d_model),
        fc289(d_model, d_model),
        fc290(d_model, d_model),
        fc291(d_model, d_model),
        fc292(d_model, d_model),
        fc293(d_model, d_model),
        fc294(d_model, d_ff),
        fc295(d_ff, d_model),
        fc296(d_model, d_model),
        fc297(d_model, d_model),
        fc298(d_model, d_model),
        fc299(d_model, d_model),
        fc300(d_model, d_model),
        fc301(d_model, d_model),
        fc302(d_model, d_ff),
        fc303(d_ff, d_model),
        fc304(d_model, d_model),
        fc305(d_model, d_model),
        fc306(d_model, d_model),
        fc307(d_model, d_model),
        fc308(d_model, d_model),
        fc309(d_model, d_model),
        fc310(d_model, d_ff),
        fc311(d_ff, d_model)
        {
            relu.to(at::kCUDA);
            fc0.to(at::kCUDA);
            fc1.to(at::kCUDA);
            fc2.to(at::kCUDA);
            fc3.to(at::kCUDA);
            fc4->to(at::kCUDA);
            fc5.to(at::kCUDA);
            fc6.to(at::kCUDA);
            fc7.to(at::kCUDA);
            fc8.to(at::kCUDA);
            fc9->to(at::kCUDA);
            fc10.to(at::kCUDA);
            fc11.to(at::kCUDA);
            fc12.to(at::kCUDA);
            fc13.to(at::kCUDA);
            fc14->to(at::kCUDA);
            fc15.to(at::kCUDA);
            fc16.to(at::kCUDA);
            fc17.to(at::kCUDA);
            fc18->to(at::kCUDA);
            fc19.to(at::kCUDA);
            fc20.to(at::kCUDA);
            fc21.to(at::kCUDA);
            fc22.to(at::kCUDA);
            fc23.to(at::kCUDA);
            fc24.to(at::kCUDA);
            fc25->to(at::kCUDA);
            fc26.to(at::kCUDA);
            fc27.to(at::kCUDA);
            fc28.to(at::kCUDA);
            fc29.to(at::kCUDA);
            fc30->to(at::kCUDA);
            fc31->to(at::kCUDA);
            fc32->to(at::kCUDA);
            fc33->to(at::kCUDA);
            fc34.to(at::kCUDA);
            fc35.to(at::kCUDA);
            fc36.to(at::kCUDA);
            fc37->to(at::kCUDA);
            fc38.to(at::kCUDA);
            fc39.to(at::kCUDA);
            fc40.to(at::kCUDA);
            fc41->to(at::kCUDA);
            fc42.to(at::kCUDA);
            fc43->to(at::kCUDA);
            fc44.to(at::kCUDA);
            fc45->to(at::kCUDA);
            fc46.to(at::kCUDA);
            fc47.to(at::kCUDA);
            fc48.to(at::kCUDA);
            fc49->to(at::kCUDA);
            fc50.to(at::kCUDA);
            fc51.to(at::kCUDA);
            fc52.to(at::kCUDA);
            fc53.to(at::kCUDA);
            fc54->to(at::kCUDA);
            fc55.to(at::kCUDA);
            fc56->to(at::kCUDA);
            fc57.to(at::kCUDA);
            fc58.to(at::kCUDA);
            fc59->to(at::kCUDA);
            fc60.to(at::kCUDA);
            fc61.to(at::kCUDA);
            fc62.to(at::kCUDA);
            fc63.to(at::kCUDA);
            fc64.to(at::kCUDA);
            fc65.to(at::kCUDA);
            fc66->to(at::kCUDA);
            fc67.to(at::kCUDA);
            fc68.to(at::kCUDA);
            fc69.to(at::kCUDA);
            fc70->to(at::kCUDA);
            fc71.to(at::kCUDA);
            fc72.to(at::kCUDA);
            fc73.to(at::kCUDA);
            fc74.to(at::kCUDA);
            fc75.to(at::kCUDA);
            fc76.to(at::kCUDA);
            fc77.to(at::kCUDA);
            fc78.to(at::kCUDA);
            fc79.to(at::kCUDA);
            fc80->to(at::kCUDA);
            fc81.to(at::kCUDA);
            fc82.to(at::kCUDA);
            fc83.to(at::kCUDA);
            fc84.to(at::kCUDA);
            fc85.to(at::kCUDA);
            fc86.to(at::kCUDA);
            fc87->to(at::kCUDA);
            fc88.to(at::kCUDA);
            fc89.to(at::kCUDA);
            fc90->to(at::kCUDA);
            fc91.to(at::kCUDA);
            fc92.to(at::kCUDA);
            fc93.to(at::kCUDA);
            fc94->to(at::kCUDA);
            fc95->to(at::kCUDA);
            fc96.to(at::kCUDA);
            fc97.to(at::kCUDA);
            fc98.to(at::kCUDA);
            fc99.to(at::kCUDA);
            fc100.to(at::kCUDA);
            fc101->to(at::kCUDA);
            fc102->to(at::kCUDA);
            fc103.to(at::kCUDA);
            fc104.to(at::kCUDA);
            fc105.to(at::kCUDA);
            fc106.to(at::kCUDA);
            fc107.to(at::kCUDA);
            fc108.to(at::kCUDA);
            fc109.to(at::kCUDA);
            fc110.to(at::kCUDA);
            fc111.to(at::kCUDA);
            fc112.to(at::kCUDA);
            fc113.to(at::kCUDA);
            fc114.to(at::kCUDA);
            fc115.to(at::kCUDA);
            fc116.to(at::kCUDA);
            fc117.to(at::kCUDA);
            fc118.to(at::kCUDA);
            fc119.to(at::kCUDA);
            fc120.to(at::kCUDA);
            fc121.to(at::kCUDA);
            fc122.to(at::kCUDA);
            fc123.to(at::kCUDA);
            fc124.to(at::kCUDA);
            fc125.to(at::kCUDA);
            fc126.to(at::kCUDA);
            fc127->to(at::kCUDA);
            fc128->to(at::kCUDA);
            fc129->to(at::kCUDA);
            fc130.to(at::kCUDA);
            fc131.to(at::kCUDA);
            fc132.to(at::kCUDA);
            fc133.to(at::kCUDA);
            fc134.to(at::kCUDA);
            fc135.to(at::kCUDA);
            fc136.to(at::kCUDA);
            fc137.to(at::kCUDA);
            fc138.to(at::kCUDA);
            fc139.to(at::kCUDA);
            fc140.to(at::kCUDA);
            fc141->to(at::kCUDA);
            fc142->to(at::kCUDA);
            fc143.to(at::kCUDA);
            fc144.to(at::kCUDA);
            fc145->to(at::kCUDA);
            fc146.to(at::kCUDA);
            fc147.to(at::kCUDA);
            fc148.to(at::kCUDA);
            fc149->to(at::kCUDA);
            fc150.to(at::kCUDA);
            fc151.to(at::kCUDA);
            fc152.to(at::kCUDA);
            fc153.to(at::kCUDA);
            fc154.to(at::kCUDA);
            fc155.to(at::kCUDA);
            fc156.to(at::kCUDA);
            fc157.to(at::kCUDA);
            fc158.to(at::kCUDA);
            fc159.to(at::kCUDA);
            fc160.to(at::kCUDA);
            fc161.to(at::kCUDA);
            fc162->to(at::kCUDA);
            fc163.to(at::kCUDA);
            fc164->to(at::kCUDA);
            fc165.to(at::kCUDA);
            fc166.to(at::kCUDA);
            fc167.to(at::kCUDA);
            fc168.to(at::kCUDA);
            fc169.to(at::kCUDA);
            fc170->to(at::kCUDA);
            fc171.to(at::kCUDA);
            fc172.to(at::kCUDA);
            fc173->to(at::kCUDA);
            fc174.to(at::kCUDA);
            fc175.to(at::kCUDA);
            fc176.to(at::kCUDA);
            fc177.to(at::kCUDA);
            fc178.to(at::kCUDA);
            fc179->to(at::kCUDA);
            fc180.to(at::kCUDA);
            fc181.to(at::kCUDA);
            fc182.to(at::kCUDA);
            fc183.to(at::kCUDA);
            fc184.to(at::kCUDA);
            fc185.to(at::kCUDA);
            fc186.to(at::kCUDA);
            fc187.to(at::kCUDA);
            fc188.to(at::kCUDA);
            fc189.to(at::kCUDA);
            fc190.to(at::kCUDA);
            fc191.to(at::kCUDA);
            fc192.to(at::kCUDA);
            fc193.to(at::kCUDA);
            fc194.to(at::kCUDA);
            fc195.to(at::kCUDA);
            fc196.to(at::kCUDA);
            fc197.to(at::kCUDA);
            fc198.to(at::kCUDA);
            fc199.to(at::kCUDA);
            fc200.to(at::kCUDA);
            fc201.to(at::kCUDA);
            fc202.to(at::kCUDA);
            fc203->to(at::kCUDA);
            fc204.to(at::kCUDA);
            fc205.to(at::kCUDA);
            fc206->to(at::kCUDA);
            fc207.to(at::kCUDA);
            fc208.to(at::kCUDA);
            fc209->to(at::kCUDA);
            fc210.to(at::kCUDA);
            fc211.to(at::kCUDA);
            fc212.to(at::kCUDA);
            fc213.to(at::kCUDA);
            fc214.to(at::kCUDA);
            fc215.to(at::kCUDA);
            fc216.to(at::kCUDA);
            fc217.to(at::kCUDA);
            fc218.to(at::kCUDA);
            fc219->to(at::kCUDA);
            fc220.to(at::kCUDA);
            fc221.to(at::kCUDA);
            fc222.to(at::kCUDA);
            fc223.to(at::kCUDA);
            fc224.to(at::kCUDA);
            fc225.to(at::kCUDA);
            fc226->to(at::kCUDA);
            fc227.to(at::kCUDA);
            fc228.to(at::kCUDA);
            fc229.to(at::kCUDA);
            fc230.to(at::kCUDA);
            fc231.to(at::kCUDA);
            fc232.to(at::kCUDA);
            fc233.to(at::kCUDA);
            fc234.to(at::kCUDA);
            fc235.to(at::kCUDA);
            fc236.to(at::kCUDA);
            fc237.to(at::kCUDA);
            fc238.to(at::kCUDA);
            fc239.to(at::kCUDA);
            fc240.to(at::kCUDA);
            fc241.to(at::kCUDA);
            fc242.to(at::kCUDA);
            fc243.to(at::kCUDA);
            fc244.to(at::kCUDA);
            fc245.to(at::kCUDA);
            fc246.to(at::kCUDA);
            fc247.to(at::kCUDA);
            fc248.to(at::kCUDA);
            fc249.to(at::kCUDA);
            fc250->to(at::kCUDA);
            fc251.to(at::kCUDA);
            fc252.to(at::kCUDA);
            fc253.to(at::kCUDA);
            fc254.to(at::kCUDA);
            fc255.to(at::kCUDA);
            fc256.to(at::kCUDA);
            fc257.to(at::kCUDA);
            fc258.to(at::kCUDA);
            fc259->to(at::kCUDA);
            fc260.to(at::kCUDA);
            fc261.to(at::kCUDA);
            fc262->to(at::kCUDA);
            fc263.to(at::kCUDA);
            fc264.to(at::kCUDA);
            fc265.to(at::kCUDA);
            fc266->to(at::kCUDA);
            fc267.to(at::kCUDA);
            fc268.to(at::kCUDA);
            fc269.to(at::kCUDA);
            fc270.to(at::kCUDA);
            fc271.to(at::kCUDA);
            fc272.to(at::kCUDA);
            fc273.to(at::kCUDA);
            fc274.to(at::kCUDA);
            fc275.to(at::kCUDA);
            fc276.to(at::kCUDA);
            fc277.to(at::kCUDA);
            fc278.to(at::kCUDA);
            fc279->to(at::kCUDA);
            fc280.to(at::kCUDA);
            fc281.to(at::kCUDA);
            fc282.to(at::kCUDA);
            fc283.to(at::kCUDA);
            fc284.to(at::kCUDA);
            fc285.to(at::kCUDA);
            fc286->to(at::kCUDA);
            fc287.to(at::kCUDA);
            fc288.to(at::kCUDA);
            fc289.to(at::kCUDA);
            fc290->to(at::kCUDA);
            fc291.to(at::kCUDA);
            fc292.to(at::kCUDA);
            fc293.to(at::kCUDA);
            fc294.to(at::kCUDA);
            fc295.to(at::kCUDA);
            fc296.to(at::kCUDA);
            fc297.to(at::kCUDA);
            fc298.to(at::kCUDA);
            fc299.to(at::kCUDA);
            fc300.to(at::kCUDA);
            fc301.to(at::kCUDA);
            fc302.to(at::kCUDA);
            fc303->to(at::kCUDA);
            fc304.to(at::kCUDA);
            fc305.to(at::kCUDA);
            fc306.to(at::kCUDA);
            fc307.to(at::kCUDA);
            fc308.to(at::kCUDA);
            fc309->to(at::kCUDA);
            fc310.to(at::kCUDA);
            fc311->to(at::kCUDA);
        }
    
    torch::Tensor forward(torch::Tensor src, torch::Tensor tgt)
    {
        int nbatches;
        torch::Tensor temp;
        torch::Tensor temp0;
        torch::Tensor temp1;
        torch::Tensor temp2;
        torch::Tensor temp3;
        torch::Tensor src0;
        torch::Tensor tgt0;

        // temp = fc13.forward(src);
        // temp = relu.forward(temp);
        // temp = fc14(temp);
        // src = src + temp;
        //encoder-layer-3
    
        src0 = src;
        nbatches = src0.size(0);
        temp = fc15.forward(src0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
       
        temp = fc16.forward(src0);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
   
        temp = fc17.forward(src0);
        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        src0 = temp.div(sqrt_dk);
        src0 = src0.softmax(3);
        src0 = src0.matmul(temp3);
        src0 = src0.transpose(1, 2);
        src0 = src0.contiguous();
        src0 = src0.view( {nbatches, -1, nheads * d_k} )[0];
        
        src = src + src0;
        temp = fc18(src);
        temp = relu.forward(temp);
        temp = fc19.forward(temp);
        src = src + temp;
 


        //encoder-layer-4


        src0 = src;
        nbatches = src0.size(0);
        temp = fc20.forward(src0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
    
        temp = fc21.forward(src0);
   
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc22.forward(src0);

        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        src0 = temp.div(sqrt_dk);
        src0 = src0.softmax(3);
        src0 = src0.matmul(temp3);
        src0 = src0.transpose(1, 2);
        src0 = src0.contiguous();
        src0 = src0.view( {nbatches, -1, nheads * d_k} )[0];
        
        src = src + src0;
  
        temp = fc23.forward(src);
        temp = relu.forward(temp);
        temp = fc24.forward(temp);
        src = src + temp;

        //encoder-layer-5


        src0 = src;
        nbatches = src0.size(0);
        temp1 = fc25(src0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
   
        temp = fc26.forward(src0);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
     
        temp = fc27.forward(src0);

        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        src0 = temp.div(sqrt_dk);
        src0 = src0.softmax(3);
        src0 = src0.matmul(temp3);
        src0 = src0.transpose(1, 2);
        src0 = src0.contiguous();
        src0 = src0.view( {nbatches, -1, nheads * d_k} )[0];
        
        src = src + src0;
        temp = fc28.forward(src);
        temp = relu.forward(temp);
        temp = fc29.forward(temp);
        src = src + temp;


        //encoder-layer-6


        src0 = src;
        nbatches = src0.size(0);
        temp1 = fc30(src0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc31(src0);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc32(src0);
        temp3 = temp3.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        src0 = temp.div(sqrt_dk);
        src0 = src0.softmax(3);
        src0 = src0.matmul(temp3);
        src0 = src0.transpose(1, 2);
        src0 = src0.contiguous();
        src0 = src0.view( {nbatches, -1, nheads * d_k} )[0];
        
        src = src + src0;
        temp = fc33(src);
        temp = F::relu(temp);
        temp = fc34.forward(temp);
        src = src + temp;


        //encoder-layer-7


        src0 = src;
        nbatches = src0.size(0);

        temp = fc35.forward(src0);
  
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc36.forward(src0);

        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc37(src0);
        temp3 = temp3.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        src0 = temp.div(sqrt_dk);
        src0 = src0.softmax(3);
        src0 = src0.matmul(temp3);
        src0 = src0.transpose(1, 2);
        src0 = src0.contiguous();
        src0 = src0.view( {nbatches, -1, nheads * d_k} )[0];
        
        src = src + src0;
    
        temp = fc38.forward(src);
        temp = relu.forward(temp);
        temp = fc39.forward(temp);
        src = src + temp;


        //encoder-layer-8


        src0 = src;
        nbatches = src0.size(0);
        temp = fc40.forward(src0);
   
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc41(src0);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc42.forward(src0);

        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        src0 = temp.div(sqrt_dk);
        src0 = src0.softmax(3);
        src0 = src0.matmul(temp3);
        src0 = src0.transpose(1, 2);
        src0 = src0.contiguous();
        src0 = src0.view( {nbatches, -1, nheads * d_k} )[0];
        
        src = src + src0;
        temp = fc43(src);
        temp = relu.forward(temp);
        temp = fc44.forward(temp);
        src = src + temp;


        //encoder-layer-9


        src0 = src;
        nbatches = src0.size(0);
        temp1 = fc45(src0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
   
        temp = fc46.forward(src0);

        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc47.forward(src0);

        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        src0 = temp.div(sqrt_dk);
        src0 = src0.softmax(3);
        src0 = src0.matmul(temp3);
        src0 = src0.transpose(1, 2);
        src0 = src0.contiguous();
        src0 = src0.view( {nbatches, -1, nheads * d_k} )[0];
        
        src = src + src0;
   
        temp = fc48.forward(src);
        temp = relu.forward(temp);
        temp = fc49(temp);
        src = src + temp;



        //encoder-layer-10


        src0 = src;
        nbatches = src0.size(0);
        temp = fc50.forward(src0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
  
        temp = fc51.forward(src0);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
      
        temp = fc52.forward(src0);

        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        src0 = temp.div(sqrt_dk);
        src0 = src0.softmax(3);
        src0 = src0.matmul(temp3);
        src0 = src0.transpose(1, 2);
        src0 = src0.contiguous();
        src0 = src0.view( {nbatches, -1, nheads * d_k} )[0];
        
        src = src + src0;

        temp = fc53.forward(src);
        temp = relu.forward(temp);
        temp = fc54(temp);
        src = src + temp;


        //encoder-layer-11


        src0 = src;
        nbatches = src0.size(0);
        temp = fc55.forward(src0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc56(src0);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
     
        temp = fc57.forward(src0);
    
        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        src0 = temp.div(sqrt_dk);
        src0 = src0.softmax(3);
        src0 = src0.matmul(temp3);
        src0 = src0.transpose(1, 2);
        src0 = src0.contiguous();
        src0 = src0.view( {nbatches, -1, nheads * d_k} )[0];
        
        src = src + src0;
        temp = fc58.forward(src);
        temp = F::relu(temp);
        temp = fc59(temp);
        src = src + temp;


        //encoder-layer-12


        src0 = src;
        nbatches = src0.size(0);
        temp = fc60.forward(src0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
    
        temp = fc61.forward(src0);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
    
        temp = fc62.forward(src0);

        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        src0 = temp.div(sqrt_dk);
        src0 = src0.softmax(3);
        src0 = src0.matmul(temp3);
        src0 = src0.transpose(1, 2);
        src0 = src0.contiguous();
        src0 = src0.view( {nbatches, -1, nheads * d_k} )[0];
        
        src = src + src0;
        temp = fc63.forward(src);
        temp = relu.forward(temp);
        temp = fc64.forward(temp);
        src = src + temp;


        //encoder-layer-13


        src0 = src;
        nbatches = src0.size(0);
        temp = fc65.forward(src0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc66(src0);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
     
        temp = fc67.forward(src0);
        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        src0 = temp.div(sqrt_dk);
        src0 = src0.softmax(3);
        src0 = src0.matmul(temp3);
        src0 = src0.transpose(1, 2);
        src0 = src0.contiguous();
        src0 = src0.view( {nbatches, -1, nheads * d_k} )[0];
        
        src = src + src0;
        temp = fc68.forward(src);
        temp = relu.forward(temp);
        temp = fc69.forward(temp);
        src = src + temp;


        //encoder-layer-14


        src0 = src;
        nbatches = src0.size(0);
        temp1 = fc70(src0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
     
        temp = fc71.forward(src0);
   
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
 
        temp = fc72.forward(src0);

        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        src0 = temp.div(sqrt_dk);
        src0 = src0.softmax(3);
        src0 = src0.matmul(temp3);
        src0 = src0.transpose(1, 2);
        src0 = src0.contiguous();
        src0 = src0.view( {nbatches, -1, nheads * d_k} )[0];
        
        src = src + src0;

        temp = fc73.forward(src);
        temp = relu.forward(temp);
        temp = fc74.forward(temp);
        src = src + temp;

        //encoder-layer-15


        src0 = src;
        nbatches = src0.size(0);
        temp = fc75.forward(src0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc76.forward(src0);

        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc77.forward(src0);

        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        src0 = temp.div(sqrt_dk);
        src0 = src0.softmax(3);
        src0 = src0.matmul(temp3);
        src0 = src0.transpose(1, 2);
        src0 = src0.contiguous();
        src0 = src0.view( {nbatches, -1, nheads * d_k} )[0];
        
        src = src + src0;

        temp = fc78.forward(src);
        temp = relu.forward(temp);
        temp = fc79.forward(temp);
        src = src + temp;


        //encoder-layer-16


        src0 = src;
        nbatches = src0.size(0);
        temp1 = fc80(src0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc81.forward(src0);

        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
  
        temp = fc82.forward(src0);
  
        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        src0 = temp.div(sqrt_dk);
        src0 = src0.softmax(3);
        src0 = src0.matmul(temp3);
        src0 = src0.transpose(1, 2);
        src0 = src0.contiguous();
        src0 = src0.view( {nbatches, -1, nheads * d_k} )[0];
        
        src = src + src0;
        temp = fc83.forward(src);
        temp = relu.forward(temp);
        temp = fc84.forward(temp);
        src = src + temp;


        //encoder-layer-17


        src0 = src;
        nbatches = src0.size(0);
        temp = fc85.forward(src0);

        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
 
        temp = fc86.forward(src0);
  
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc87(src0);
        temp3 = temp3.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        src0 = temp.div(sqrt_dk);
        src0 = src0.softmax(3);
        src0 = src0.matmul(temp3);
        src0 = src0.transpose(1, 2);
        src0 = src0.contiguous();
        src0 = src0.view( {nbatches, -1, nheads * d_k} )[0];
        
        src = src + src0;
        temp = fc88.forward(src);
        temp = F::relu(temp);
        temp = fc89.forward(temp);
        src = src + temp;


        //encoder-layer-18


        src0 = src;
        nbatches = src0.size(0);
        temp1 = fc90(src0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
   
        temp = fc91.forward(src0);
  
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc92.forward(src0);

        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        src0 = temp.div(sqrt_dk);
        src0 = src0.softmax(3);
        src0 = src0.matmul(temp3);
        src0 = src0.transpose(1, 2);
        src0 = src0.contiguous();
        src0 = src0.view( {nbatches, -1, nheads * d_k} )[0];
        
        src = src + src0;

        temp = fc93.forward(src);
        temp = F::relu(temp);
        temp = fc94(temp);
        src = src + temp;


        //encoder-layer-19


        src0 = src;
        nbatches = src0.size(0);
        temp1 = fc95(src0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
   
        temp = fc96.forward(src0);

        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc97.forward(src0);

        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        src0 = temp.div(sqrt_dk);
        src0 = src0.softmax(3);
        src0 = src0.matmul(temp3);
        src0 = src0.transpose(1, 2);
        src0 = src0.contiguous();
        src0 = src0.view( {nbatches, -1, nheads * d_k} )[0];
        
        src = src + src0;

        temp = fc98.forward(src);
        temp = relu.forward(temp);
        temp = fc99.forward(temp);
        src = src + temp;


        //encoder-layer-20


        src0 = src;
        nbatches = src0.size(0);
        temp = fc100.forward(src0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc101(src0);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc102(src0);
        temp3 = temp3.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        src0 = temp.div(sqrt_dk);
        src0 = src0.softmax(3);
        src0 = src0.matmul(temp3);
        src0 = src0.transpose(1, 2);
        src0 = src0.contiguous();
        src0 = src0.view( {nbatches, -1, nheads * d_k} )[0];
        
        src = src + src0;
        temp = fc103.forward(src);
        temp = relu.forward(temp);
        temp = fc104.forward(temp);
        src = src + temp;


        //encoder-layer-21


        src0 = src;
        nbatches = src0.size(0);
        temp = fc105.forward(src0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc106.forward(src0);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
    
        temp = fc107.forward(src0);
        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        src0 = temp.div(sqrt_dk);
        src0 = src0.softmax(3);
        src0 = src0.matmul(temp3);
        src0 = src0.transpose(1, 2);
        src0 = src0.contiguous();
        src0 = src0.view( {nbatches, -1, nheads * d_k} )[0];
        
        src = src + src0;
        temp = fc108.forward(src);
        temp = relu.forward(temp);
        temp = fc109.forward(temp);
        src = src + temp;


        //encoder-layer-22


        src0 = src;
        nbatches = src0.size(0);
        temp = fc110.forward(src0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc111.forward(src0);

        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc112.forward(src0);

        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        src0 = temp.div(sqrt_dk);
        src0 = src0.softmax(3);
        src0 = src0.matmul(temp3);
        src0 = src0.transpose(1, 2);
        src0 = src0.contiguous();
        src0 = src0.view( {nbatches, -1, nheads * d_k} )[0];
        
        src = src + src0;

        temp = fc113.forward(src);
        temp = F::relu(temp);
        temp = fc114.forward(temp);
        src = src + temp;


        //encoder-layer-23


        src0 = src;
        nbatches = src0.size(0);

        temp = fc115.forward(src0);
  
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc116.forward(src0);

        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc117.forward(src0);

        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        src0 = temp.div(sqrt_dk);
        src0 = src0.softmax(3);
        src0 = src0.matmul(temp3);
        src0 = src0.transpose(1, 2);
        src0 = src0.contiguous();
        src0 = src0.view( {nbatches, -1, nheads * d_k} )[0];
        
        src = src + src0;
        temp = fc118.forward(src);
        temp = relu.forward(temp);
        temp = fc119.forward(temp);
        src = src + temp;



        //decoder-layer-0


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc120.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc121.forward(tgt0);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc122.forward(tgt0);
        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt = tgt + tgt0;
        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc123.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
  
        temp = fc124.forward(src);

        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
   
        temp = fc125.forward(src);

        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt0 = tgt + tgt0;
        
        temp = fc126.forward(tgt);
        temp = relu.forward(temp);
        temp = fc127(temp);
        tgt = tgt + temp;

        //decoder-layer-1


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp1 = fc128(tgt0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc129(tgt0);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc130.forward(tgt0);

        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt = tgt + tgt0;
        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc131.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
     
        temp = fc132.forward(src);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
   
        temp = fc133.forward(src);

        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt0 = tgt + tgt0;
  
        temp = fc134.forward(tgt);
        temp = F::relu(temp);
        temp = fc135.forward(temp);
        tgt = tgt + temp;


        //decoder-layer-2


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc136.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc137.forward(tgt0);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc138.forward(tgt0);
        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt = tgt + tgt0;
        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc139.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc140.forward(src);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc141(src);
        temp3 = temp3.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt0 = tgt + tgt0;
        
        temp = fc142(tgt);
        temp = relu.forward(temp);
        temp = fc143.forward(temp);
        tgt = tgt + temp;


        //decoder-layer-3


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc144.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc145(tgt0);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc146.forward(tgt0);

        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt = tgt + tgt0;
        tgt0 = tgt;
        nbatches = tgt0.size(0);

        temp = fc147.forward(tgt0);

        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc148.forward(src);
   
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc149(src);
        temp3 = temp3.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt0 = tgt + tgt0;
        
        temp = fc150.forward(tgt);
        temp = relu.forward(temp);
        temp = fc151.forward(temp);
        tgt = tgt + temp;


        //decoder-layer-4


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc152.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc153.forward(tgt0);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc154.forward(tgt0);

        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt = tgt + tgt0;
        tgt0 = tgt;
        nbatches = tgt0.size(0);

        temp = fc155.forward(tgt0);

        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
 
        temp = fc156.forward(src);

        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc157.forward(src);

        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt0 = tgt + tgt0;

        temp = fc158.forward(tgt);
        temp = relu.forward(temp);
        temp = fc159.forward(temp);
        tgt = tgt + temp;


        //decoder-layer-5


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc160.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc161.forward(tgt0);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc162(tgt0);
        temp3 = temp3.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt = tgt + tgt0;
        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc163.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc164(src);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc165.forward(src);
        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt0 = tgt + tgt0;
        
        temp = fc166.forward(tgt);
        temp = F::relu(temp);
        temp = fc167.forward(temp);
        tgt = tgt + temp;


        //decoder-layer-6


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc168.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
    
        temp = fc169.forward(tgt0);
    
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc170(tgt0);
        temp3 = temp3.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt = tgt + tgt0;
        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc171.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
      
        temp = fc172.forward(src);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc173(src);
        temp3 = temp3.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt0 = tgt + tgt0;
        
        temp = fc174.forward(tgt);
        temp = relu.forward(temp);
        temp = fc175.forward(temp);
        tgt = tgt + temp;


        //decoder-layer-7


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc176.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc177.forward(tgt0);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc178.forward(tgt0);
        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt = tgt + tgt0;
        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp1 = fc179(tgt0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
  
        temp = fc180.forward(src);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc181.forward(src);
        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt0 = tgt + tgt0;
  
        temp = fc182.forward(tgt);
        temp = relu.forward(temp);
        temp = fc183.forward(temp);
        tgt = tgt + temp;


        //decoder-layer-8


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc184.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
 
        temp = fc185.forward(tgt0);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
   
        temp = fc186.forward(tgt0);
        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt = tgt + tgt0;
        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc187.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
  
        temp = fc188.forward(src);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc189.forward(src);
        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt0 = tgt + tgt0;
        
        temp = fc190.forward(tgt);
        temp = relu.forward(temp);
        temp = fc191.forward(temp);
        tgt = tgt + temp;


        //decoder-layer-9


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc192.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc193.forward(tgt0);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
   
        temp = fc194.forward(tgt0);
        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt = tgt + tgt0;
        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc195.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
   
        temp = fc196.forward(src);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc197.forward(src);
        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt0 = tgt + tgt0;
 
        temp = fc198.forward(tgt);
        temp = relu.forward(temp);
        temp = fc199.forward(temp);
        tgt = tgt + temp;


        //decoder-layer-10


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc200.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc201.forward(tgt0);

        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
   
        temp = fc202.forward(tgt0);
  
        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt = tgt + tgt0;
        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp1 = fc203(tgt0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc204.forward(src);
     
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc205.forward(src);

        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt0 = tgt + tgt0;
        
        temp = fc206(tgt);
        temp = relu.forward(temp);
        temp = fc207.forward(temp);
        tgt = tgt + temp;


        //decoder-layer-11


        tgt0 = tgt;
        nbatches = tgt0.size(0);
 
        temp = fc208.forward(tgt0);
   
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc209(tgt0);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc210.forward(tgt0);

        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt = tgt + tgt0;
        tgt0 = tgt;
        nbatches = tgt0.size(0);
 
        temp = fc211.forward(tgt0);

        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc212.forward(src);

        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc213.forward(src);

        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt0 = tgt + tgt0;
        

        temp = fc214.forward(tgt);
        temp = F::relu(temp);
        temp = fc215.forward(temp);
        tgt = tgt + temp;


        //decoder-layer-12


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc216.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
   
        temp = fc217.forward(tgt0);

        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc218.forward(tgt0);

        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt = tgt + tgt0;
        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp1 = fc219(tgt0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc220.forward(src);

        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc221.forward(src);

        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt0 = tgt + tgt0;
        

        temp = fc222.forward(tgt);
        temp = relu.forward(temp);
        temp = fc223.forward(temp);
        tgt = tgt + temp;


        //decoder-layer-13


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc224.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
 
        temp = fc225.forward(tgt0);
 
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc226(tgt0);
        temp3 = temp3.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt = tgt + tgt0;
        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc227.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc228.forward(src);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc229.forward(src);
        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt0 = tgt + tgt0;
        
        temp = fc230.forward(tgt);
        temp = relu.forward(temp);
        temp = fc231.forward(temp);
        tgt = tgt + temp;


        //decoder-layer-14


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc232.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
      
        temp = fc233.forward(tgt0);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc234.forward(tgt0);
        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt = tgt + tgt0;
        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc235.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc236.forward(src);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
    
        temp = fc237.forward(src);
        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt0 = tgt + tgt0;
        
        temp = fc238.forward(tgt);
        temp = relu.forward(temp);
        temp = fc239.forward(temp);
        tgt = tgt + temp;


        //decoder-layer-15


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc240.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
    
        temp = fc241.forward(tgt0);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
    
        temp = fc242.forward(tgt0);
        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt = tgt + tgt0;
        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc243.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
 
        temp = fc244.forward(src);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
  
        temp = fc245.forward(src);
        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt0 = tgt + tgt0;
        
        temp = fc246.forward(tgt);
        temp = relu.forward(temp);
        temp = fc247.forward(temp);
        tgt = tgt + temp;


        //decoder-layer-16


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc248.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
 
        temp = fc249.forward(tgt0);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc250(tgt0);
        temp3 = temp3.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt = tgt + tgt0;
        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc251.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
 
        temp = fc252.forward(src);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc253.forward(src);
        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt0 = tgt + tgt0;
        
        temp = fc254.forward(tgt);
        temp = relu.forward(temp);
        temp = fc255.forward(temp);
        tgt = tgt + temp;


        //decoder-layer-17


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc256.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
   
        temp = fc257.forward(tgt0);

        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
    
        temp = fc258.forward(tgt0);
        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt = tgt + tgt0;
        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp1 = fc259(tgt0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);

        temp = fc260.forward(src);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
    
        temp = fc261.forward(src);

        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt0 = tgt + tgt0;
        
        temp = fc262(tgt);
        temp = F::relu(temp);
        temp = fc263.forward(temp);
        tgt = tgt + temp;


        //decoder-layer-18


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc264.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        
        temp = fc265.forward(tgt0);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc266(tgt0);
        temp3 = temp3.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt = tgt + tgt0;
        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc267.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
       
        temp = fc268.forward(src);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
      
        temp = fc269.forward(src);
        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt0 = tgt + tgt0;
        
        temp = fc270.forward(tgt);
        temp = relu.forward(temp);
        temp = fc271.forward(temp);
        tgt = tgt + temp;


        //decoder-layer-19


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc272.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
 
        temp = fc273.forward(tgt0);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
    
        temp = fc274.forward(tgt0);
        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt = tgt + tgt0;
        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc275.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
       
        temp = fc276.forward(src);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
      
        temp = fc277.forward(src);
        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt0 = tgt + tgt0;
        
        temp = fc278.forward(tgt);
        temp = relu.forward(temp);
        temp = fc279(temp);
        tgt = tgt + temp;


        //decoder-layer-20


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc280.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
     
        temp = fc281.forward(tgt0);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
     
        temp = fc282.forward(tgt0);

        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt = tgt + tgt0;
        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc283.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
    
        temp = fc284.forward(src);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
      
        temp = fc285.forward(src);
        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt0 = tgt + tgt0;
        
        temp = fc286(tgt);
        temp = relu.forward(temp);
        temp = fc287.forward(temp);
        tgt = tgt + temp;


        //decoder-layer-21


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc288.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
       
        temp = fc289.forward(tgt0);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc290(tgt0);
        temp3 = temp3.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt = tgt + tgt0;
        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc291.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
 
        temp = fc292.forward(src);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
      
        temp = fc293.forward(src);
        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt0 = tgt + tgt0;
        
        temp = fc294.forward(tgt);
        temp = F::relu(temp);
        temp = fc295.forward(temp);
        tgt = tgt + temp;

        //decoder-layer-22


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc296.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
      
        temp = fc297.forward(tgt0);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
     
        temp = fc298.forward(tgt0);
        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt = tgt + tgt0;
        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc299.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
  
        temp = fc300.forward(src);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
      
        temp = fc301.forward(src);
        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt0 = tgt + tgt0;
        
        temp = fc302.forward(tgt);
        temp = relu.forward(temp);
        temp = fc303(temp);
        tgt = tgt + temp;


        //decoder-layer-23


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc304.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
       
        temp = fc305.forward(tgt0);
    
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
  
        temp = fc306.forward(tgt0);
        temp3 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt = tgt + tgt0;
        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = fc307.forward(tgt0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
    
        temp = fc308.forward(src);
        temp2 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc309(src);
        temp3 = temp3.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = temp2.transpose(2, 3);
        temp = temp1.matmul(temp);
        tgt0 = temp.div(sqrt_dk);
        tgt0 = tgt0.softmax(3);
        tgt0 = tgt0.matmul(temp3);
        tgt0 = tgt0.transpose(1, 2);
        tgt0 = tgt0.contiguous();
        tgt0 = tgt0.view( {nbatches, -1, nheads * d_k} )[0];
        tgt0 = tgt + tgt0;
        
        temp = fc310.forward(tgt);
        temp = F::relu(temp);
        temp = fc311(temp);
        tgt = tgt + temp;

        return tgt;
    }

    void morphpara(){
        // 
    }
};

trans_part* models[] = {
    new trans_warmup(),
    new trans_gpu_part1(),
    new trans_gpu_part2()
}; 

// Logic and data behind the server's behavior.
class GreeterServiceImpl final : public Greeter::Service {
  std::string prefix;
  
  Status SayHello(ServerContext* context, const HelloRequest* request,
                  HelloReply* reply) override {
    int tag = request->tag();
    if (tag == 0) {
        std::cout << "[Preprocessing phase] Receive TEE signal, GPU is now warming up ..."<< std::endl;
        for (size_t i = 0; i < count; i++) {
            out = models[tag]->forward(src, tgt);
        }
        std::cout << "[Inference phase] GPU is serving ..." << std::endl;
    } else {
        torch::Tensor inter_active;
        std::istringstream ss(request->name());
        torch::load(inter_active, ss);
        out = models[tag]->forward(inter_active.to(at::kCUDA), tgt); 
        std::stringstream so;
        torch::save(out.to(at::kCPU), so);
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
