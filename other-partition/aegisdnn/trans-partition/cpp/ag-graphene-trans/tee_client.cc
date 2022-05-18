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
int count = 10; // number of inference tests
int d_model = 1024; //token size
int nheads = 8;
int d_k = d_model / nheads;
int d_v = d_k;
int sqrt_dk = (int)floor(sqrt((double)d_v));
int d_ff = 2048;
torch::Tensor src = torch::rand( {32, 10, d_model} );
torch::Tensor tgt = torch::rand( {32, 20, d_model} );
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

struct trans : public torch::nn::Module
{
    int record_flag = 0;
    // cornerstone fingerprint
    torch::Tensor cfp11;
    torch::Tensor cfp12;
    torch::Tensor cfp21;
    torch::Tensor cfp22;
    // fingerprint
    torch::Tensor fp1_1;
    torch::Tensor fp1_2;
    torch::Tensor fp1_3;
    torch::Tensor fp2_1;
    torch::Tensor fp2_2;
    torch::Tensor fp2_3;
    
    // fingerprint proof
    torch::Tensor pf1_1;
    torch::Tensor pf1_2;
    torch::Tensor pf1_3;
    torch::Tensor pf2_1;
    torch::Tensor pf2_2;
    torch::Tensor pf2_3;

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
    
    std::vector<std::function<torch::Tensor(torch::Tensor)>> forwards;
        
    trans():
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
            forwards.push_back(std::bind(&trans::forward1_new, this, std::placeholders::_1));
            forwards.push_back(std::bind(&trans::forward2_new, this, std::placeholders::_1));
        }

    torch::Tensor forward1_new(torch::Tensor src) {
        // std::cout<<"forward1_new"<<std::endl;
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
        
        // temp = fc3.forward(src);
        // temp = relu.forward(temp);
        // temp = fc4(temp);
        // src = src + temp;
        // std::cout<<"forward 1 size = "<<src.sizes()<<std::endl;
        return src;
    }

    torch::Tensor forward2_new(torch::Tensor src) {
        // std::cout<<"forward2_new"<<std::endl;
        int nbatches;
        torch::Tensor temp_re;
        torch::Tensor temp;
        torch::Tensor temp0;
        torch::Tensor temp1;
        torch::Tensor temp2;
        torch::Tensor temp3;
        torch::Tensor src0;
        torch::Tensor tgt0;

        //encoder-layer-2
        src0 = src;
        nbatches = src0.size(0);
        temp = fc10.forward(src0);
        temp1 = temp.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp_re = fc10.forward(src0);
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
        temp_re = fc13.forward(src);
        temp = relu.forward(temp);
        temp = fc14(temp);
        src = src + temp;
        return src;
    }
   

    torch::Tensor forward(torch::Tensor x) {
        torch::Tensor intermedia;
        torch::Tensor tmp;
        std::cout<<"[Inference phase] Inference & integrity check ("<< (record_flag+1) << "/"<<count<<")" <<std::endl;
        for (int i = 0; i < 2;i++) {
            intermedia = forwards[i](x);
            std::stringstream ss;
            torch::save(intermedia, ss);
            auto reply = request(ss.str().data(), ss.str().size(), i + 1);
            std::istringstream is(reply);
            torch::load(tmp, is);             
            x = tmp/scalar; 
        }
        
        record_flag ++;
        return x;
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

    ch_args.SetMaxReceiveMessageSize(28 * 1024 * 1024); 
    greeter = new GreeterClient(grpc::CreateCustomChannel(target_str, grpc::InsecureChannelCredentials(), ch_args));

    trans model;
    model.eval();

    // GPU warm up
    std::cout << "[Preprocessing phase (1/3)] Model partitioned & Parameter morphed!" <<std::endl;
    reply = greeter->SayHello(msg, 0);

    // tee warm up
    for (size_t i = 0; i < 10; i++){
        out = model.forward(src);
    }

    struct timeval tvs, tve;
    gettimeofday(&tvs, 0);
    for (size_t i = 0; i < count; i++){
        out = model.forward(src);
    }
    gettimeofday(&tve, 0);
    float ms_time = (tve.tv_sec - tvs.tv_sec) * 1000 + (tve.tv_usec - tvs.tv_usec) / 1000;
    std::cout << "For " << count << " inferences ..." << std::endl;
    std::cout << "Time elapsed: " << ms_time << " ms." << std::endl;
    std::cout << "Fetch here. Time consuming: " << ms_time/count << " ms per inference." << std::endl;
    std::cout << "Completed successfully !!!" << std::endl;

    return 0;
}
