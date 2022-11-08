#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <torch/torch.h>
#include <iostream>
#include <random>
#include <time.h>
#include <sys/time.h>
#include <string>
#include <math.h>

namespace F = torch::nn::functional;

int d_model;
int d_v;
int d_k;
int sqrt_dk;
int nheads;
int d_ff;

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

struct transformer : public torch::nn::Module
{
    operator1 relu;
    torch::nn::Linear fc0;
    operator7 fc1;
    operator7 fc2;
    operator7 fc3;
    operator7 fc4;
    torch::nn::Linear fc5;
    operator7 fc6;
    operator7 fc7;
    operator7 fc8;
    torch::nn::Linear fc9;
    operator7 fc10;
    operator7 fc11;
    operator7 fc12;
    torch::nn::Linear fc13;
    operator7 fc14;
    torch::nn::Linear fc15;
    torch::nn::Linear fc16;
    torch::nn::Linear fc17;
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
    torch::nn::Linear fc29;
    torch::nn::Linear fc30;
    operator7 fc31;
    operator7 fc32;
    torch::nn::Linear fc33;
    operator7 fc34;
    torch::nn::Linear fc35;
    torch::nn::Linear fc36;
    torch::nn::Linear fc37;
    operator7 fc38;
    torch::nn::Linear fc39;
    torch::nn::Linear fc40;
    torch::nn::Linear fc41;
    operator7 fc42;
    torch::nn::Linear fc43;
    operator7 fc44;
    operator7 fc45;
    torch::nn::Linear fc46;
    torch::nn::Linear fc47;
    torch::nn::Linear fc48;
    torch::nn::Linear fc49;
    operator7 fc50;
    torch::nn::Linear fc51;
    torch::nn::Linear fc52;
    operator7 fc53;
    torch::nn::Linear fc54;
    torch::nn::Linear fc55;
    torch::nn::Linear fc56;
    operator7 fc57;
    torch::nn::Linear fc58;
    torch::nn::Linear fc59;
    operator7 fc60;
    operator7 fc61;
    torch::nn::Linear fc62;
    operator7 fc63;
    operator7 fc64;
    operator7 fc65;
    operator7 fc66;
    torch::nn::Linear fc67;
    operator7 fc68;
    operator7 fc69;
    operator7 fc70;
    operator7 fc71;
    torch::nn::Linear fc72;
    operator7 fc73;
    operator7 fc74;
    torch::nn::Linear fc75;
    torch::nn::Linear fc76;
    operator7 fc77;
    operator7 fc78;
    torch::nn::Linear fc79;
    operator7 fc80;
    operator7 fc81;
    operator7 fc82;
    operator7 fc83;
    torch::nn::Linear fc84;
    torch::nn::Linear fc85;
    torch::nn::Linear fc86;
    torch::nn::Linear fc87;
    torch::nn::Linear fc88;
    operator7 fc89;
    torch::nn::Linear fc90;
    torch::nn::Linear fc91;
    operator7 fc92;
    torch::nn::Linear fc93;
    operator7 fc94;
    operator7 fc95;
    operator7 fc96;
    operator7 fc97;
    operator7 fc98;
    torch::nn::Linear fc99;
    torch::nn::Linear fc100;
    torch::nn::Linear fc101;
    operator7 fc102;
    torch::nn::Linear fc103;
    torch::nn::Linear fc104;
    operator7 fc105;
    operator7 fc106;
    torch::nn::Linear fc107;
    torch::nn::Linear fc108;
    torch::nn::Linear fc109;
    torch::nn::Linear fc110;
    operator7 fc111;
    operator7 fc112;
    operator7 fc113;
    operator7 fc114;
    operator7 fc115;
    torch::nn::Linear fc116;
    operator7 fc117;
    torch::nn::Linear fc118;
    operator7 fc119;
    torch::nn::Linear fc120;
    operator7 fc121;
    operator7 fc122;
    operator7 fc123;
    torch::nn::Linear fc124;
    operator7 fc125;
    operator7 fc126;
    operator7 fc127;
    operator7 fc128;
    operator7 fc129;
    torch::nn::Linear fc130;
    operator7 fc131;
    operator7 fc132;
    operator7 fc133;
    operator7 fc134;
    operator7 fc135;
    torch::nn::Linear fc136;
    operator7 fc137;
    operator7 fc138;
    torch::nn::Linear fc139;
    torch::nn::Linear fc140;
    torch::nn::Linear fc141;
    operator7 fc142;
    operator7 fc143;
    operator7 fc144;
    operator7 fc145;
    torch::nn::Linear fc146;
    torch::nn::Linear fc147;
    operator7 fc148;
    operator7 fc149;
    operator7 fc150;
    operator7 fc151;
    torch::nn::Linear fc152;
    torch::nn::Linear fc153;
    torch::nn::Linear fc154;
    operator7 fc155;
    operator7 fc156;
    torch::nn::Linear fc157;
    torch::nn::Linear fc158;
    operator7 fc159;
    torch::nn::Linear fc160;
    torch::nn::Linear fc161;
    torch::nn::Linear fc162;
    torch::nn::Linear fc163;
    torch::nn::Linear fc164;
    operator7 fc165;
    operator7 fc166;
    operator7 fc167;
    operator7 fc168;
    operator7 fc169;
    torch::nn::Linear fc170;
    operator7 fc171;
    operator7 fc172;
    operator7 fc173;
    torch::nn::Linear fc174;
    operator7 fc175;
    operator7 fc176;
    operator7 fc177;
    torch::nn::Linear fc178;
    torch::nn::Linear fc179;
    operator7 fc180;
    operator7 fc181;
    operator7 fc182;
    torch::nn::Linear fc183;
    torch::nn::Linear fc184;
    operator7 fc185;
    torch::nn::Linear fc186;
    operator7 fc187;
    torch::nn::Linear fc188;
    operator7 fc189;
    operator7 fc190;
    torch::nn::Linear fc191;
    operator7 fc192;
    operator7 fc193;
    operator7 fc194;
    operator7 fc195;
    operator7 fc196;
    torch::nn::Linear fc197;
    torch::nn::Linear fc198;
    operator7 fc199;
    operator7 fc200;
    operator7 fc201;
    torch::nn::Linear fc202;
    operator7 fc203;
    operator7 fc204;
    operator7 fc205;
    torch::nn::Linear fc206;
    operator7 fc207;
    operator7 fc208;
    torch::nn::Linear fc209;
    torch::nn::Linear fc210;
    operator7 fc211;
    torch::nn::Linear fc212;
    operator7 fc213;
    operator7 fc214;
    torch::nn::Linear fc215;
    operator7 fc216;
    operator7 fc217;
    operator7 fc218;
    torch::nn::Linear fc219;
    operator7 fc220;
    operator7 fc221;
    operator7 fc222;
    torch::nn::Linear fc223;
    operator7 fc224;
    operator7 fc225;
    torch::nn::Linear fc226;
    operator7 fc227;
    operator7 fc228;
    torch::nn::Linear fc229;
    operator7 fc230;
    torch::nn::Linear fc231;
    operator7 fc232;
    torch::nn::Linear fc233;
    operator7 fc234;
    torch::nn::Linear fc235;
    operator7 fc236;
    operator7 fc237;
    operator7 fc238;
    torch::nn::Linear fc239;
    operator7 fc240;
    operator7 fc241;
    operator7 fc242;
    torch::nn::Linear fc243;
    operator7 fc244;
    operator7 fc245;
    operator7 fc246;
    torch::nn::Linear fc247;
    operator7 fc248;
    operator7 fc249;
    torch::nn::Linear fc250;
    operator7 fc251;
    operator7 fc252;
    torch::nn::Linear fc253;
    torch::nn::Linear fc254;
    operator7 fc255;
    operator7 fc256;
    operator7 fc257;
    torch::nn::Linear fc258;
    operator7 fc259;
    operator7 fc260;
    torch::nn::Linear fc261;
    torch::nn::Linear fc262;
    torch::nn::Linear fc263;
    torch::nn::Linear fc264;
    operator7 fc265;
    torch::nn::Linear fc266;
    operator7 fc267;
    torch::nn::Linear fc268;
    torch::nn::Linear fc269;
    operator7 fc270;
    torch::nn::Linear fc271;
    torch::nn::Linear fc272;
    torch::nn::Linear fc273;
    torch::nn::Linear fc274;
    torch::nn::Linear fc275;
    operator7 fc276;
    operator7 fc277;
    operator7 fc278;
    operator7 fc279;
    operator7 fc280;
    operator7 fc281;
    torch::nn::Linear fc282;
    operator7 fc283;
    torch::nn::Linear fc284;
    operator7 fc285;
    torch::nn::Linear fc286;
    torch::nn::Linear fc287;
    operator7 fc288;
    torch::nn::Linear fc289;
    operator7 fc290;
    torch::nn::Linear fc291;
    operator7 fc292;
    operator7 fc293;
    torch::nn::Linear fc294;
    torch::nn::Linear fc295;
    operator7 fc296;
    operator7 fc297;
    operator7 fc298;
    torch::nn::Linear fc299;
    operator7 fc300;
    torch::nn::Linear fc301;
    torch::nn::Linear fc302;
    operator7 fc303;
    torch::nn::Linear fc304;
    operator7 fc305;
    operator7 fc306;
    torch::nn::Linear fc307;
    torch::nn::Linear fc308;
    torch::nn::Linear fc309;
    operator7 fc310;
    operator7 fc311;
    transformer():
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
        register_module("fc0", fc0);
        fc1.to(at::kCUDA);
        fc2.to(at::kCUDA);
        fc3.to(at::kCUDA);
        fc4.to(at::kCUDA);
        register_module("fc5", fc5);
        fc6.to(at::kCUDA);
        fc7.to(at::kCUDA);
        fc8.to(at::kCUDA);
        register_module("fc9", fc9);
        fc10.to(at::kCUDA);
        fc11.to(at::kCUDA);
        fc12.to(at::kCUDA);
        register_module("fc13", fc13);
        fc14.to(at::kCUDA);
        register_module("fc15", fc15);
        register_module("fc16", fc16);
        register_module("fc17", fc17);
        register_module("fc18", fc18);
        fc19.to(at::kCUDA);
        fc20.to(at::kCUDA);
        fc21.to(at::kCUDA);
        fc22.to(at::kCUDA);
        fc23.to(at::kCUDA);
        fc24.to(at::kCUDA);
        register_module("fc25", fc25);
        fc26.to(at::kCUDA);
        fc27.to(at::kCUDA);
        fc28.to(at::kCUDA);
        register_module("fc29", fc29);
        register_module("fc30", fc30);
        fc31.to(at::kCUDA);
        fc32.to(at::kCUDA);
        register_module("fc33", fc33);
        fc34.to(at::kCUDA);
        register_module("fc35", fc35);
        register_module("fc36", fc36);
        register_module("fc37", fc37);
        fc38.to(at::kCUDA);
        register_module("fc39", fc39);
        register_module("fc40", fc40);
        register_module("fc41", fc41);
        fc42.to(at::kCUDA);
        register_module("fc43", fc43);
        fc44.to(at::kCUDA);
        fc45.to(at::kCUDA);
        register_module("fc46", fc46);
        register_module("fc47", fc47);
        register_module("fc48", fc48);
        register_module("fc49", fc49);
        fc50.to(at::kCUDA);
        register_module("fc51", fc51);
        register_module("fc52", fc52);
        fc53.to(at::kCUDA);
        register_module("fc54", fc54);
        register_module("fc55", fc55);
        register_module("fc56", fc56);
        fc57.to(at::kCUDA);
        register_module("fc58", fc58);
        register_module("fc59", fc59);
        fc60.to(at::kCUDA);
        fc61.to(at::kCUDA);
        register_module("fc62", fc62);
        fc63.to(at::kCUDA);
        fc64.to(at::kCUDA);
        fc65.to(at::kCUDA);
        fc66.to(at::kCUDA);
        register_module("fc67", fc67);
        fc68.to(at::kCUDA);
        fc69.to(at::kCUDA);
        fc70.to(at::kCUDA);
        fc71.to(at::kCUDA);
        register_module("fc72", fc72);
        fc73.to(at::kCUDA);
        fc74.to(at::kCUDA);
        register_module("fc75", fc75);
        register_module("fc76", fc76);
        fc77.to(at::kCUDA);
        fc78.to(at::kCUDA);
        register_module("fc79", fc79);
        fc80.to(at::kCUDA);
        fc81.to(at::kCUDA);
        fc82.to(at::kCUDA);
        fc83.to(at::kCUDA);
        register_module("fc84", fc84);
        register_module("fc85", fc85);
        register_module("fc86", fc86);
        register_module("fc87", fc87);
        register_module("fc88", fc88);
        fc89.to(at::kCUDA);
        register_module("fc90", fc90);
        register_module("fc91", fc91);
        fc92.to(at::kCUDA);
        register_module("fc93", fc93);
        fc94.to(at::kCUDA);
        fc95.to(at::kCUDA);
        fc96.to(at::kCUDA);
        fc97.to(at::kCUDA);
        fc98.to(at::kCUDA);
        register_module("fc99", fc99);
        register_module("fc100", fc100);
        register_module("fc101", fc101);
        fc102.to(at::kCUDA);
        register_module("fc103", fc103);
        register_module("fc104", fc104);
        fc105.to(at::kCUDA);
        fc106.to(at::kCUDA);
        register_module("fc107", fc107);
        register_module("fc108", fc108);
        register_module("fc109", fc109);
        register_module("fc110", fc110);
        fc111.to(at::kCUDA);
        fc112.to(at::kCUDA);
        fc113.to(at::kCUDA);
        fc114.to(at::kCUDA);
        fc115.to(at::kCUDA);
        register_module("fc116", fc116);
        fc117.to(at::kCUDA);
        register_module("fc118", fc118);
        fc119.to(at::kCUDA);
        register_module("fc120", fc120);
        fc121.to(at::kCUDA);
        fc122.to(at::kCUDA);
        fc123.to(at::kCUDA);
        register_module("fc124", fc124);
        fc125.to(at::kCUDA);
        fc126.to(at::kCUDA);
        fc127.to(at::kCUDA);
        fc128.to(at::kCUDA);
        fc129.to(at::kCUDA);
        register_module("fc130", fc130);
        fc131.to(at::kCUDA);
        fc132.to(at::kCUDA);
        fc133.to(at::kCUDA);
        fc134.to(at::kCUDA);
        fc135.to(at::kCUDA);
        register_module("fc136", fc136);
        fc137.to(at::kCUDA);
        fc138.to(at::kCUDA);
        register_module("fc139", fc139);
        register_module("fc140", fc140);
        register_module("fc141", fc141);
        fc142.to(at::kCUDA);
        fc143.to(at::kCUDA);
        fc144.to(at::kCUDA);
        fc145.to(at::kCUDA);
        register_module("fc146", fc146);
        register_module("fc147", fc147);
        fc148.to(at::kCUDA);
        fc149.to(at::kCUDA);
        fc150.to(at::kCUDA);
        fc151.to(at::kCUDA);
        register_module("fc152", fc152);
        register_module("fc153", fc153);
        register_module("fc154", fc154);
        fc155.to(at::kCUDA);
        fc156.to(at::kCUDA);
        register_module("fc157", fc157);
        register_module("fc158", fc158);
        fc159.to(at::kCUDA);
        register_module("fc160", fc160);
        register_module("fc161", fc161);
        register_module("fc162", fc162);
        register_module("fc163", fc163);
        register_module("fc164", fc164);
        fc165.to(at::kCUDA);
        fc166.to(at::kCUDA);
        fc167.to(at::kCUDA);
        fc168.to(at::kCUDA);
        fc169.to(at::kCUDA);
        register_module("fc170", fc170);
        fc171.to(at::kCUDA);
        fc172.to(at::kCUDA);
        fc173.to(at::kCUDA);
        register_module("fc174", fc174);
        fc175.to(at::kCUDA);
        fc176.to(at::kCUDA);
        fc177.to(at::kCUDA);
        register_module("fc178", fc178);
        register_module("fc179", fc179);
        fc180.to(at::kCUDA);
        fc181.to(at::kCUDA);
        fc182.to(at::kCUDA);
        register_module("fc183", fc183);
        register_module("fc184", fc184);
        fc185.to(at::kCUDA);
        register_module("fc186", fc186);
        fc187.to(at::kCUDA);
        register_module("fc188", fc188);
        fc189.to(at::kCUDA);
        fc190.to(at::kCUDA);
        register_module("fc191", fc191);
        fc192.to(at::kCUDA);
        fc193.to(at::kCUDA);
        fc194.to(at::kCUDA);
        fc195.to(at::kCUDA);
        fc196.to(at::kCUDA);
        register_module("fc197", fc197);
        register_module("fc198", fc198);
        fc199.to(at::kCUDA);
        fc200.to(at::kCUDA);
        fc201.to(at::kCUDA);
        register_module("fc202", fc202);
        fc203.to(at::kCUDA);
        fc204.to(at::kCUDA);
        fc205.to(at::kCUDA);
        register_module("fc206", fc206);
        fc207.to(at::kCUDA);
        fc208.to(at::kCUDA);
        register_module("fc209", fc209);
        register_module("fc210", fc210);
        fc211.to(at::kCUDA);
        register_module("fc212", fc212);
        fc213.to(at::kCUDA);
        fc214.to(at::kCUDA);
        register_module("fc215", fc215);
        fc216.to(at::kCUDA);
        fc217.to(at::kCUDA);
        fc218.to(at::kCUDA);
        register_module("fc219", fc219);
        fc220.to(at::kCUDA);
        fc221.to(at::kCUDA);
        fc222.to(at::kCUDA);
        register_module("fc223", fc223);
        fc224.to(at::kCUDA);
        fc225.to(at::kCUDA);
        register_module("fc226", fc226);
        fc227.to(at::kCUDA);
        fc228.to(at::kCUDA);
        register_module("fc229", fc229);
        fc230.to(at::kCUDA);
        register_module("fc231", fc231);
        fc232.to(at::kCUDA);
        register_module("fc233", fc233);
        fc234.to(at::kCUDA);
        register_module("fc235", fc235);
        fc236.to(at::kCUDA);
        fc237.to(at::kCUDA);
        fc238.to(at::kCUDA);
        register_module("fc239", fc239);
        fc240.to(at::kCUDA);
        fc241.to(at::kCUDA);
        fc242.to(at::kCUDA);
        register_module("fc243", fc243);
        fc244.to(at::kCUDA);
        fc245.to(at::kCUDA);
        fc246.to(at::kCUDA);
        register_module("fc247", fc247);
        fc248.to(at::kCUDA);
        fc249.to(at::kCUDA);
        register_module("fc250", fc250);
        fc251.to(at::kCUDA);
        fc252.to(at::kCUDA);
        register_module("fc253", fc253);
        register_module("fc254", fc254);
        fc255.to(at::kCUDA);
        fc256.to(at::kCUDA);
        fc257.to(at::kCUDA);
        register_module("fc258", fc258);
        fc259.to(at::kCUDA);
        fc260.to(at::kCUDA);
        register_module("fc261", fc261);
        register_module("fc262", fc262);
        register_module("fc263", fc263);
        register_module("fc264", fc264);
        fc265.to(at::kCUDA);
        register_module("fc266", fc266);
        fc267.to(at::kCUDA);
        register_module("fc268", fc268);
        register_module("fc269", fc269);
        fc270.to(at::kCUDA);
        register_module("fc271", fc271);
        register_module("fc272", fc272);
        register_module("fc273", fc273);
        register_module("fc274", fc274);
        register_module("fc275", fc275);
        fc276.to(at::kCUDA);
        fc277.to(at::kCUDA);
        fc278.to(at::kCUDA);
        fc279.to(at::kCUDA);
        fc280.to(at::kCUDA);
        fc281.to(at::kCUDA);
        register_module("fc282", fc282);
        fc283.to(at::kCUDA);
        register_module("fc284", fc284);
        fc285.to(at::kCUDA);
        register_module("fc286", fc286);
        register_module("fc287", fc287);
        fc288.to(at::kCUDA);
        register_module("fc289", fc289);
        fc290.to(at::kCUDA);
        register_module("fc291", fc291);
        fc292.to(at::kCUDA);
        fc293.to(at::kCUDA);
        register_module("fc294", fc294);
        register_module("fc295", fc295);
        fc296.to(at::kCUDA);
        fc297.to(at::kCUDA);
        fc298.to(at::kCUDA);
        register_module("fc299", fc299);
        fc300.to(at::kCUDA);
        register_module("fc301", fc301);
        register_module("fc302", fc302);
        fc303.to(at::kCUDA);
        register_module("fc304", fc304);
        fc305.to(at::kCUDA);
        fc306.to(at::kCUDA);
        register_module("fc307", fc307);
        register_module("fc308", fc308);
        register_module("fc309", fc309);
        fc310.to(at::kCUDA);
        fc311.to(at::kCUDA);
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
        temp1 = fc0(src0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src0.to(at::kCUDA);
        temp = fc1.forward(temp);
        temp2 = temp.to(at::kCPU);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src0.to(at::kCUDA);
        temp = fc2.forward(temp);
        temp3 = temp.to(at::kCPU);
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
        temp = src.to(at::kCUDA);
        temp = fc3.forward(temp);
        temp = temp.to(at::kCPU);
        temp = F::relu(temp);
        temp = temp.to(at::kCUDA);
        temp = fc4.forward(temp);
        temp = temp.to(at::kCPU);
        src = src + temp;


        //encoder-layer-1


        src0 = src;
        nbatches = src0.size(0);
        temp1 = fc5(src0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src0.to(at::kCUDA);
        temp = fc6.forward(temp);
        temp2 = temp.to(at::kCPU);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src0.to(at::kCUDA);
        temp = fc7.forward(temp);
        temp3 = temp.to(at::kCPU);
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
        temp = src.to(at::kCUDA);
        temp = fc8.forward(temp);
        temp = relu.forward(temp);
        temp = temp.to(at::kCPU);
        temp = fc9(temp);
        src = src + temp;


        //encoder-layer-2


        src0 = src;
        nbatches = src0.size(0);
        temp = src0.to(at::kCUDA);
        temp = fc10.forward(temp);
        temp1 = temp.to(at::kCPU);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src0.to(at::kCUDA);
        temp = fc11.forward(temp);
        temp2 = temp.to(at::kCPU);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src0.to(at::kCUDA);
        temp = fc12.forward(temp);
        temp3 = temp.to(at::kCPU);
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
        temp = fc13(src);
        temp = F::relu(temp);
        temp = temp.to(at::kCUDA);
        temp = fc14.forward(temp);
        temp = temp.to(at::kCPU);
        src = src + temp;


        //encoder-layer-3


        src0 = src;
        nbatches = src0.size(0);
        temp1 = fc15(src0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc16(src0);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc17(src0);
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
        temp = fc18(src);
        temp = temp.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = fc19.forward(temp);
        temp = temp.to(at::kCPU);
        src = src + temp;


        //encoder-layer-4


        src0 = src;
        nbatches = src0.size(0);
        temp = src0.to(at::kCUDA);
        temp = fc20.forward(temp);
        temp1 = temp.to(at::kCPU);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src0.to(at::kCUDA);
        temp = fc21.forward(temp);
        temp2 = temp.to(at::kCPU);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src0.to(at::kCUDA);
        temp = fc22.forward(temp);
        temp3 = temp.to(at::kCPU);
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
        temp = src.to(at::kCUDA);
        temp = fc23.forward(temp);
        temp = relu.forward(temp);
        temp = fc24.forward(temp);
        temp = temp.to(at::kCPU);
        src = src + temp;


        //encoder-layer-5


        src0 = src;
        nbatches = src0.size(0);
        temp1 = fc25(src0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src0.to(at::kCUDA);
        temp = fc26.forward(temp);
        temp2 = temp.to(at::kCPU);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src0.to(at::kCUDA);
        temp = fc27.forward(temp);
        temp3 = temp.to(at::kCPU);
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
        temp = src.to(at::kCUDA);
        temp = fc28.forward(temp);
        temp = temp.to(at::kCPU);
        temp = F::relu(temp);
        temp = fc29(temp);
        src = src + temp;


        //encoder-layer-6


        src0 = src;
        nbatches = src0.size(0);
        temp1 = fc30(src0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src0.to(at::kCUDA);
        temp = fc31.forward(temp);
        temp2 = temp.to(at::kCPU);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src0.to(at::kCUDA);
        temp = fc32.forward(temp);
        temp3 = temp.to(at::kCPU);
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
        temp = temp.to(at::kCUDA);
        temp = fc34.forward(temp);
        temp = temp.to(at::kCPU);
        src = src + temp;


        //encoder-layer-7


        src0 = src;
        nbatches = src0.size(0);
        temp1 = fc35(src0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc36(src0);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
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
        temp = src.to(at::kCUDA);
        temp = fc38.forward(temp);
        temp = relu.forward(temp);
        temp = temp.to(at::kCPU);
        temp = fc39(temp);
        src = src + temp;


        //encoder-layer-8


        src0 = src;
        nbatches = src0.size(0);
        temp1 = fc40(src0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc41(src0);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src0.to(at::kCUDA);
        temp = fc42.forward(temp);
        temp3 = temp.to(at::kCPU);
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
        temp = fc43(src);
        temp = temp.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = fc44.forward(temp);
        temp = temp.to(at::kCPU);
        src = src + temp;


        //encoder-layer-9


        src0 = src;
        nbatches = src0.size(0);
        temp = src0.to(at::kCUDA);
        temp = fc45.forward(temp);
        temp1 = temp.to(at::kCPU);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc46(src0);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc47(src0);
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
        temp = fc48(src);
        temp = temp.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = temp.to(at::kCPU);
        temp = fc49(temp);
        src = src + temp;


        //encoder-layer-10


        src0 = src;
        nbatches = src0.size(0);
        temp = src0.to(at::kCUDA);
        temp = fc50.forward(temp);
        temp1 = temp.to(at::kCPU);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc51(src0);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc52(src0);
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
        temp = src.to(at::kCUDA);
        temp = fc53.forward(temp);
        temp = relu.forward(temp);
        temp = temp.to(at::kCPU);
        temp = fc54(temp);
        src = src + temp;


        //encoder-layer-11


        src0 = src;
        nbatches = src0.size(0);
        temp1 = fc55(src0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc56(src0);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src0.to(at::kCUDA);
        temp = fc57.forward(temp);
        temp3 = temp.to(at::kCPU);
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
        temp = fc58(src);
        temp = F::relu(temp);
        temp = fc59(temp);
        src = src + temp;


        //encoder-layer-12


        src0 = src;
        nbatches = src0.size(0);
        temp = src0.to(at::kCUDA);
        temp = fc60.forward(temp);
        temp1 = temp.to(at::kCPU);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src0.to(at::kCUDA);
        temp = fc61.forward(temp);
        temp2 = temp.to(at::kCPU);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc62(src0);
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
        temp = src.to(at::kCUDA);
        temp = fc63.forward(temp);
        temp = temp.to(at::kCPU);
        temp = F::relu(temp);
        temp = temp.to(at::kCUDA);
        temp = fc64.forward(temp);
        temp = temp.to(at::kCPU);
        src = src + temp;


        //encoder-layer-13


        src0 = src;
        nbatches = src0.size(0);
        temp = src0.to(at::kCUDA);
        temp = fc65.forward(temp);
        temp1 = temp.to(at::kCPU);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src0.to(at::kCUDA);
        temp = fc66.forward(temp);
        temp2 = temp.to(at::kCPU);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc67(src0);
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
        temp = src.to(at::kCUDA);
        temp = fc68.forward(temp);
        temp = temp.to(at::kCPU);
        temp = F::relu(temp);
        temp = temp.to(at::kCUDA);
        temp = fc69.forward(temp);
        temp = temp.to(at::kCPU);
        src = src + temp;


        //encoder-layer-14


        src0 = src;
        nbatches = src0.size(0);
        temp = src0.to(at::kCUDA);
        temp = fc70.forward(temp);
        temp1 = temp.to(at::kCPU);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src0.to(at::kCUDA);
        temp = fc71.forward(temp);
        temp2 = temp.to(at::kCPU);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc72(src0);
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
        temp = src.to(at::kCUDA);
        temp = fc73.forward(temp);
        temp = relu.forward(temp);
        temp = fc74.forward(temp);
        temp = temp.to(at::kCPU);
        src = src + temp;


        //encoder-layer-15


        src0 = src;
        nbatches = src0.size(0);
        temp1 = fc75(src0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc76(src0);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src0.to(at::kCUDA);
        temp = fc77.forward(temp);
        temp3 = temp.to(at::kCPU);
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
        temp = src.to(at::kCUDA);
        temp = fc78.forward(temp);
        temp = relu.forward(temp);
        temp = temp.to(at::kCPU);
        temp = fc79(temp);
        src = src + temp;


        //encoder-layer-16


        src0 = src;
        nbatches = src0.size(0);
        temp = src0.to(at::kCUDA);
        temp = fc80.forward(temp);
        temp1 = temp.to(at::kCPU);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src0.to(at::kCUDA);
        temp = fc81.forward(temp);
        temp2 = temp.to(at::kCPU);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src0.to(at::kCUDA);
        temp = fc82.forward(temp);
        temp3 = temp.to(at::kCPU);
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
        temp = src.to(at::kCUDA);
        temp = fc83.forward(temp);
        temp = temp.to(at::kCPU);
        temp = F::relu(temp);
        temp = fc84(temp);
        src = src + temp;


        //encoder-layer-17


        src0 = src;
        nbatches = src0.size(0);
        temp1 = fc85(src0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc86(src0);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
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
        temp = fc88(src);
        temp = F::relu(temp);
        temp = temp.to(at::kCUDA);
        temp = fc89.forward(temp);
        temp = temp.to(at::kCPU);
        src = src + temp;


        //encoder-layer-18


        src0 = src;
        nbatches = src0.size(0);
        temp1 = fc90(src0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc91(src0);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src0.to(at::kCUDA);
        temp = fc92.forward(temp);
        temp3 = temp.to(at::kCPU);
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
        temp = fc93(src);
        temp = F::relu(temp);
        temp = temp.to(at::kCUDA);
        temp = fc94.forward(temp);
        temp = temp.to(at::kCPU);
        src = src + temp;


        //encoder-layer-19


        src0 = src;
        nbatches = src0.size(0);
        temp = src0.to(at::kCUDA);
        temp = fc95.forward(temp);
        temp1 = temp.to(at::kCPU);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src0.to(at::kCUDA);
        temp = fc96.forward(temp);
        temp2 = temp.to(at::kCPU);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src0.to(at::kCUDA);
        temp = fc97.forward(temp);
        temp3 = temp.to(at::kCPU);
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
        temp = src.to(at::kCUDA);
        temp = fc98.forward(temp);
        temp = temp.to(at::kCPU);
        temp = F::relu(temp);
        temp = fc99(temp);
        src = src + temp;


        //encoder-layer-20


        src0 = src;
        nbatches = src0.size(0);
        temp1 = fc100(src0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc101(src0);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src0.to(at::kCUDA);
        temp = fc102.forward(temp);
        temp3 = temp.to(at::kCPU);
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
        temp = fc103(src);
        temp = F::relu(temp);
        temp = fc104(temp);
        src = src + temp;


        //encoder-layer-21


        src0 = src;
        nbatches = src0.size(0);
        temp = src0.to(at::kCUDA);
        temp = fc105.forward(temp);
        temp1 = temp.to(at::kCPU);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src0.to(at::kCUDA);
        temp = fc106.forward(temp);
        temp2 = temp.to(at::kCPU);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc107(src0);
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
        temp = fc108(src);
        temp = temp.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = temp.to(at::kCPU);
        temp = fc109(temp);
        src = src + temp;


        //encoder-layer-22


        src0 = src;
        nbatches = src0.size(0);
        temp1 = fc110(src0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src0.to(at::kCUDA);
        temp = fc111.forward(temp);
        temp2 = temp.to(at::kCPU);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src0.to(at::kCUDA);
        temp = fc112.forward(temp);
        temp3 = temp.to(at::kCPU);
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
        temp = src.to(at::kCUDA);
        temp = fc113.forward(temp);
        temp = temp.to(at::kCPU);
        temp = F::relu(temp);
        temp = temp.to(at::kCUDA);
        temp = fc114.forward(temp);
        temp = temp.to(at::kCPU);
        src = src + temp;


        //encoder-layer-23


        src0 = src;
        nbatches = src0.size(0);
        temp = src0.to(at::kCUDA);
        temp = fc115.forward(temp);
        temp1 = temp.to(at::kCPU);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc116(src0);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src0.to(at::kCUDA);
        temp = fc117.forward(temp);
        temp3 = temp.to(at::kCPU);
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
        temp = fc118(src);
        temp = F::relu(temp);
        temp = temp.to(at::kCUDA);
        temp = fc119.forward(temp);
        temp = temp.to(at::kCPU);
        src = src + temp;


        //decoder-layer-0


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp1 = fc120(tgt0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = tgt0.to(at::kCUDA);
        temp = fc121.forward(temp);
        temp2 = temp.to(at::kCPU);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = tgt0.to(at::kCUDA);
        temp = fc122.forward(temp);
        temp3 = temp.to(at::kCPU);
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
        temp = tgt0.to(at::kCUDA);
        temp = fc123.forward(temp);
        temp1 = temp.to(at::kCPU);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc124(src);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src.to(at::kCUDA);
        temp = fc125.forward(temp);
        temp3 = temp.to(at::kCPU);
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
        
        temp = tgt.to(at::kCUDA);
        temp = fc126.forward(temp);
        temp = relu.forward(temp);
        temp = fc127.forward(temp);
        temp = temp.to(at::kCPU);
        tgt = tgt + temp;


        //decoder-layer-1


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = tgt0.to(at::kCUDA);
        temp = fc128.forward(temp);
        temp1 = temp.to(at::kCPU);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = tgt0.to(at::kCUDA);
        temp = fc129.forward(temp);
        temp2 = temp.to(at::kCPU);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc130(tgt0);
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
        temp = tgt0.to(at::kCUDA);
        temp = fc131.forward(temp);
        temp1 = temp.to(at::kCPU);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src.to(at::kCUDA);
        temp = fc132.forward(temp);
        temp2 = temp.to(at::kCPU);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src.to(at::kCUDA);
        temp = fc133.forward(temp);
        temp3 = temp.to(at::kCPU);
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
        
        temp = tgt.to(at::kCUDA);
        temp = fc134.forward(temp);
        temp = relu.forward(temp);
        temp = fc135.forward(temp);
        temp = temp.to(at::kCPU);
        tgt = tgt + temp;


        //decoder-layer-2


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp1 = fc136(tgt0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = tgt0.to(at::kCUDA);
        temp = fc137.forward(temp);
        temp2 = temp.to(at::kCPU);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = tgt0.to(at::kCUDA);
        temp = fc138.forward(temp);
        temp3 = temp.to(at::kCPU);
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
        temp1 = fc139(tgt0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc140(src);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
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
        
        temp = tgt.to(at::kCUDA);
        temp = fc142.forward(temp);
        temp = relu.forward(temp);
        temp = fc143.forward(temp);
        temp = temp.to(at::kCPU);
        tgt = tgt + temp;


        //decoder-layer-3


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = tgt0.to(at::kCUDA);
        temp = fc144.forward(temp);
        temp1 = temp.to(at::kCPU);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = tgt0.to(at::kCUDA);
        temp = fc145.forward(temp);
        temp2 = temp.to(at::kCPU);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc146(tgt0);
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
        temp1 = fc147(tgt0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src.to(at::kCUDA);
        temp = fc148.forward(temp);
        temp2 = temp.to(at::kCPU);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src.to(at::kCUDA);
        temp = fc149.forward(temp);
        temp3 = temp.to(at::kCPU);
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
        
        temp = tgt.to(at::kCUDA);
        temp = fc150.forward(temp);
        temp = temp.to(at::kCPU);
        temp = F::relu(temp);
        temp = temp.to(at::kCUDA);
        temp = fc151.forward(temp);
        temp = temp.to(at::kCPU);
        tgt = tgt + temp;


        //decoder-layer-4


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp1 = fc152(tgt0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc153(tgt0);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc154(tgt0);
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
        temp = tgt0.to(at::kCUDA);
        temp = fc155.forward(temp);
        temp1 = temp.to(at::kCPU);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src.to(at::kCUDA);
        temp = fc156.forward(temp);
        temp2 = temp.to(at::kCPU);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc157(src);
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
        
        temp = fc158(tgt);
        temp = F::relu(temp);
        temp = temp.to(at::kCUDA);
        temp = fc159.forward(temp);
        temp = temp.to(at::kCPU);
        tgt = tgt + temp;


        //decoder-layer-5


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp1 = fc160(tgt0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc161(tgt0);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
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
        temp1 = fc163(tgt0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc164(src);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src.to(at::kCUDA);
        temp = fc165.forward(temp);
        temp3 = temp.to(at::kCPU);
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
        
        temp = tgt.to(at::kCUDA);
        temp = fc166.forward(temp);
        temp = relu.forward(temp);
        temp = fc167.forward(temp);
        temp = temp.to(at::kCPU);
        tgt = tgt + temp;


        //decoder-layer-6


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = tgt0.to(at::kCUDA);
        temp = fc168.forward(temp);
        temp1 = temp.to(at::kCPU);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = tgt0.to(at::kCUDA);
        temp = fc169.forward(temp);
        temp2 = temp.to(at::kCPU);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
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
        temp = tgt0.to(at::kCUDA);
        temp = fc171.forward(temp);
        temp1 = temp.to(at::kCPU);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src.to(at::kCUDA);
        temp = fc172.forward(temp);
        temp2 = temp.to(at::kCPU);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src.to(at::kCUDA);
        temp = fc173.forward(temp);
        temp3 = temp.to(at::kCPU);
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
        
        temp = fc174(tgt);
        temp = F::relu(temp);
        temp = temp.to(at::kCUDA);
        temp = fc175.forward(temp);
        temp = temp.to(at::kCPU);
        tgt = tgt + temp;


        //decoder-layer-7


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = tgt0.to(at::kCUDA);
        temp = fc176.forward(temp);
        temp1 = temp.to(at::kCPU);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = tgt0.to(at::kCUDA);
        temp = fc177.forward(temp);
        temp2 = temp.to(at::kCPU);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc178(tgt0);
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
        temp1 = fc179(tgt0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src.to(at::kCUDA);
        temp = fc180.forward(temp);
        temp2 = temp.to(at::kCPU);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src.to(at::kCUDA);
        temp = fc181.forward(temp);
        temp3 = temp.to(at::kCPU);
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
        
        temp = tgt.to(at::kCUDA);
        temp = fc182.forward(temp);
        temp = temp.to(at::kCPU);
        temp = F::relu(temp);
        temp = fc183(temp);
        tgt = tgt + temp;


        //decoder-layer-8


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp1 = fc184(tgt0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = tgt0.to(at::kCUDA);
        temp = fc185.forward(temp);
        temp2 = temp.to(at::kCPU);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc186(tgt0);
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
        temp = tgt0.to(at::kCUDA);
        temp = fc187.forward(temp);
        temp1 = temp.to(at::kCPU);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc188(src);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src.to(at::kCUDA);
        temp = fc189.forward(temp);
        temp3 = temp.to(at::kCPU);
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
        
        temp = tgt.to(at::kCUDA);
        temp = fc190.forward(temp);
        temp = temp.to(at::kCPU);
        temp = F::relu(temp);
        temp = fc191(temp);
        tgt = tgt + temp;


        //decoder-layer-9


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = tgt0.to(at::kCUDA);
        temp = fc192.forward(temp);
        temp1 = temp.to(at::kCPU);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = tgt0.to(at::kCUDA);
        temp = fc193.forward(temp);
        temp2 = temp.to(at::kCPU);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = tgt0.to(at::kCUDA);
        temp = fc194.forward(temp);
        temp3 = temp.to(at::kCPU);
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
        temp = tgt0.to(at::kCUDA);
        temp = fc195.forward(temp);
        temp1 = temp.to(at::kCPU);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src.to(at::kCUDA);
        temp = fc196.forward(temp);
        temp2 = temp.to(at::kCPU);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc197(src);
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
        
        temp = fc198(tgt);
        temp = temp.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = fc199.forward(temp);
        temp = temp.to(at::kCPU);
        tgt = tgt + temp;


        //decoder-layer-10


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = tgt0.to(at::kCUDA);
        temp = fc200.forward(temp);
        temp1 = temp.to(at::kCPU);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = tgt0.to(at::kCUDA);
        temp = fc201.forward(temp);
        temp2 = temp.to(at::kCPU);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc202(tgt0);
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
        temp = tgt0.to(at::kCUDA);
        temp = fc203.forward(temp);
        temp1 = temp.to(at::kCPU);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src.to(at::kCUDA);
        temp = fc204.forward(temp);
        temp2 = temp.to(at::kCPU);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src.to(at::kCUDA);
        temp = fc205.forward(temp);
        temp3 = temp.to(at::kCPU);
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
        
        temp = fc206(tgt);
        temp = temp.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = fc207.forward(temp);
        temp = temp.to(at::kCPU);
        tgt = tgt + temp;


        //decoder-layer-11


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = tgt0.to(at::kCUDA);
        temp = fc208.forward(temp);
        temp1 = temp.to(at::kCPU);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc209(tgt0);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc210(tgt0);
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
        temp = tgt0.to(at::kCUDA);
        temp = fc211.forward(temp);
        temp1 = temp.to(at::kCPU);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc212(src);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src.to(at::kCUDA);
        temp = fc213.forward(temp);
        temp3 = temp.to(at::kCPU);
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
        
        temp = tgt.to(at::kCUDA);
        temp = fc214.forward(temp);
        temp = relu.forward(temp);
        temp = temp.to(at::kCPU);
        temp = fc215(temp);
        tgt = tgt + temp;


        //decoder-layer-12


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = tgt0.to(at::kCUDA);
        temp = fc216.forward(temp);
        temp1 = temp.to(at::kCPU);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = tgt0.to(at::kCUDA);
        temp = fc217.forward(temp);
        temp2 = temp.to(at::kCPU);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = tgt0.to(at::kCUDA);
        temp = fc218.forward(temp);
        temp3 = temp.to(at::kCPU);
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
        temp1 = fc219(tgt0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src.to(at::kCUDA);
        temp = fc220.forward(temp);
        temp2 = temp.to(at::kCPU);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src.to(at::kCUDA);
        temp = fc221.forward(temp);
        temp3 = temp.to(at::kCPU);
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
        
        temp = tgt.to(at::kCUDA);
        temp = fc222.forward(temp);
        temp = relu.forward(temp);
        temp = temp.to(at::kCPU);
        temp = fc223(temp);
        tgt = tgt + temp;


        //decoder-layer-13


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = tgt0.to(at::kCUDA);
        temp = fc224.forward(temp);
        temp1 = temp.to(at::kCPU);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = tgt0.to(at::kCUDA);
        temp = fc225.forward(temp);
        temp2 = temp.to(at::kCPU);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
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
        temp = tgt0.to(at::kCUDA);
        temp = fc227.forward(temp);
        temp1 = temp.to(at::kCPU);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src.to(at::kCUDA);
        temp = fc228.forward(temp);
        temp2 = temp.to(at::kCPU);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc229(src);
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
        
        temp = tgt.to(at::kCUDA);
        temp = fc230.forward(temp);
        temp = relu.forward(temp);
        temp = temp.to(at::kCPU);
        temp = fc231(temp);
        tgt = tgt + temp;


        //decoder-layer-14


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = tgt0.to(at::kCUDA);
        temp = fc232.forward(temp);
        temp1 = temp.to(at::kCPU);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc233(tgt0);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = tgt0.to(at::kCUDA);
        temp = fc234.forward(temp);
        temp3 = temp.to(at::kCPU);
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
        temp1 = fc235(tgt0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src.to(at::kCUDA);
        temp = fc236.forward(temp);
        temp2 = temp.to(at::kCPU);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src.to(at::kCUDA);
        temp = fc237.forward(temp);
        temp3 = temp.to(at::kCPU);
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
        
        temp = tgt.to(at::kCUDA);
        temp = fc238.forward(temp);
        temp = temp.to(at::kCPU);
        temp = F::relu(temp);
        temp = fc239(temp);
        tgt = tgt + temp;


        //decoder-layer-15


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = tgt0.to(at::kCUDA);
        temp = fc240.forward(temp);
        temp1 = temp.to(at::kCPU);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = tgt0.to(at::kCUDA);
        temp = fc241.forward(temp);
        temp2 = temp.to(at::kCPU);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = tgt0.to(at::kCUDA);
        temp = fc242.forward(temp);
        temp3 = temp.to(at::kCPU);
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
        temp1 = fc243(tgt0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src.to(at::kCUDA);
        temp = fc244.forward(temp);
        temp2 = temp.to(at::kCPU);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src.to(at::kCUDA);
        temp = fc245.forward(temp);
        temp3 = temp.to(at::kCPU);
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
        
        temp = tgt.to(at::kCUDA);
        temp = fc246.forward(temp);
        temp = temp.to(at::kCPU);
        temp = F::relu(temp);
        temp = fc247(temp);
        tgt = tgt + temp;


        //decoder-layer-16


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = tgt0.to(at::kCUDA);
        temp = fc248.forward(temp);
        temp1 = temp.to(at::kCPU);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = tgt0.to(at::kCUDA);
        temp = fc249.forward(temp);
        temp2 = temp.to(at::kCPU);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
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
        temp = tgt0.to(at::kCUDA);
        temp = fc251.forward(temp);
        temp1 = temp.to(at::kCPU);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src.to(at::kCUDA);
        temp = fc252.forward(temp);
        temp2 = temp.to(at::kCPU);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc253(src);
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
        
        temp = fc254(tgt);
        temp = temp.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = fc255.forward(temp);
        temp = temp.to(at::kCPU);
        tgt = tgt + temp;


        //decoder-layer-17


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = tgt0.to(at::kCUDA);
        temp = fc256.forward(temp);
        temp1 = temp.to(at::kCPU);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = tgt0.to(at::kCUDA);
        temp = fc257.forward(temp);
        temp2 = temp.to(at::kCPU);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc258(tgt0);
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
        temp = tgt0.to(at::kCUDA);
        temp = fc259.forward(temp);
        temp1 = temp.to(at::kCPU);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src.to(at::kCUDA);
        temp = fc260.forward(temp);
        temp2 = temp.to(at::kCPU);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc261(src);
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
        
        temp = fc262(tgt);
        temp = F::relu(temp);
        temp = fc263(temp);
        tgt = tgt + temp;


        //decoder-layer-18


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp1 = fc264(tgt0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = tgt0.to(at::kCUDA);
        temp = fc265.forward(temp);
        temp2 = temp.to(at::kCPU);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
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
        temp = tgt0.to(at::kCUDA);
        temp = fc267.forward(temp);
        temp1 = temp.to(at::kCPU);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc268(src);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc269(src);
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
        
        temp = tgt.to(at::kCUDA);
        temp = fc270.forward(temp);
        temp = temp.to(at::kCPU);
        temp = F::relu(temp);
        temp = fc271(temp);
        tgt = tgt + temp;


        //decoder-layer-19


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp1 = fc272(tgt0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc273(tgt0);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc274(tgt0);
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
        temp1 = fc275(tgt0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src.to(at::kCUDA);
        temp = fc276.forward(temp);
        temp2 = temp.to(at::kCPU);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src.to(at::kCUDA);
        temp = fc277.forward(temp);
        temp3 = temp.to(at::kCPU);
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
        
        temp = tgt.to(at::kCUDA);
        temp = fc278.forward(temp);
        temp = relu.forward(temp);
        temp = fc279.forward(temp);
        temp = temp.to(at::kCPU);
        tgt = tgt + temp;


        //decoder-layer-20


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = tgt0.to(at::kCUDA);
        temp = fc280.forward(temp);
        temp1 = temp.to(at::kCPU);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = tgt0.to(at::kCUDA);
        temp = fc281.forward(temp);
        temp2 = temp.to(at::kCPU);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc282(tgt0);
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
        temp = tgt0.to(at::kCUDA);
        temp = fc283.forward(temp);
        temp1 = temp.to(at::kCPU);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc284(src);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src.to(at::kCUDA);
        temp = fc285.forward(temp);
        temp3 = temp.to(at::kCPU);
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
        
        temp = fc286(tgt);
        temp = F::relu(temp);
        temp = fc287(temp);
        tgt = tgt + temp;


        //decoder-layer-21


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = tgt0.to(at::kCUDA);
        temp = fc288.forward(temp);
        temp1 = temp.to(at::kCPU);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc289(tgt0);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = tgt0.to(at::kCUDA);
        temp = fc290.forward(temp);
        temp3 = temp.to(at::kCPU);
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
        temp1 = fc291(tgt0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src.to(at::kCUDA);
        temp = fc292.forward(temp);
        temp2 = temp.to(at::kCPU);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src.to(at::kCUDA);
        temp = fc293.forward(temp);
        temp3 = temp.to(at::kCPU);
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
        
        temp = fc294(tgt);
        temp = F::relu(temp);
        temp = fc295(temp);
        tgt = tgt + temp;


        //decoder-layer-22


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = tgt0.to(at::kCUDA);
        temp = fc296.forward(temp);
        temp1 = temp.to(at::kCPU);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = tgt0.to(at::kCUDA);
        temp = fc297.forward(temp);
        temp2 = temp.to(at::kCPU);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = tgt0.to(at::kCUDA);
        temp = fc298.forward(temp);
        temp3 = temp.to(at::kCPU);
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
        temp1 = fc299(tgt0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src.to(at::kCUDA);
        temp = fc300.forward(temp);
        temp2 = temp.to(at::kCPU);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc301(src);
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
        
        temp = fc302(tgt);
        temp = F::relu(temp);
        temp = temp.to(at::kCUDA);
        temp = fc303.forward(temp);
        temp = temp.to(at::kCPU);
        tgt = tgt + temp;


        //decoder-layer-23


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp1 = fc304(tgt0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = tgt0.to(at::kCUDA);
        temp = fc305.forward(temp);
        temp2 = temp.to(at::kCPU);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = tgt0.to(at::kCUDA);
        temp = fc306.forward(temp);
        temp3 = temp.to(at::kCPU);
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
        temp1 = fc307(tgt0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc308(src);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
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
        
        temp = tgt.to(at::kCUDA);
        temp = fc310.forward(temp);
        temp = relu.forward(temp);
        temp = fc311.forward(temp);
        temp = temp.to(at::kCPU);
        tgt = tgt + temp;

        return tgt;
    }
};


int main(int argc, char* argv[])
{
    std::cout << "Please input the token size..." << std::endl;
    d_model = std::stoi(argv[1]);
    nheads = 8;
    d_k = d_model / nheads;
    d_v = d_k;
    sqrt_dk = (int)floor(sqrt((double)d_v));
    d_ff = 2048;
    std::cout << "Read integer: " << d_model << std::endl;

    transformer model;
    std::cout << "Transformer - 50% outsourced to CUDA version" << std::endl;
    std::cout << "Kernel Switch: 296" << std::endl;
    std::cout << "Associative Op: 360" << std::endl;
    std::cout << "Outsourced Op: 202" << std::endl;

    torch::Tensor src = torch::rand( {32, 10, d_model} );
    torch::Tensor tgt = torch::rand( {32, 20, d_model} );
    torch::Tensor out;

    int count = 1000;
    int warmup = 3000;

    for (size_t i = 0; i < warmup; i++)
    {
        out = model.forward(src, tgt);
    }

    cudaEvent_t start, stop;
    float esp_time_gpu;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float total = 0;

    for (size_t i = 0; i < count; i++)
    {
        cudaEventRecord(start, 0);
        out = model.forward(src, tgt);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&esp_time_gpu, start, stop);
        total += esp_time_gpu;
    }

    float latency;
    latency = total / ((float)count);
    std::cout << "For " << count << " inferences..." << std::endl;
    std::cout << "Time elapsed: " << total << " ms." << std::endl;
    std::cout << "Time consuming: " << latency << " ms per instance." << std::endl;
    std::cout << "completed." << std::endl;
    return 0;
}