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
    operator7 fc0;
    operator7 fc1;
    operator7 fc2;
    operator7 fc3;
    operator7 fc4;
    torch::nn::Linear fc5;
    operator7 fc6;
    operator7 fc7;
    torch::nn::Linear fc8;
    operator7 fc9;
    operator7 fc10;
    operator7 fc11;
    torch::nn::Linear fc12;
    torch::nn::Linear fc13;
    operator7 fc14;
    operator7 fc15;
    torch::nn::Linear fc16;
    torch::nn::Linear fc17;
    operator7 fc18;
    operator7 fc19;
    torch::nn::Linear fc20;
    torch::nn::Linear fc21;
    operator7 fc22;
    operator7 fc23;
    torch::nn::Linear fc24;
    operator7 fc25;
    torch::nn::Linear fc26;
    torch::nn::Linear fc27;
    torch::nn::Linear fc28;
    torch::nn::Linear fc29;
    torch::nn::Linear fc30;
    torch::nn::Linear fc31;
    torch::nn::Linear fc32;
    operator7 fc33;
    torch::nn::Linear fc34;
    torch::nn::Linear fc35;
    torch::nn::Linear fc36;
    operator7 fc37;
    torch::nn::Linear fc38;
    torch::nn::Linear fc39;
    operator7 fc40;
    torch::nn::Linear fc41;
    torch::nn::Linear fc42;
    operator7 fc43;
    torch::nn::Linear fc44;
    torch::nn::Linear fc45;
    operator7 fc46;
    operator7 fc47;
    operator7 fc48;
    operator7 fc49;
    torch::nn::Linear fc50;
    torch::nn::Linear fc51;
    torch::nn::Linear fc52;
    operator7 fc53;
    torch::nn::Linear fc54;
    torch::nn::Linear fc55;
    operator7 fc56;
    torch::nn::Linear fc57;
    operator7 fc58;
    operator7 fc59;
    operator7 fc60;
    torch::nn::Linear fc61;
    operator7 fc62;
    torch::nn::Linear fc63;
    operator7 fc64;
    torch::nn::Linear fc65;
    torch::nn::Linear fc66;
    torch::nn::Linear fc67;
    operator7 fc68;
    operator7 fc69;
    torch::nn::Linear fc70;
    torch::nn::Linear fc71;
    torch::nn::Linear fc72;
    torch::nn::Linear fc73;
    torch::nn::Linear fc74;
    operator7 fc75;
    torch::nn::Linear fc76;
    torch::nn::Linear fc77;
    torch::nn::Linear fc78;
    torch::nn::Linear fc79;
    torch::nn::Linear fc80;
    operator7 fc81;
    torch::nn::Linear fc82;
    operator7 fc83;
    torch::nn::Linear fc84;
    torch::nn::Linear fc85;
    operator7 fc86;
    operator7 fc87;
    torch::nn::Linear fc88;
    operator7 fc89;
    torch::nn::Linear fc90;
    torch::nn::Linear fc91;
    torch::nn::Linear fc92;
    torch::nn::Linear fc93;
    operator7 fc94;
    torch::nn::Linear fc95;
    operator7 fc96;
    torch::nn::Linear fc97;
    torch::nn::Linear fc98;
    torch::nn::Linear fc99;
    torch::nn::Linear fc100;
    operator7 fc101;
    torch::nn::Linear fc102;
    operator7 fc103;
    operator7 fc104;
    torch::nn::Linear fc105;
    operator7 fc106;
    torch::nn::Linear fc107;
    torch::nn::Linear fc108;
    operator7 fc109;
    operator7 fc110;
    operator7 fc111;
    torch::nn::Linear fc112;
    torch::nn::Linear fc113;
    torch::nn::Linear fc114;
    torch::nn::Linear fc115;
    operator7 fc116;
    torch::nn::Linear fc117;
    torch::nn::Linear fc118;
    operator7 fc119;
    torch::nn::Linear fc120;
    torch::nn::Linear fc121;
    torch::nn::Linear fc122;
    operator7 fc123;
    torch::nn::Linear fc124;
    torch::nn::Linear fc125;
    operator7 fc126;
    torch::nn::Linear fc127;
    torch::nn::Linear fc128;
    torch::nn::Linear fc129;
    operator7 fc130;
    operator7 fc131;
    operator7 fc132;
    torch::nn::Linear fc133;
    operator7 fc134;
    torch::nn::Linear fc135;
    torch::nn::Linear fc136;
    operator7 fc137;
    operator7 fc138;
    torch::nn::Linear fc139;
    torch::nn::Linear fc140;
    torch::nn::Linear fc141;
    operator7 fc142;
    torch::nn::Linear fc143;
    torch::nn::Linear fc144;
    operator7 fc145;
    operator7 fc146;
    torch::nn::Linear fc147;
    torch::nn::Linear fc148;
    torch::nn::Linear fc149;
    torch::nn::Linear fc150;
    torch::nn::Linear fc151;
    torch::nn::Linear fc152;
    torch::nn::Linear fc153;
    torch::nn::Linear fc154;
    operator7 fc155;
    operator7 fc156;
    torch::nn::Linear fc157;
    torch::nn::Linear fc158;
    torch::nn::Linear fc159;
    operator7 fc160;
    torch::nn::Linear fc161;
    operator7 fc162;
    operator7 fc163;
    operator7 fc164;
    torch::nn::Linear fc165;
    torch::nn::Linear fc166;
    torch::nn::Linear fc167;
    torch::nn::Linear fc168;
    operator7 fc169;
    torch::nn::Linear fc170;
    torch::nn::Linear fc171;
    operator7 fc172;
    operator7 fc173;
    torch::nn::Linear fc174;
    torch::nn::Linear fc175;
    torch::nn::Linear fc176;
    operator7 fc177;
    torch::nn::Linear fc178;
    operator7 fc179;
    operator7 fc180;
    torch::nn::Linear fc181;
    operator7 fc182;
    torch::nn::Linear fc183;
    torch::nn::Linear fc184;
    torch::nn::Linear fc185;
    torch::nn::Linear fc186;
    torch::nn::Linear fc187;
    torch::nn::Linear fc188;
    torch::nn::Linear fc189;
    torch::nn::Linear fc190;
    operator7 fc191;
    torch::nn::Linear fc192;
    torch::nn::Linear fc193;
    torch::nn::Linear fc194;
    torch::nn::Linear fc195;
    operator7 fc196;
    torch::nn::Linear fc197;
    torch::nn::Linear fc198;
    torch::nn::Linear fc199;
    operator7 fc200;
    torch::nn::Linear fc201;
    torch::nn::Linear fc202;
    torch::nn::Linear fc203;
    torch::nn::Linear fc204;
    torch::nn::Linear fc205;
    operator7 fc206;
    operator7 fc207;
    operator7 fc208;
    torch::nn::Linear fc209;
    torch::nn::Linear fc210;
    torch::nn::Linear fc211;
    operator7 fc212;
    operator7 fc213;
    torch::nn::Linear fc214;
    torch::nn::Linear fc215;
    torch::nn::Linear fc216;
    operator7 fc217;
    torch::nn::Linear fc218;
    operator7 fc219;
    operator7 fc220;
    torch::nn::Linear fc221;
    torch::nn::Linear fc222;
    operator7 fc223;
    torch::nn::Linear fc224;
    torch::nn::Linear fc225;
    torch::nn::Linear fc226;
    operator7 fc227;
    torch::nn::Linear fc228;
    operator7 fc229;
    operator7 fc230;
    torch::nn::Linear fc231;
    torch::nn::Linear fc232;
    torch::nn::Linear fc233;
    operator7 fc234;
    torch::nn::Linear fc235;
    torch::nn::Linear fc236;
    torch::nn::Linear fc237;
    operator7 fc238;
    torch::nn::Linear fc239;
    torch::nn::Linear fc240;
    operator7 fc241;
    operator7 fc242;
    torch::nn::Linear fc243;
    operator7 fc244;
    operator7 fc245;
    torch::nn::Linear fc246;
    operator7 fc247;
    torch::nn::Linear fc248;
    torch::nn::Linear fc249;
    operator7 fc250;
    operator7 fc251;
    operator7 fc252;
    operator7 fc253;
    operator7 fc254;
    torch::nn::Linear fc255;
    torch::nn::Linear fc256;
    operator7 fc257;
    torch::nn::Linear fc258;
    torch::nn::Linear fc259;
    torch::nn::Linear fc260;
    torch::nn::Linear fc261;
    torch::nn::Linear fc262;
    operator7 fc263;
    torch::nn::Linear fc264;
    operator7 fc265;
    torch::nn::Linear fc266;
    operator7 fc267;
    operator7 fc268;
    operator7 fc269;
    torch::nn::Linear fc270;
    torch::nn::Linear fc271;
    torch::nn::Linear fc272;
    torch::nn::Linear fc273;
    torch::nn::Linear fc274;
    torch::nn::Linear fc275;
    operator7 fc276;
    torch::nn::Linear fc277;
    torch::nn::Linear fc278;
    operator7 fc279;
    torch::nn::Linear fc280;
    torch::nn::Linear fc281;
    torch::nn::Linear fc282;
    torch::nn::Linear fc283;
    torch::nn::Linear fc284;
    operator7 fc285;
    torch::nn::Linear fc286;
    torch::nn::Linear fc287;
    torch::nn::Linear fc288;
    operator7 fc289;
    operator7 fc290;
    torch::nn::Linear fc291;
    torch::nn::Linear fc292;
    operator7 fc293;
    operator7 fc294;
    torch::nn::Linear fc295;
    torch::nn::Linear fc296;
    torch::nn::Linear fc297;
    operator7 fc298;
    operator7 fc299;
    torch::nn::Linear fc300;
    torch::nn::Linear fc301;
    operator7 fc302;
    torch::nn::Linear fc303;
    operator7 fc304;
    torch::nn::Linear fc305;
    operator7 fc306;
    torch::nn::Linear fc307;
    operator7 fc308;
    torch::nn::Linear fc309;
    torch::nn::Linear fc310;
    torch::nn::Linear fc311;
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
        fc0.to(at::kCUDA);
        fc1.to(at::kCUDA);
        fc2.to(at::kCUDA);
        fc3.to(at::kCUDA);
        fc4.to(at::kCUDA);
        register_module("fc5", fc5);
        fc6.to(at::kCUDA);
        fc7.to(at::kCUDA);
        register_module("fc8", fc8);
        fc9.to(at::kCUDA);
        fc10.to(at::kCUDA);
        fc11.to(at::kCUDA);
        register_module("fc12", fc12);
        register_module("fc13", fc13);
        fc14.to(at::kCUDA);
        fc15.to(at::kCUDA);
        register_module("fc16", fc16);
        register_module("fc17", fc17);
        fc18.to(at::kCUDA);
        fc19.to(at::kCUDA);
        register_module("fc20", fc20);
        register_module("fc21", fc21);
        fc22.to(at::kCUDA);
        fc23.to(at::kCUDA);
        register_module("fc24", fc24);
        fc25.to(at::kCUDA);
        register_module("fc26", fc26);
        register_module("fc27", fc27);
        register_module("fc28", fc28);
        register_module("fc29", fc29);
        register_module("fc30", fc30);
        register_module("fc31", fc31);
        register_module("fc32", fc32);
        fc33.to(at::kCUDA);
        register_module("fc34", fc34);
        register_module("fc35", fc35);
        register_module("fc36", fc36);
        fc37.to(at::kCUDA);
        register_module("fc38", fc38);
        register_module("fc39", fc39);
        fc40.to(at::kCUDA);
        register_module("fc41", fc41);
        register_module("fc42", fc42);
        fc43.to(at::kCUDA);
        register_module("fc44", fc44);
        register_module("fc45", fc45);
        fc46.to(at::kCUDA);
        fc47.to(at::kCUDA);
        fc48.to(at::kCUDA);
        fc49.to(at::kCUDA);
        register_module("fc50", fc50);
        register_module("fc51", fc51);
        register_module("fc52", fc52);
        fc53.to(at::kCUDA);
        register_module("fc54", fc54);
        register_module("fc55", fc55);
        fc56.to(at::kCUDA);
        register_module("fc57", fc57);
        fc58.to(at::kCUDA);
        fc59.to(at::kCUDA);
        fc60.to(at::kCUDA);
        register_module("fc61", fc61);
        fc62.to(at::kCUDA);
        register_module("fc63", fc63);
        fc64.to(at::kCUDA);
        register_module("fc65", fc65);
        register_module("fc66", fc66);
        register_module("fc67", fc67);
        fc68.to(at::kCUDA);
        fc69.to(at::kCUDA);
        register_module("fc70", fc70);
        register_module("fc71", fc71);
        register_module("fc72", fc72);
        register_module("fc73", fc73);
        register_module("fc74", fc74);
        fc75.to(at::kCUDA);
        register_module("fc76", fc76);
        register_module("fc77", fc77);
        register_module("fc78", fc78);
        register_module("fc79", fc79);
        register_module("fc80", fc80);
        fc81.to(at::kCUDA);
        register_module("fc82", fc82);
        fc83.to(at::kCUDA);
        register_module("fc84", fc84);
        register_module("fc85", fc85);
        fc86.to(at::kCUDA);
        fc87.to(at::kCUDA);
        register_module("fc88", fc88);
        fc89.to(at::kCUDA);
        register_module("fc90", fc90);
        register_module("fc91", fc91);
        register_module("fc92", fc92);
        register_module("fc93", fc93);
        fc94.to(at::kCUDA);
        register_module("fc95", fc95);
        fc96.to(at::kCUDA);
        register_module("fc97", fc97);
        register_module("fc98", fc98);
        register_module("fc99", fc99);
        register_module("fc100", fc100);
        fc101.to(at::kCUDA);
        register_module("fc102", fc102);
        fc103.to(at::kCUDA);
        fc104.to(at::kCUDA);
        register_module("fc105", fc105);
        fc106.to(at::kCUDA);
        register_module("fc107", fc107);
        register_module("fc108", fc108);
        fc109.to(at::kCUDA);
        fc110.to(at::kCUDA);
        fc111.to(at::kCUDA);
        register_module("fc112", fc112);
        register_module("fc113", fc113);
        register_module("fc114", fc114);
        register_module("fc115", fc115);
        fc116.to(at::kCUDA);
        register_module("fc117", fc117);
        register_module("fc118", fc118);
        fc119.to(at::kCUDA);
        register_module("fc120", fc120);
        register_module("fc121", fc121);
        register_module("fc122", fc122);
        fc123.to(at::kCUDA);
        register_module("fc124", fc124);
        register_module("fc125", fc125);
        fc126.to(at::kCUDA);
        register_module("fc127", fc127);
        register_module("fc128", fc128);
        register_module("fc129", fc129);
        fc130.to(at::kCUDA);
        fc131.to(at::kCUDA);
        fc132.to(at::kCUDA);
        register_module("fc133", fc133);
        fc134.to(at::kCUDA);
        register_module("fc135", fc135);
        register_module("fc136", fc136);
        fc137.to(at::kCUDA);
        fc138.to(at::kCUDA);
        register_module("fc139", fc139);
        register_module("fc140", fc140);
        register_module("fc141", fc141);
        fc142.to(at::kCUDA);
        register_module("fc143", fc143);
        register_module("fc144", fc144);
        fc145.to(at::kCUDA);
        fc146.to(at::kCUDA);
        register_module("fc147", fc147);
        register_module("fc148", fc148);
        register_module("fc149", fc149);
        register_module("fc150", fc150);
        register_module("fc151", fc151);
        register_module("fc152", fc152);
        register_module("fc153", fc153);
        register_module("fc154", fc154);
        fc155.to(at::kCUDA);
        fc156.to(at::kCUDA);
        register_module("fc157", fc157);
        register_module("fc158", fc158);
        register_module("fc159", fc159);
        fc160.to(at::kCUDA);
        register_module("fc161", fc161);
        fc162.to(at::kCUDA);
        fc163.to(at::kCUDA);
        fc164.to(at::kCUDA);
        register_module("fc165", fc165);
        register_module("fc166", fc166);
        register_module("fc167", fc167);
        register_module("fc168", fc168);
        fc169.to(at::kCUDA);
        register_module("fc170", fc170);
        register_module("fc171", fc171);
        fc172.to(at::kCUDA);
        fc173.to(at::kCUDA);
        register_module("fc174", fc174);
        register_module("fc175", fc175);
        register_module("fc176", fc176);
        fc177.to(at::kCUDA);
        register_module("fc178", fc178);
        fc179.to(at::kCUDA);
        fc180.to(at::kCUDA);
        register_module("fc181", fc181);
        fc182.to(at::kCUDA);
        register_module("fc183", fc183);
        register_module("fc184", fc184);
        register_module("fc185", fc185);
        register_module("fc186", fc186);
        register_module("fc187", fc187);
        register_module("fc188", fc188);
        register_module("fc189", fc189);
        register_module("fc190", fc190);
        fc191.to(at::kCUDA);
        register_module("fc192", fc192);
        register_module("fc193", fc193);
        register_module("fc194", fc194);
        register_module("fc195", fc195);
        fc196.to(at::kCUDA);
        register_module("fc197", fc197);
        register_module("fc198", fc198);
        register_module("fc199", fc199);
        fc200.to(at::kCUDA);
        register_module("fc201", fc201);
        register_module("fc202", fc202);
        register_module("fc203", fc203);
        register_module("fc204", fc204);
        register_module("fc205", fc205);
        fc206.to(at::kCUDA);
        fc207.to(at::kCUDA);
        fc208.to(at::kCUDA);
        register_module("fc209", fc209);
        register_module("fc210", fc210);
        register_module("fc211", fc211);
        fc212.to(at::kCUDA);
        fc213.to(at::kCUDA);
        register_module("fc214", fc214);
        register_module("fc215", fc215);
        register_module("fc216", fc216);
        fc217.to(at::kCUDA);
        register_module("fc218", fc218);
        fc219.to(at::kCUDA);
        fc220.to(at::kCUDA);
        register_module("fc221", fc221);
        register_module("fc222", fc222);
        fc223.to(at::kCUDA);
        register_module("fc224", fc224);
        register_module("fc225", fc225);
        register_module("fc226", fc226);
        fc227.to(at::kCUDA);
        register_module("fc228", fc228);
        fc229.to(at::kCUDA);
        fc230.to(at::kCUDA);
        register_module("fc231", fc231);
        register_module("fc232", fc232);
        register_module("fc233", fc233);
        fc234.to(at::kCUDA);
        register_module("fc235", fc235);
        register_module("fc236", fc236);
        register_module("fc237", fc237);
        fc238.to(at::kCUDA);
        register_module("fc239", fc239);
        register_module("fc240", fc240);
        fc241.to(at::kCUDA);
        fc242.to(at::kCUDA);
        register_module("fc243", fc243);
        fc244.to(at::kCUDA);
        fc245.to(at::kCUDA);
        register_module("fc246", fc246);
        fc247.to(at::kCUDA);
        register_module("fc248", fc248);
        register_module("fc249", fc249);
        fc250.to(at::kCUDA);
        fc251.to(at::kCUDA);
        fc252.to(at::kCUDA);
        fc253.to(at::kCUDA);
        fc254.to(at::kCUDA);
        register_module("fc255", fc255);
        register_module("fc256", fc256);
        fc257.to(at::kCUDA);
        register_module("fc258", fc258);
        register_module("fc259", fc259);
        register_module("fc260", fc260);
        register_module("fc261", fc261);
        register_module("fc262", fc262);
        fc263.to(at::kCUDA);
        register_module("fc264", fc264);
        fc265.to(at::kCUDA);
        register_module("fc266", fc266);
        fc267.to(at::kCUDA);
        fc268.to(at::kCUDA);
        fc269.to(at::kCUDA);
        register_module("fc270", fc270);
        register_module("fc271", fc271);
        register_module("fc272", fc272);
        register_module("fc273", fc273);
        register_module("fc274", fc274);
        register_module("fc275", fc275);
        fc276.to(at::kCUDA);
        register_module("fc277", fc277);
        register_module("fc278", fc278);
        fc279.to(at::kCUDA);
        register_module("fc280", fc280);
        register_module("fc281", fc281);
        register_module("fc282", fc282);
        register_module("fc283", fc283);
        register_module("fc284", fc284);
        fc285.to(at::kCUDA);
        register_module("fc286", fc286);
        register_module("fc287", fc287);
        register_module("fc288", fc288);
        fc289.to(at::kCUDA);
        fc290.to(at::kCUDA);
        register_module("fc291", fc291);
        register_module("fc292", fc292);
        fc293.to(at::kCUDA);
        fc294.to(at::kCUDA);
        register_module("fc295", fc295);
        register_module("fc296", fc296);
        register_module("fc297", fc297);
        fc298.to(at::kCUDA);
        fc299.to(at::kCUDA);
        register_module("fc300", fc300);
        register_module("fc301", fc301);
        fc302.to(at::kCUDA);
        register_module("fc303", fc303);
        fc304.to(at::kCUDA);
        register_module("fc305", fc305);
        fc306.to(at::kCUDA);
        register_module("fc307", fc307);
        fc308.to(at::kCUDA);
        register_module("fc309", fc309);
        register_module("fc310", fc310);
        register_module("fc311", fc311);
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
        temp = src0.to(at::kCUDA);
        temp = fc0.forward(temp);
        temp1 = temp.to(at::kCPU);
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
        temp = fc8(src);
        temp = F::relu(temp);
        temp = temp.to(at::kCUDA);
        temp = fc9.forward(temp);
        temp = temp.to(at::kCPU);
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
        temp3 = fc12(src0);
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
        temp = src0.to(at::kCUDA);
        temp = fc15.forward(temp);
        temp1 = temp.to(at::kCPU);
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
        temp = src.to(at::kCUDA);
        temp = fc18.forward(temp);
        temp = temp.to(at::kCPU);
        temp = F::relu(temp);
        temp = temp.to(at::kCUDA);
        temp = fc19.forward(temp);
        temp = temp.to(at::kCPU);
        src = src + temp;


        //encoder-layer-4


        src0 = src;
        nbatches = src0.size(0);
        temp1 = fc20(src0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc21(src0);
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
        temp = temp.to(at::kCPU);
        temp = F::relu(temp);
        temp = fc24(temp);
        src = src + temp;


        //encoder-layer-5


        src0 = src;
        nbatches = src0.size(0);
        temp = src0.to(at::kCUDA);
        temp = fc25.forward(temp);
        temp1 = temp.to(at::kCPU);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc26(src0);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc27(src0);
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
        temp = fc28(src);
        temp = temp.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = temp.to(at::kCPU);
        temp = fc29(temp);
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
        temp = src.to(at::kCUDA);
        temp = fc33.forward(temp);
        temp = relu.forward(temp);
        temp = temp.to(at::kCPU);
        temp = fc34(temp);
        src = src + temp;


        //encoder-layer-7


        src0 = src;
        nbatches = src0.size(0);
        temp1 = fc35(src0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc36(src0);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src0.to(at::kCUDA);
        temp = fc37.forward(temp);
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
        temp = fc38(src);
        temp = F::relu(temp);
        temp = fc39(temp);
        src = src + temp;


        //encoder-layer-8


        src0 = src;
        nbatches = src0.size(0);
        temp = src0.to(at::kCUDA);
        temp = fc40.forward(temp);
        temp1 = temp.to(at::kCPU);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc41(src0);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc42(src0);
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
        temp = fc43.forward(temp);
        temp = relu.forward(temp);
        temp = temp.to(at::kCPU);
        temp = fc44(temp);
        src = src + temp;


        //encoder-layer-9


        src0 = src;
        nbatches = src0.size(0);
        temp1 = fc45(src0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src0.to(at::kCUDA);
        temp = fc46.forward(temp);
        temp2 = temp.to(at::kCPU);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src0.to(at::kCUDA);
        temp = fc47.forward(temp);
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
        temp = fc48.forward(temp);
        temp = temp.to(at::kCPU);
        temp = F::relu(temp);
        temp = temp.to(at::kCUDA);
        temp = fc49.forward(temp);
        temp = temp.to(at::kCPU);
        src = src + temp;


        //encoder-layer-10


        src0 = src;
        nbatches = src0.size(0);
        temp1 = fc50(src0);
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
        temp = src0.to(at::kCUDA);
        temp = fc56.forward(temp);
        temp2 = temp.to(at::kCPU);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc57(src0);
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
        temp = fc58.forward(temp);
        temp = relu.forward(temp);
        temp = fc59.forward(temp);
        temp = temp.to(at::kCPU);
        src = src + temp;


        //encoder-layer-12


        src0 = src;
        nbatches = src0.size(0);
        temp = src0.to(at::kCUDA);
        temp = fc60.forward(temp);
        temp1 = temp.to(at::kCPU);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc61(src0);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src0.to(at::kCUDA);
        temp = fc62.forward(temp);
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
        temp = fc63(src);
        temp = F::relu(temp);
        temp = temp.to(at::kCUDA);
        temp = fc64.forward(temp);
        temp = temp.to(at::kCPU);
        src = src + temp;


        //encoder-layer-13


        src0 = src;
        nbatches = src0.size(0);
        temp1 = fc65(src0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc66(src0);
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
        temp = relu.forward(temp);
        temp = fc69.forward(temp);
        temp = temp.to(at::kCPU);
        src = src + temp;


        //encoder-layer-14


        src0 = src;
        nbatches = src0.size(0);
        temp1 = fc70(src0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc71(src0);
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
        temp = fc73(src);
        temp = F::relu(temp);
        temp = fc74(temp);
        src = src + temp;


        //encoder-layer-15


        src0 = src;
        nbatches = src0.size(0);
        temp = src0.to(at::kCUDA);
        temp = fc75.forward(temp);
        temp1 = temp.to(at::kCPU);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc76(src0);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc77(src0);
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
        temp = fc78(src);
        temp = temp.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = temp.to(at::kCPU);
        temp = fc79(temp);
        src = src + temp;


        //encoder-layer-16


        src0 = src;
        nbatches = src0.size(0);
        temp1 = fc80(src0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src0.to(at::kCUDA);
        temp = fc81.forward(temp);
        temp2 = temp.to(at::kCPU);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc82(src0);
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
        temp = relu.forward(temp);
        temp = temp.to(at::kCPU);
        temp = fc84(temp);
        src = src + temp;


        //encoder-layer-17


        src0 = src;
        nbatches = src0.size(0);
        temp1 = fc85(src0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src0.to(at::kCUDA);
        temp = fc86.forward(temp);
        temp2 = temp.to(at::kCPU);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src0.to(at::kCUDA);
        temp = fc87.forward(temp);
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
        temp = fc88(src);
        temp = temp.to(at::kCUDA);
        temp = relu.forward(temp);
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
        temp3 = fc92(src0);
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
        temp1 = fc95(src0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src0.to(at::kCUDA);
        temp = fc96.forward(temp);
        temp2 = temp.to(at::kCPU);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc97(src0);
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
        temp = fc98(src);
        temp = F::relu(temp);
        temp = fc99(temp);
        src = src + temp;


        //encoder-layer-20


        src0 = src;
        nbatches = src0.size(0);
        temp1 = fc100(src0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src0.to(at::kCUDA);
        temp = fc101.forward(temp);
        temp2 = temp.to(at::kCPU);
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
        temp = src.to(at::kCUDA);
        temp = fc103.forward(temp);
        temp = relu.forward(temp);
        temp = fc104.forward(temp);
        temp = temp.to(at::kCPU);
        src = src + temp;


        //encoder-layer-21


        src0 = src;
        nbatches = src0.size(0);
        temp1 = fc105(src0);
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
        temp = F::relu(temp);
        temp = temp.to(at::kCUDA);
        temp = fc109.forward(temp);
        temp = temp.to(at::kCPU);
        src = src + temp;


        //encoder-layer-22


        src0 = src;
        nbatches = src0.size(0);
        temp = src0.to(at::kCUDA);
        temp = fc110.forward(temp);
        temp1 = temp.to(at::kCPU);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src0.to(at::kCUDA);
        temp = fc111.forward(temp);
        temp2 = temp.to(at::kCPU);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc112(src0);
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
        temp = fc113(src);
        temp = F::relu(temp);
        temp = fc114(temp);
        src = src + temp;


        //encoder-layer-23


        src0 = src;
        nbatches = src0.size(0);
        temp1 = fc115(src0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src0.to(at::kCUDA);
        temp = fc116.forward(temp);
        temp2 = temp.to(at::kCPU);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc117(src0);
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
        temp2 = fc121(tgt0);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc122(tgt0);
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
        temp3 = fc125(src);
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
        temp = temp.to(at::kCPU);
        temp = fc127(temp);
        tgt = tgt + temp;


        //decoder-layer-1


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp1 = fc128(tgt0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc129(tgt0);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = tgt0.to(at::kCUDA);
        temp = fc130.forward(temp);
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
        temp = fc131.forward(temp);
        temp1 = temp.to(at::kCPU);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src.to(at::kCUDA);
        temp = fc132.forward(temp);
        temp2 = temp.to(at::kCPU);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc133(src);
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
        temp = temp.to(at::kCPU);
        temp = F::relu(temp);
        temp = fc135(temp);
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
        temp = temp.to(at::kCPU);
        temp = F::relu(temp);
        temp = fc143(temp);
        tgt = tgt + temp;


        //decoder-layer-3


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp1 = fc144(tgt0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = tgt0.to(at::kCUDA);
        temp = fc145.forward(temp);
        temp2 = temp.to(at::kCPU);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = tgt0.to(at::kCUDA);
        temp = fc146.forward(temp);
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
        temp1 = fc147(tgt0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc148(src);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
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
        
        temp = fc150(tgt);
        temp = temp.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = temp.to(at::kCPU);
        temp = fc151(temp);
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
        temp = fc159(temp);
        tgt = tgt + temp;


        //decoder-layer-5


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = tgt0.to(at::kCUDA);
        temp = fc160.forward(temp);
        temp1 = temp.to(at::kCPU);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc161(tgt0);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = tgt0.to(at::kCUDA);
        temp = fc162.forward(temp);
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
        temp = fc163.forward(temp);
        temp1 = temp.to(at::kCPU);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src.to(at::kCUDA);
        temp = fc164.forward(temp);
        temp2 = temp.to(at::kCPU);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc165(src);
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
        
        temp = fc166(tgt);
        temp = F::relu(temp);
        temp = fc167(temp);
        tgt = tgt + temp;


        //decoder-layer-6


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp1 = fc168(tgt0);
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
        temp1 = fc171(tgt0);
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
        temp = temp.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = temp.to(at::kCPU);
        temp = fc175(temp);
        tgt = tgt + temp;


        //decoder-layer-7


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp1 = fc176(tgt0);
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
        temp = tgt0.to(at::kCUDA);
        temp = fc179.forward(temp);
        temp1 = temp.to(at::kCPU);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src.to(at::kCUDA);
        temp = fc180.forward(temp);
        temp2 = temp.to(at::kCPU);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc181(src);
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
        temp2 = fc185(tgt0);
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
        temp1 = fc187(tgt0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc188(src);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc189(src);
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
        
        temp = fc190(tgt);
        temp = F::relu(temp);
        temp = temp.to(at::kCUDA);
        temp = fc191.forward(temp);
        temp = temp.to(at::kCPU);
        tgt = tgt + temp;


        //decoder-layer-9


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp1 = fc192(tgt0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc193(tgt0);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc194(tgt0);
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
        temp1 = fc195(tgt0);
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
        temp = temp.to(at::kCPU);
        temp = fc199(temp);
        tgt = tgt + temp;


        //decoder-layer-10


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = tgt0.to(at::kCUDA);
        temp = fc200.forward(temp);
        temp1 = temp.to(at::kCPU);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc201(tgt0);
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
        temp1 = fc203(tgt0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc204(src);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc205(src);
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
        temp = fc206.forward(temp);
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
        temp1 = fc211(tgt0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src.to(at::kCUDA);
        temp = fc212.forward(temp);
        temp2 = temp.to(at::kCPU);
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
        
        temp = fc214(tgt);
        temp = F::relu(temp);
        temp = fc215(temp);
        tgt = tgt + temp;


        //decoder-layer-12


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp1 = fc216(tgt0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = tgt0.to(at::kCUDA);
        temp = fc217.forward(temp);
        temp2 = temp.to(at::kCPU);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc218(tgt0);
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
        temp = fc219.forward(temp);
        temp1 = temp.to(at::kCPU);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src.to(at::kCUDA);
        temp = fc220.forward(temp);
        temp2 = temp.to(at::kCPU);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc221(src);
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
        
        temp = fc222(tgt);
        temp = F::relu(temp);
        temp = temp.to(at::kCUDA);
        temp = fc223.forward(temp);
        temp = temp.to(at::kCPU);
        tgt = tgt + temp;


        //decoder-layer-13


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp1 = fc224(tgt0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc225(tgt0);
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
        temp2 = fc228(src);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src.to(at::kCUDA);
        temp = fc229.forward(temp);
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
        temp = fc230.forward(temp);
        temp = temp.to(at::kCPU);
        temp = F::relu(temp);
        temp = fc231(temp);
        tgt = tgt + temp;


        //decoder-layer-14


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp1 = fc232(tgt0);
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
        temp2 = fc236(src);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc237(src);
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
        temp = relu.forward(temp);
        temp = temp.to(at::kCPU);
        temp = fc239(temp);
        tgt = tgt + temp;


        //decoder-layer-15


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp1 = fc240(tgt0);
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
        
        temp = fc246(tgt);
        temp = temp.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = fc247.forward(temp);
        temp = temp.to(at::kCPU);
        tgt = tgt + temp;


        //decoder-layer-16


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp1 = fc248(tgt0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc249(tgt0);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = tgt0.to(at::kCUDA);
        temp = fc250.forward(temp);
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
        temp = fc251.forward(temp);
        temp1 = temp.to(at::kCPU);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src.to(at::kCUDA);
        temp = fc252.forward(temp);
        temp2 = temp.to(at::kCPU);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src.to(at::kCUDA);
        temp = fc253.forward(temp);
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
        temp = fc254.forward(temp);
        temp = relu.forward(temp);
        temp = temp.to(at::kCPU);
        temp = fc255(temp);
        tgt = tgt + temp;


        //decoder-layer-17


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp1 = fc256(tgt0);
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
        temp1 = fc259(tgt0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc260(src);
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
        temp = temp.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = fc263.forward(temp);
        temp = temp.to(at::kCPU);
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
        temp = src.to(at::kCUDA);
        temp = fc268.forward(temp);
        temp2 = temp.to(at::kCPU);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src.to(at::kCUDA);
        temp = fc269.forward(temp);
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
        
        temp = fc270(tgt);
        temp = temp.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = temp.to(at::kCPU);
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
        temp3 = fc277(src);
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
        
        temp = fc278(tgt);
        temp = F::relu(temp);
        temp = temp.to(at::kCUDA);
        temp = fc279.forward(temp);
        temp = temp.to(at::kCPU);
        tgt = tgt + temp;


        //decoder-layer-20


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp1 = fc280(tgt0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc281(tgt0);
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
        temp1 = fc283(tgt0);
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
        temp1 = fc288(tgt0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = tgt0.to(at::kCUDA);
        temp = fc289.forward(temp);
        temp2 = temp.to(at::kCPU);
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
        temp2 = fc292(src);
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
        
        temp = tgt.to(at::kCUDA);
        temp = fc294.forward(temp);
        temp = temp.to(at::kCPU);
        temp = F::relu(temp);
        temp = fc295(temp);
        tgt = tgt + temp;


        //decoder-layer-22


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp1 = fc296(tgt0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc297(tgt0);
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
        temp = tgt0.to(at::kCUDA);
        temp = fc299.forward(temp);
        temp1 = temp.to(at::kCPU);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc300(src);
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
        
        temp = tgt.to(at::kCUDA);
        temp = fc302.forward(temp);
        temp = temp.to(at::kCPU);
        temp = F::relu(temp);
        temp = fc303(temp);
        tgt = tgt + temp;


        //decoder-layer-23


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = tgt0.to(at::kCUDA);
        temp = fc304.forward(temp);
        temp1 = temp.to(at::kCPU);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc305(tgt0);
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
        temp = src.to(at::kCUDA);
        temp = fc308.forward(temp);
        temp2 = temp.to(at::kCPU);
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
        
        temp = fc310(tgt);
        temp = temp.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = temp.to(at::kCPU);
        temp = fc311(temp);
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
    std::cout << "Transformer - 40% outsourced to CUDA version" << std::endl;
    std::cout << "Kernel Switch: 210" << std::endl;
    std::cout << "Associative Op: 360" << std::endl;
    std::cout << "Outsourced Op: 142" << std::endl;

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