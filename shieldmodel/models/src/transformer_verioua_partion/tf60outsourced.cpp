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
    torch::nn::Linear fc3;
    torch::nn::Linear fc4;
    operator7 fc5;
    torch::nn::Linear fc6;
    operator7 fc7;
    torch::nn::Linear fc8;
    torch::nn::Linear fc9;
    torch::nn::Linear fc10;
    operator7 fc11;
    operator7 fc12;
    torch::nn::Linear fc13;
    operator7 fc14;
    operator7 fc15;
    torch::nn::Linear fc16;
    operator7 fc17;
    operator7 fc18;
    operator7 fc19;
    torch::nn::Linear fc20;
    operator7 fc21;
    operator7 fc22;
    operator7 fc23;
    torch::nn::Linear fc24;
    operator7 fc25;
    operator7 fc26;
    torch::nn::Linear fc27;
    torch::nn::Linear fc28;
    operator7 fc29;
    torch::nn::Linear fc30;
    torch::nn::Linear fc31;
    operator7 fc32;
    operator7 fc33;
    operator7 fc34;
    operator7 fc35;
    operator7 fc36;
    operator7 fc37;
    operator7 fc38;
    operator7 fc39;
    operator7 fc40;
    operator7 fc41;
    operator7 fc42;
    torch::nn::Linear fc43;
    operator7 fc44;
    operator7 fc45;
    operator7 fc46;
    operator7 fc47;
    torch::nn::Linear fc48;
    operator7 fc49;
    torch::nn::Linear fc50;
    operator7 fc51;
    operator7 fc52;
    torch::nn::Linear fc53;
    operator7 fc54;
    operator7 fc55;
    torch::nn::Linear fc56;
    operator7 fc57;
    operator7 fc58;
    operator7 fc59;
    operator7 fc60;
    operator7 fc61;
    operator7 fc62;
    torch::nn::Linear fc63;
    torch::nn::Linear fc64;
    operator7 fc65;
    torch::nn::Linear fc66;
    operator7 fc67;
    torch::nn::Linear fc68;
    operator7 fc69;
    operator7 fc70;
    operator7 fc71;
    operator7 fc72;
    torch::nn::Linear fc73;
    torch::nn::Linear fc74;
    torch::nn::Linear fc75;
    operator7 fc76;
    torch::nn::Linear fc77;
    operator7 fc78;
    operator7 fc79;
    torch::nn::Linear fc80;
    torch::nn::Linear fc81;
    operator7 fc82;
    torch::nn::Linear fc83;
    torch::nn::Linear fc84;
    operator7 fc85;
    operator7 fc86;
    operator7 fc87;
    torch::nn::Linear fc88;
    operator7 fc89;
    operator7 fc90;
    operator7 fc91;
    operator7 fc92;
    operator7 fc93;
    operator7 fc94;
    torch::nn::Linear fc95;
    operator7 fc96;
    torch::nn::Linear fc97;
    torch::nn::Linear fc98;
    operator7 fc99;
    operator7 fc100;
    torch::nn::Linear fc101;
    operator7 fc102;
    operator7 fc103;
    torch::nn::Linear fc104;
    torch::nn::Linear fc105;
    operator7 fc106;
    operator7 fc107;
    operator7 fc108;
    operator7 fc109;
    operator7 fc110;
    torch::nn::Linear fc111;
    operator7 fc112;
    operator7 fc113;
    operator7 fc114;
    operator7 fc115;
    torch::nn::Linear fc116;
    torch::nn::Linear fc117;
    operator7 fc118;
    torch::nn::Linear fc119;
    operator7 fc120;
    torch::nn::Linear fc121;
    torch::nn::Linear fc122;
    torch::nn::Linear fc123;
    operator7 fc124;
    torch::nn::Linear fc125;
    torch::nn::Linear fc126;
    operator7 fc127;
    operator7 fc128;
    operator7 fc129;
    torch::nn::Linear fc130;
    operator7 fc131;
    torch::nn::Linear fc132;
    operator7 fc133;
    torch::nn::Linear fc134;
    operator7 fc135;
    torch::nn::Linear fc136;
    torch::nn::Linear fc137;
    operator7 fc138;
    operator7 fc139;
    operator7 fc140;
    torch::nn::Linear fc141;
    torch::nn::Linear fc142;
    operator7 fc143;
    operator7 fc144;
    operator7 fc145;
    operator7 fc146;
    operator7 fc147;
    operator7 fc148;
    operator7 fc149;
    torch::nn::Linear fc150;
    operator7 fc151;
    operator7 fc152;
    operator7 fc153;
    operator7 fc154;
    operator7 fc155;
    operator7 fc156;
    torch::nn::Linear fc157;
    torch::nn::Linear fc158;
    torch::nn::Linear fc159;
    operator7 fc160;
    operator7 fc161;
    torch::nn::Linear fc162;
    operator7 fc163;
    operator7 fc164;
    operator7 fc165;
    operator7 fc166;
    operator7 fc167;
    operator7 fc168;
    operator7 fc169;
    torch::nn::Linear fc170;
    torch::nn::Linear fc171;
    operator7 fc172;
    operator7 fc173;
    operator7 fc174;
    torch::nn::Linear fc175;
    operator7 fc176;
    operator7 fc177;
    operator7 fc178;
    operator7 fc179;
    operator7 fc180;
    torch::nn::Linear fc181;
    torch::nn::Linear fc182;
    operator7 fc183;
    operator7 fc184;
    operator7 fc185;
    operator7 fc186;
    operator7 fc187;
    operator7 fc188;
    torch::nn::Linear fc189;
    operator7 fc190;
    torch::nn::Linear fc191;
    operator7 fc192;
    operator7 fc193;
    torch::nn::Linear fc194;
    operator7 fc195;
    torch::nn::Linear fc196;
    torch::nn::Linear fc197;
    operator7 fc198;
    operator7 fc199;
    torch::nn::Linear fc200;
    operator7 fc201;
    operator7 fc202;
    operator7 fc203;
    torch::nn::Linear fc204;
    operator7 fc205;
    torch::nn::Linear fc206;
    operator7 fc207;
    torch::nn::Linear fc208;
    torch::nn::Linear fc209;
    operator7 fc210;
    operator7 fc211;
    torch::nn::Linear fc212;
    torch::nn::Linear fc213;
    operator7 fc214;
    torch::nn::Linear fc215;
    operator7 fc216;
    operator7 fc217;
    torch::nn::Linear fc218;
    operator7 fc219;
    operator7 fc220;
    torch::nn::Linear fc221;
    operator7 fc222;
    torch::nn::Linear fc223;
    torch::nn::Linear fc224;
    operator7 fc225;
    torch::nn::Linear fc226;
    torch::nn::Linear fc227;
    operator7 fc228;
    operator7 fc229;
    torch::nn::Linear fc230;
    torch::nn::Linear fc231;
    operator7 fc232;
    operator7 fc233;
    torch::nn::Linear fc234;
    operator7 fc235;
    operator7 fc236;
    torch::nn::Linear fc237;
    torch::nn::Linear fc238;
    operator7 fc239;
    operator7 fc240;
    operator7 fc241;
    operator7 fc242;
    torch::nn::Linear fc243;
    operator7 fc244;
    operator7 fc245;
    torch::nn::Linear fc246;
    operator7 fc247;
    operator7 fc248;
    operator7 fc249;
    operator7 fc250;
    torch::nn::Linear fc251;
    torch::nn::Linear fc252;
    torch::nn::Linear fc253;
    torch::nn::Linear fc254;
    torch::nn::Linear fc255;
    torch::nn::Linear fc256;
    operator7 fc257;
    torch::nn::Linear fc258;
    torch::nn::Linear fc259;
    operator7 fc260;
    operator7 fc261;
    torch::nn::Linear fc262;
    operator7 fc263;
    operator7 fc264;
    operator7 fc265;
    operator7 fc266;
    operator7 fc267;
    operator7 fc268;
    operator7 fc269;
    torch::nn::Linear fc270;
    operator7 fc271;
    torch::nn::Linear fc272;
    torch::nn::Linear fc273;
    torch::nn::Linear fc274;
    operator7 fc275;
    operator7 fc276;
    operator7 fc277;
    torch::nn::Linear fc278;
    operator7 fc279;
    operator7 fc280;
    torch::nn::Linear fc281;
    operator7 fc282;
    operator7 fc283;
    operator7 fc284;
    torch::nn::Linear fc285;
    torch::nn::Linear fc286;
    torch::nn::Linear fc287;
    torch::nn::Linear fc288;
    operator7 fc289;
    operator7 fc290;
    torch::nn::Linear fc291;
    operator7 fc292;
    operator7 fc293;
    operator7 fc294;
    torch::nn::Linear fc295;
    operator7 fc296;
    operator7 fc297;
    operator7 fc298;
    operator7 fc299;
    torch::nn::Linear fc300;
    torch::nn::Linear fc301;
    operator7 fc302;
    torch::nn::Linear fc303;
    operator7 fc304;
    torch::nn::Linear fc305;
    operator7 fc306;
    operator7 fc307;
    torch::nn::Linear fc308;
    operator7 fc309;
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
        register_module("fc3", fc3);
        register_module("fc4", fc4);
        fc5.to(at::kCUDA);
        register_module("fc6", fc6);
        fc7.to(at::kCUDA);
        register_module("fc8", fc8);
        register_module("fc9", fc9);
        register_module("fc10", fc10);
        fc11.to(at::kCUDA);
        fc12.to(at::kCUDA);
        register_module("fc13", fc13);
        fc14.to(at::kCUDA);
        fc15.to(at::kCUDA);
        register_module("fc16", fc16);
        fc17.to(at::kCUDA);
        fc18.to(at::kCUDA);
        fc19.to(at::kCUDA);
        register_module("fc20", fc20);
        fc21.to(at::kCUDA);
        fc22.to(at::kCUDA);
        fc23.to(at::kCUDA);
        register_module("fc24", fc24);
        fc25.to(at::kCUDA);
        fc26.to(at::kCUDA);
        register_module("fc27", fc27);
        register_module("fc28", fc28);
        fc29.to(at::kCUDA);
        register_module("fc30", fc30);
        register_module("fc31", fc31);
        fc32.to(at::kCUDA);
        fc33.to(at::kCUDA);
        fc34.to(at::kCUDA);
        fc35.to(at::kCUDA);
        fc36.to(at::kCUDA);
        fc37.to(at::kCUDA);
        fc38.to(at::kCUDA);
        fc39.to(at::kCUDA);
        fc40.to(at::kCUDA);
        fc41.to(at::kCUDA);
        fc42.to(at::kCUDA);
        register_module("fc43", fc43);
        fc44.to(at::kCUDA);
        fc45.to(at::kCUDA);
        fc46.to(at::kCUDA);
        fc47.to(at::kCUDA);
        register_module("fc48", fc48);
        fc49.to(at::kCUDA);
        register_module("fc50", fc50);
        fc51.to(at::kCUDA);
        fc52.to(at::kCUDA);
        register_module("fc53", fc53);
        fc54.to(at::kCUDA);
        fc55.to(at::kCUDA);
        register_module("fc56", fc56);
        fc57.to(at::kCUDA);
        fc58.to(at::kCUDA);
        fc59.to(at::kCUDA);
        fc60.to(at::kCUDA);
        fc61.to(at::kCUDA);
        fc62.to(at::kCUDA);
        register_module("fc63", fc63);
        register_module("fc64", fc64);
        fc65.to(at::kCUDA);
        register_module("fc66", fc66);
        fc67.to(at::kCUDA);
        register_module("fc68", fc68);
        fc69.to(at::kCUDA);
        fc70.to(at::kCUDA);
        fc71.to(at::kCUDA);
        fc72.to(at::kCUDA);
        register_module("fc73", fc73);
        register_module("fc74", fc74);
        register_module("fc75", fc75);
        fc76.to(at::kCUDA);
        register_module("fc77", fc77);
        fc78.to(at::kCUDA);
        fc79.to(at::kCUDA);
        register_module("fc80", fc80);
        register_module("fc81", fc81);
        fc82.to(at::kCUDA);
        register_module("fc83", fc83);
        register_module("fc84", fc84);
        fc85.to(at::kCUDA);
        fc86.to(at::kCUDA);
        fc87.to(at::kCUDA);
        register_module("fc88", fc88);
        fc89.to(at::kCUDA);
        fc90.to(at::kCUDA);
        fc91.to(at::kCUDA);
        fc92.to(at::kCUDA);
        fc93.to(at::kCUDA);
        fc94.to(at::kCUDA);
        register_module("fc95", fc95);
        fc96.to(at::kCUDA);
        register_module("fc97", fc97);
        register_module("fc98", fc98);
        fc99.to(at::kCUDA);
        fc100.to(at::kCUDA);
        register_module("fc101", fc101);
        fc102.to(at::kCUDA);
        fc103.to(at::kCUDA);
        register_module("fc104", fc104);
        register_module("fc105", fc105);
        fc106.to(at::kCUDA);
        fc107.to(at::kCUDA);
        fc108.to(at::kCUDA);
        fc109.to(at::kCUDA);
        fc110.to(at::kCUDA);
        register_module("fc111", fc111);
        fc112.to(at::kCUDA);
        fc113.to(at::kCUDA);
        fc114.to(at::kCUDA);
        fc115.to(at::kCUDA);
        register_module("fc116", fc116);
        register_module("fc117", fc117);
        fc118.to(at::kCUDA);
        register_module("fc119", fc119);
        fc120.to(at::kCUDA);
        register_module("fc121", fc121);
        register_module("fc122", fc122);
        register_module("fc123", fc123);
        fc124.to(at::kCUDA);
        register_module("fc125", fc125);
        register_module("fc126", fc126);
        fc127.to(at::kCUDA);
        fc128.to(at::kCUDA);
        fc129.to(at::kCUDA);
        register_module("fc130", fc130);
        fc131.to(at::kCUDA);
        register_module("fc132", fc132);
        fc133.to(at::kCUDA);
        register_module("fc134", fc134);
        fc135.to(at::kCUDA);
        register_module("fc136", fc136);
        register_module("fc137", fc137);
        fc138.to(at::kCUDA);
        fc139.to(at::kCUDA);
        fc140.to(at::kCUDA);
        register_module("fc141", fc141);
        register_module("fc142", fc142);
        fc143.to(at::kCUDA);
        fc144.to(at::kCUDA);
        fc145.to(at::kCUDA);
        fc146.to(at::kCUDA);
        fc147.to(at::kCUDA);
        fc148.to(at::kCUDA);
        fc149.to(at::kCUDA);
        register_module("fc150", fc150);
        fc151.to(at::kCUDA);
        fc152.to(at::kCUDA);
        fc153.to(at::kCUDA);
        fc154.to(at::kCUDA);
        fc155.to(at::kCUDA);
        fc156.to(at::kCUDA);
        register_module("fc157", fc157);
        register_module("fc158", fc158);
        register_module("fc159", fc159);
        fc160.to(at::kCUDA);
        fc161.to(at::kCUDA);
        register_module("fc162", fc162);
        fc163.to(at::kCUDA);
        fc164.to(at::kCUDA);
        fc165.to(at::kCUDA);
        fc166.to(at::kCUDA);
        fc167.to(at::kCUDA);
        fc168.to(at::kCUDA);
        fc169.to(at::kCUDA);
        register_module("fc170", fc170);
        register_module("fc171", fc171);
        fc172.to(at::kCUDA);
        fc173.to(at::kCUDA);
        fc174.to(at::kCUDA);
        register_module("fc175", fc175);
        fc176.to(at::kCUDA);
        fc177.to(at::kCUDA);
        fc178.to(at::kCUDA);
        fc179.to(at::kCUDA);
        fc180.to(at::kCUDA);
        register_module("fc181", fc181);
        register_module("fc182", fc182);
        fc183.to(at::kCUDA);
        fc184.to(at::kCUDA);
        fc185.to(at::kCUDA);
        fc186.to(at::kCUDA);
        fc187.to(at::kCUDA);
        fc188.to(at::kCUDA);
        register_module("fc189", fc189);
        fc190.to(at::kCUDA);
        register_module("fc191", fc191);
        fc192.to(at::kCUDA);
        fc193.to(at::kCUDA);
        register_module("fc194", fc194);
        fc195.to(at::kCUDA);
        register_module("fc196", fc196);
        register_module("fc197", fc197);
        fc198.to(at::kCUDA);
        fc199.to(at::kCUDA);
        register_module("fc200", fc200);
        fc201.to(at::kCUDA);
        fc202.to(at::kCUDA);
        fc203.to(at::kCUDA);
        register_module("fc204", fc204);
        fc205.to(at::kCUDA);
        register_module("fc206", fc206);
        fc207.to(at::kCUDA);
        register_module("fc208", fc208);
        register_module("fc209", fc209);
        fc210.to(at::kCUDA);
        fc211.to(at::kCUDA);
        register_module("fc212", fc212);
        register_module("fc213", fc213);
        fc214.to(at::kCUDA);
        register_module("fc215", fc215);
        fc216.to(at::kCUDA);
        fc217.to(at::kCUDA);
        register_module("fc218", fc218);
        fc219.to(at::kCUDA);
        fc220.to(at::kCUDA);
        register_module("fc221", fc221);
        fc222.to(at::kCUDA);
        register_module("fc223", fc223);
        register_module("fc224", fc224);
        fc225.to(at::kCUDA);
        register_module("fc226", fc226);
        register_module("fc227", fc227);
        fc228.to(at::kCUDA);
        fc229.to(at::kCUDA);
        register_module("fc230", fc230);
        register_module("fc231", fc231);
        fc232.to(at::kCUDA);
        fc233.to(at::kCUDA);
        register_module("fc234", fc234);
        fc235.to(at::kCUDA);
        fc236.to(at::kCUDA);
        register_module("fc237", fc237);
        register_module("fc238", fc238);
        fc239.to(at::kCUDA);
        fc240.to(at::kCUDA);
        fc241.to(at::kCUDA);
        fc242.to(at::kCUDA);
        register_module("fc243", fc243);
        fc244.to(at::kCUDA);
        fc245.to(at::kCUDA);
        register_module("fc246", fc246);
        fc247.to(at::kCUDA);
        fc248.to(at::kCUDA);
        fc249.to(at::kCUDA);
        fc250.to(at::kCUDA);
        register_module("fc251", fc251);
        register_module("fc252", fc252);
        register_module("fc253", fc253);
        register_module("fc254", fc254);
        register_module("fc255", fc255);
        register_module("fc256", fc256);
        fc257.to(at::kCUDA);
        register_module("fc258", fc258);
        register_module("fc259", fc259);
        fc260.to(at::kCUDA);
        fc261.to(at::kCUDA);
        register_module("fc262", fc262);
        fc263.to(at::kCUDA);
        fc264.to(at::kCUDA);
        fc265.to(at::kCUDA);
        fc266.to(at::kCUDA);
        fc267.to(at::kCUDA);
        fc268.to(at::kCUDA);
        fc269.to(at::kCUDA);
        register_module("fc270", fc270);
        fc271.to(at::kCUDA);
        register_module("fc272", fc272);
        register_module("fc273", fc273);
        register_module("fc274", fc274);
        fc275.to(at::kCUDA);
        fc276.to(at::kCUDA);
        fc277.to(at::kCUDA);
        register_module("fc278", fc278);
        fc279.to(at::kCUDA);
        fc280.to(at::kCUDA);
        register_module("fc281", fc281);
        fc282.to(at::kCUDA);
        fc283.to(at::kCUDA);
        fc284.to(at::kCUDA);
        register_module("fc285", fc285);
        register_module("fc286", fc286);
        register_module("fc287", fc287);
        register_module("fc288", fc288);
        fc289.to(at::kCUDA);
        fc290.to(at::kCUDA);
        register_module("fc291", fc291);
        fc292.to(at::kCUDA);
        fc293.to(at::kCUDA);
        fc294.to(at::kCUDA);
        register_module("fc295", fc295);
        fc296.to(at::kCUDA);
        fc297.to(at::kCUDA);
        fc298.to(at::kCUDA);
        fc299.to(at::kCUDA);
        register_module("fc300", fc300);
        register_module("fc301", fc301);
        fc302.to(at::kCUDA);
        register_module("fc303", fc303);
        fc304.to(at::kCUDA);
        register_module("fc305", fc305);
        fc306.to(at::kCUDA);
        fc307.to(at::kCUDA);
        register_module("fc308", fc308);
        fc309.to(at::kCUDA);
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
        temp = fc3(src);
        temp = F::relu(temp);
        temp = fc4(temp);
        src = src + temp;


        //encoder-layer-1


        src0 = src;
        nbatches = src0.size(0);
        temp = src0.to(at::kCUDA);
        temp = fc5.forward(temp);
        temp1 = temp.to(at::kCPU);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc6(src0);
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
        temp = fc9(temp);
        src = src + temp;


        //encoder-layer-2


        src0 = src;
        nbatches = src0.size(0);
        temp1 = fc10(src0);
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
        temp = src0.to(at::kCUDA);
        temp = fc15.forward(temp);
        temp1 = temp.to(at::kCPU);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc16(src0);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src0.to(at::kCUDA);
        temp = fc17.forward(temp);
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
        temp = fc18.forward(temp);
        temp = relu.forward(temp);
        temp = fc19.forward(temp);
        temp = temp.to(at::kCPU);
        src = src + temp;


        //encoder-layer-4


        src0 = src;
        nbatches = src0.size(0);
        temp1 = fc20(src0);
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
        temp = src0.to(at::kCUDA);
        temp = fc26.forward(temp);
        temp2 = temp.to(at::kCPU);
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
        temp = F::relu(temp);
        temp = temp.to(at::kCUDA);
        temp = fc29.forward(temp);
        temp = temp.to(at::kCPU);
        src = src + temp;


        //encoder-layer-6


        src0 = src;
        nbatches = src0.size(0);
        temp1 = fc30(src0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc31(src0);
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
        temp = src.to(at::kCUDA);
        temp = fc33.forward(temp);
        temp = relu.forward(temp);
        temp = fc34.forward(temp);
        temp = temp.to(at::kCPU);
        src = src + temp;


        //encoder-layer-7


        src0 = src;
        nbatches = src0.size(0);
        temp = src0.to(at::kCUDA);
        temp = fc35.forward(temp);
        temp1 = temp.to(at::kCPU);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src0.to(at::kCUDA);
        temp = fc36.forward(temp);
        temp2 = temp.to(at::kCPU);
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
        temp = src.to(at::kCUDA);
        temp = fc38.forward(temp);
        temp = temp.to(at::kCPU);
        temp = F::relu(temp);
        temp = temp.to(at::kCUDA);
        temp = fc39.forward(temp);
        temp = temp.to(at::kCPU);
        src = src + temp;


        //encoder-layer-8


        src0 = src;
        nbatches = src0.size(0);
        temp = src0.to(at::kCUDA);
        temp = fc40.forward(temp);
        temp1 = temp.to(at::kCPU);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src0.to(at::kCUDA);
        temp = fc41.forward(temp);
        temp2 = temp.to(at::kCPU);
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
        temp = fc48(src);
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
        temp = src0.to(at::kCUDA);
        temp = fc51.forward(temp);
        temp2 = temp.to(at::kCPU);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src0.to(at::kCUDA);
        temp = fc52.forward(temp);
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
        temp = fc53(src);
        temp = temp.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = fc54.forward(temp);
        temp = temp.to(at::kCPU);
        src = src + temp;


        //encoder-layer-11


        src0 = src;
        nbatches = src0.size(0);
        temp = src0.to(at::kCUDA);
        temp = fc55.forward(temp);
        temp1 = temp.to(at::kCPU);
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
        temp = src.to(at::kCUDA);
        temp = fc58.forward(temp);
        temp = temp.to(at::kCPU);
        temp = F::relu(temp);
        temp = temp.to(at::kCUDA);
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
        temp = src0.to(at::kCUDA);
        temp = fc61.forward(temp);
        temp2 = temp.to(at::kCPU);
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
        temp = fc64(temp);
        src = src + temp;


        //encoder-layer-13


        src0 = src;
        nbatches = src0.size(0);
        temp = src0.to(at::kCUDA);
        temp = fc65.forward(temp);
        temp1 = temp.to(at::kCPU);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc66(src0);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src0.to(at::kCUDA);
        temp = fc67.forward(temp);
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
        temp = fc68(src);
        temp = temp.to(at::kCUDA);
        temp = relu.forward(temp);
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
        temp = src0.to(at::kCUDA);
        temp = fc72.forward(temp);
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
        temp = fc73(src);
        temp = F::relu(temp);
        temp = fc74(temp);
        src = src + temp;


        //encoder-layer-15


        src0 = src;
        nbatches = src0.size(0);
        temp1 = fc75(src0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src0.to(at::kCUDA);
        temp = fc76.forward(temp);
        temp2 = temp.to(at::kCPU);
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
        temp = src.to(at::kCUDA);
        temp = fc78.forward(temp);
        temp = temp.to(at::kCPU);
        temp = F::relu(temp);
        temp = temp.to(at::kCUDA);
        temp = fc79.forward(temp);
        temp = temp.to(at::kCPU);
        src = src + temp;


        //encoder-layer-16


        src0 = src;
        nbatches = src0.size(0);
        temp1 = fc80(src0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc81(src0);
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
        temp = fc83(src);
        temp = temp.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = temp.to(at::kCPU);
        temp = fc84(temp);
        src = src + temp;


        //encoder-layer-17


        src0 = src;
        nbatches = src0.size(0);
        temp = src0.to(at::kCUDA);
        temp = fc85.forward(temp);
        temp1 = temp.to(at::kCPU);
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
        temp = src0.to(at::kCUDA);
        temp = fc90.forward(temp);
        temp1 = temp.to(at::kCPU);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src0.to(at::kCUDA);
        temp = fc91.forward(temp);
        temp2 = temp.to(at::kCPU);
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
        temp = src.to(at::kCUDA);
        temp = fc93.forward(temp);
        temp = relu.forward(temp);
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
        temp = temp.to(at::kCUDA);
        temp = fc99.forward(temp);
        temp = temp.to(at::kCPU);
        src = src + temp;


        //encoder-layer-20


        src0 = src;
        nbatches = src0.size(0);
        temp = src0.to(at::kCUDA);
        temp = fc100.forward(temp);
        temp1 = temp.to(at::kCPU);
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
        temp = src.to(at::kCUDA);
        temp = fc103.forward(temp);
        temp = temp.to(at::kCPU);
        temp = F::relu(temp);
        temp = fc104(temp);
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
        temp = src0.to(at::kCUDA);
        temp = fc107.forward(temp);
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
        temp = fc108.forward(temp);
        temp = relu.forward(temp);
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
        temp2 = fc111(src0);
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
        temp = relu.forward(temp);
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
        temp = src.to(at::kCUDA);
        temp = fc118.forward(temp);
        temp = temp.to(at::kCPU);
        temp = F::relu(temp);
        temp = fc119(temp);
        src = src + temp;


        //decoder-layer-0


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = tgt0.to(at::kCUDA);
        temp = fc120.forward(temp);
        temp1 = temp.to(at::kCPU);
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
        temp1 = fc123(tgt0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src.to(at::kCUDA);
        temp = fc124.forward(temp);
        temp2 = temp.to(at::kCPU);
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
        
        temp = fc126(tgt);
        temp = temp.to(at::kCUDA);
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
        temp2 = fc132(src);
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
        
        temp = fc134(tgt);
        temp = temp.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = fc135.forward(temp);
        temp = temp.to(at::kCPU);
        tgt = tgt + temp;


        //decoder-layer-2


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp1 = fc136(tgt0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc137(tgt0);
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
        temp = tgt0.to(at::kCUDA);
        temp = fc139.forward(temp);
        temp1 = temp.to(at::kCPU);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src.to(at::kCUDA);
        temp = fc140.forward(temp);
        temp2 = temp.to(at::kCPU);
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
        
        temp = fc142(tgt);
        temp = F::relu(temp);
        temp = temp.to(at::kCUDA);
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
        temp = tgt0.to(at::kCUDA);
        temp = fc147.forward(temp);
        temp1 = temp.to(at::kCPU);
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
        
        temp = fc150(tgt);
        temp = temp.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = fc151.forward(temp);
        temp = temp.to(at::kCPU);
        tgt = tgt + temp;


        //decoder-layer-4


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = tgt0.to(at::kCUDA);
        temp = fc152.forward(temp);
        temp1 = temp.to(at::kCPU);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = tgt0.to(at::kCUDA);
        temp = fc153.forward(temp);
        temp2 = temp.to(at::kCPU);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = tgt0.to(at::kCUDA);
        temp = fc154.forward(temp);
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
        temp = temp.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = temp.to(at::kCPU);
        temp = fc159(temp);
        tgt = tgt + temp;


        //decoder-layer-5


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = tgt0.to(at::kCUDA);
        temp = fc160.forward(temp);
        temp1 = temp.to(at::kCPU);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = tgt0.to(at::kCUDA);
        temp = fc161.forward(temp);
        temp2 = temp.to(at::kCPU);
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
        temp = tgt0.to(at::kCUDA);
        temp = fc163.forward(temp);
        temp1 = temp.to(at::kCPU);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src.to(at::kCUDA);
        temp = fc164.forward(temp);
        temp2 = temp.to(at::kCPU);
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
        
        temp = tgt.to(at::kCUDA);
        temp = fc174.forward(temp);
        temp = temp.to(at::kCPU);
        temp = F::relu(temp);
        temp = fc175(temp);
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
        temp = tgt0.to(at::kCUDA);
        temp = fc178.forward(temp);
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
        
        temp = fc182(tgt);
        temp = temp.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = fc183.forward(temp);
        temp = temp.to(at::kCPU);
        tgt = tgt + temp;


        //decoder-layer-8


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = tgt0.to(at::kCUDA);
        temp = fc184.forward(temp);
        temp1 = temp.to(at::kCPU);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = tgt0.to(at::kCUDA);
        temp = fc185.forward(temp);
        temp2 = temp.to(at::kCPU);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = tgt0.to(at::kCUDA);
        temp = fc186.forward(temp);
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
        temp = fc187.forward(temp);
        temp1 = temp.to(at::kCPU);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src.to(at::kCUDA);
        temp = fc188.forward(temp);
        temp2 = temp.to(at::kCPU);
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
        
        temp = tgt.to(at::kCUDA);
        temp = fc190.forward(temp);
        temp = relu.forward(temp);
        temp = temp.to(at::kCPU);
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
        temp = tgt0.to(at::kCUDA);
        temp = fc195.forward(temp);
        temp1 = temp.to(at::kCPU);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc196(src);
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
        
        temp = tgt.to(at::kCUDA);
        temp = fc198.forward(temp);
        temp = temp.to(at::kCPU);
        temp = F::relu(temp);
        temp = temp.to(at::kCUDA);
        temp = fc199.forward(temp);
        temp = temp.to(at::kCPU);
        tgt = tgt + temp;


        //decoder-layer-10


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp1 = fc200(tgt0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = tgt0.to(at::kCUDA);
        temp = fc201.forward(temp);
        temp2 = temp.to(at::kCPU);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = tgt0.to(at::kCUDA);
        temp = fc202.forward(temp);
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
        temp = fc203.forward(temp);
        temp1 = temp.to(at::kCPU);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc204(src);
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
        temp = F::relu(temp);
        temp = temp.to(at::kCUDA);
        temp = fc207.forward(temp);
        temp = temp.to(at::kCPU);
        tgt = tgt + temp;


        //decoder-layer-11


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp1 = fc208(tgt0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc209(tgt0);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = tgt0.to(at::kCUDA);
        temp = fc210.forward(temp);
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
        temp = fc211.forward(temp);
        temp1 = temp.to(at::kCPU);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc212(src);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc213(src);
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
        
        temp = tgt.to(at::kCUDA);
        temp = fc222.forward(temp);
        temp = relu.forward(temp);
        temp = temp.to(at::kCPU);
        temp = fc223(temp);
        tgt = tgt + temp;


        //decoder-layer-13


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp1 = fc224(tgt0);
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
        temp1 = fc227(tgt0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src.to(at::kCUDA);
        temp = fc228.forward(temp);
        temp2 = temp.to(at::kCPU);
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
        
        temp = fc230(tgt);
        temp = F::relu(temp);
        temp = fc231(temp);
        tgt = tgt + temp;


        //decoder-layer-14


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = tgt0.to(at::kCUDA);
        temp = fc232.forward(temp);
        temp1 = temp.to(at::kCPU);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = tgt0.to(at::kCUDA);
        temp = fc233.forward(temp);
        temp2 = temp.to(at::kCPU);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc234(tgt0);
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
        temp = fc235.forward(temp);
        temp1 = temp.to(at::kCPU);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src.to(at::kCUDA);
        temp = fc236.forward(temp);
        temp2 = temp.to(at::kCPU);
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
        
        temp = fc238(tgt);
        temp = temp.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = fc239.forward(temp);
        temp = temp.to(at::kCPU);
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
        
        temp = fc246(tgt);
        temp = temp.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = fc247.forward(temp);
        temp = temp.to(at::kCPU);
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
        temp1 = fc251(tgt0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc252(src);
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
        temp = F::relu(temp);
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
        temp = src.to(at::kCUDA);
        temp = fc260.forward(temp);
        temp2 = temp.to(at::kCPU);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src.to(at::kCUDA);
        temp = fc261.forward(temp);
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
        
        temp = fc262(tgt);
        temp = F::relu(temp);
        temp = temp.to(at::kCUDA);
        temp = fc263.forward(temp);
        temp = temp.to(at::kCPU);
        tgt = tgt + temp;


        //decoder-layer-18


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp = tgt0.to(at::kCUDA);
        temp = fc264.forward(temp);
        temp1 = temp.to(at::kCPU);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = tgt0.to(at::kCUDA);
        temp = fc265.forward(temp);
        temp2 = temp.to(at::kCPU);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = tgt0.to(at::kCUDA);
        temp = fc266.forward(temp);
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
        temp = fc271.forward(temp);
        temp = temp.to(at::kCPU);
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
        temp = tgt0.to(at::kCUDA);
        temp = fc275.forward(temp);
        temp1 = temp.to(at::kCPU);
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
        
        temp = fc278(tgt);
        temp = F::relu(temp);
        temp = temp.to(at::kCUDA);
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
        temp2 = fc281(tgt0);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = tgt0.to(at::kCUDA);
        temp = fc282.forward(temp);
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
        temp = fc283.forward(temp);
        temp1 = temp.to(at::kCPU);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src.to(at::kCUDA);
        temp = fc284.forward(temp);
        temp2 = temp.to(at::kCPU);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc285(src);
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
        temp = temp.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = temp.to(at::kCPU);
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
        
        temp = tgt.to(at::kCUDA);
        temp = fc294.forward(temp);
        temp = relu.forward(temp);
        temp = temp.to(at::kCPU);
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
        temp = relu.forward(temp);
        temp = temp.to(at::kCPU);
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
        temp = tgt0.to(at::kCUDA);
        temp = fc307.forward(temp);
        temp1 = temp.to(at::kCPU);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc308(src);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp = src.to(at::kCUDA);
        temp = fc309.forward(temp);
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
    std::cout << "Transformer - 60% outsourced to CUDA version" << std::endl;
    std::cout << "Kernel Switch: 322" << std::endl;
    std::cout << "Associative Op: 360" << std::endl;
    std::cout << "Outsourced Op: 221" << std::endl;

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