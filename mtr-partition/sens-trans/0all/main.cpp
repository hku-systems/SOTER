
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
    torch::nn::Linear fc1;
    torch::nn::Linear fc2;
    torch::nn::Linear fc3;
    torch::nn::Linear fc4;
    torch::nn::Linear fc5;
    torch::nn::Linear fc6;
    torch::nn::Linear fc7;
    torch::nn::Linear fc8;
    torch::nn::Linear fc9;
    torch::nn::Linear fc10;
    torch::nn::Linear fc11;
    torch::nn::Linear fc12;
    torch::nn::Linear fc13;
    torch::nn::Linear fc14;
    torch::nn::Linear fc15;
    torch::nn::Linear fc16;
    torch::nn::Linear fc17;
    torch::nn::Linear fc18;
    torch::nn::Linear fc19;
    torch::nn::Linear fc20;
    torch::nn::Linear fc21;
    torch::nn::Linear fc22;
    torch::nn::Linear fc23;
    torch::nn::Linear fc24;
    torch::nn::Linear fc25;
    torch::nn::Linear fc26;
    torch::nn::Linear fc27;
    torch::nn::Linear fc28;
    torch::nn::Linear fc29;
    torch::nn::Linear fc30;
    torch::nn::Linear fc31;
    torch::nn::Linear fc32;
    torch::nn::Linear fc33;
    torch::nn::Linear fc34;
    torch::nn::Linear fc35;
    torch::nn::Linear fc36;
    torch::nn::Linear fc37;
    torch::nn::Linear fc38;
    torch::nn::Linear fc39;
    torch::nn::Linear fc40;
    torch::nn::Linear fc41;
    torch::nn::Linear fc42;
    torch::nn::Linear fc43;
    torch::nn::Linear fc44;
    torch::nn::Linear fc45;
    torch::nn::Linear fc46;
    torch::nn::Linear fc47;
    torch::nn::Linear fc48;
    torch::nn::Linear fc49;
    torch::nn::Linear fc50;
    torch::nn::Linear fc51;
    torch::nn::Linear fc52;
    torch::nn::Linear fc53;
    torch::nn::Linear fc54;
    torch::nn::Linear fc55;
    torch::nn::Linear fc56;
    torch::nn::Linear fc57;
    torch::nn::Linear fc58;
    torch::nn::Linear fc59;
    torch::nn::Linear fc60;
    torch::nn::Linear fc61;
    torch::nn::Linear fc62;
    torch::nn::Linear fc63;
    torch::nn::Linear fc64;
    torch::nn::Linear fc65;
    torch::nn::Linear fc66;
    torch::nn::Linear fc67;
    torch::nn::Linear fc68;
    torch::nn::Linear fc69;
    torch::nn::Linear fc70;
    torch::nn::Linear fc71;
    torch::nn::Linear fc72;
    torch::nn::Linear fc73;
    torch::nn::Linear fc74;
    torch::nn::Linear fc75;
    torch::nn::Linear fc76;
    torch::nn::Linear fc77;
    torch::nn::Linear fc78;
    torch::nn::Linear fc79;
    torch::nn::Linear fc80;
    torch::nn::Linear fc81;
    torch::nn::Linear fc82;
    torch::nn::Linear fc83;
    torch::nn::Linear fc84;
    torch::nn::Linear fc85;
    torch::nn::Linear fc86;
    torch::nn::Linear fc87;
    torch::nn::Linear fc88;
    torch::nn::Linear fc89;
    torch::nn::Linear fc90;
    torch::nn::Linear fc91;
    torch::nn::Linear fc92;
    torch::nn::Linear fc93;
    torch::nn::Linear fc94;
    torch::nn::Linear fc95;
    torch::nn::Linear fc96;
    torch::nn::Linear fc97;
    torch::nn::Linear fc98;
    torch::nn::Linear fc99;
    torch::nn::Linear fc100;
    torch::nn::Linear fc101;
    torch::nn::Linear fc102;
    torch::nn::Linear fc103;
    torch::nn::Linear fc104;
    torch::nn::Linear fc105;
    torch::nn::Linear fc106;
    torch::nn::Linear fc107;
    torch::nn::Linear fc108;
    torch::nn::Linear fc109;
    torch::nn::Linear fc110;
    torch::nn::Linear fc111;
    torch::nn::Linear fc112;
    torch::nn::Linear fc113;
    torch::nn::Linear fc114;
    torch::nn::Linear fc115;
    torch::nn::Linear fc116;
    torch::nn::Linear fc117;
    torch::nn::Linear fc118;
    torch::nn::Linear fc119;
    torch::nn::Linear fc120;
    torch::nn::Linear fc121;
    torch::nn::Linear fc122;
    torch::nn::Linear fc123;
    torch::nn::Linear fc124;
    torch::nn::Linear fc125;
    torch::nn::Linear fc126;
    torch::nn::Linear fc127;
    torch::nn::Linear fc128;
    torch::nn::Linear fc129;
    torch::nn::Linear fc130;
    torch::nn::Linear fc131;
    torch::nn::Linear fc132;
    torch::nn::Linear fc133;
    torch::nn::Linear fc134;
    torch::nn::Linear fc135;
    torch::nn::Linear fc136;
    torch::nn::Linear fc137;
    torch::nn::Linear fc138;
    torch::nn::Linear fc139;
    torch::nn::Linear fc140;
    torch::nn::Linear fc141;
    torch::nn::Linear fc142;
    torch::nn::Linear fc143;
    torch::nn::Linear fc144;
    torch::nn::Linear fc145;
    torch::nn::Linear fc146;
    torch::nn::Linear fc147;
    torch::nn::Linear fc148;
    torch::nn::Linear fc149;
    torch::nn::Linear fc150;
    torch::nn::Linear fc151;
    torch::nn::Linear fc152;
    torch::nn::Linear fc153;
    torch::nn::Linear fc154;
    torch::nn::Linear fc155;
    torch::nn::Linear fc156;
    torch::nn::Linear fc157;
    torch::nn::Linear fc158;
    torch::nn::Linear fc159;
    torch::nn::Linear fc160;
    torch::nn::Linear fc161;
    torch::nn::Linear fc162;
    torch::nn::Linear fc163;
    torch::nn::Linear fc164;
    torch::nn::Linear fc165;
    torch::nn::Linear fc166;
    torch::nn::Linear fc167;
    torch::nn::Linear fc168;
    torch::nn::Linear fc169;
    torch::nn::Linear fc170;
    torch::nn::Linear fc171;
    torch::nn::Linear fc172;
    torch::nn::Linear fc173;
    torch::nn::Linear fc174;
    torch::nn::Linear fc175;
    torch::nn::Linear fc176;
    torch::nn::Linear fc177;
    torch::nn::Linear fc178;
    torch::nn::Linear fc179;
    torch::nn::Linear fc180;
    torch::nn::Linear fc181;
    torch::nn::Linear fc182;
    torch::nn::Linear fc183;
    torch::nn::Linear fc184;
    torch::nn::Linear fc185;
    torch::nn::Linear fc186;
    torch::nn::Linear fc187;
    torch::nn::Linear fc188;
    torch::nn::Linear fc189;
    torch::nn::Linear fc190;
    torch::nn::Linear fc191;
    torch::nn::Linear fc192;
    torch::nn::Linear fc193;
    torch::nn::Linear fc194;
    torch::nn::Linear fc195;
    torch::nn::Linear fc196;
    torch::nn::Linear fc197;
    torch::nn::Linear fc198;
    torch::nn::Linear fc199;
    torch::nn::Linear fc200;
    torch::nn::Linear fc201;
    torch::nn::Linear fc202;
    torch::nn::Linear fc203;
    torch::nn::Linear fc204;
    torch::nn::Linear fc205;
    torch::nn::Linear fc206;
    torch::nn::Linear fc207;
    torch::nn::Linear fc208;
    torch::nn::Linear fc209;
    torch::nn::Linear fc210;
    torch::nn::Linear fc211;
    torch::nn::Linear fc212;
    torch::nn::Linear fc213;
    torch::nn::Linear fc214;
    torch::nn::Linear fc215;
    torch::nn::Linear fc216;
    torch::nn::Linear fc217;
    torch::nn::Linear fc218;
    torch::nn::Linear fc219;
    torch::nn::Linear fc220;
    torch::nn::Linear fc221;
    torch::nn::Linear fc222;
    torch::nn::Linear fc223;
    torch::nn::Linear fc224;
    torch::nn::Linear fc225;
    torch::nn::Linear fc226;
    torch::nn::Linear fc227;
    torch::nn::Linear fc228;
    torch::nn::Linear fc229;
    torch::nn::Linear fc230;
    torch::nn::Linear fc231;
    torch::nn::Linear fc232;
    torch::nn::Linear fc233;
    torch::nn::Linear fc234;
    torch::nn::Linear fc235;
    torch::nn::Linear fc236;
    torch::nn::Linear fc237;
    torch::nn::Linear fc238;
    torch::nn::Linear fc239;
    torch::nn::Linear fc240;
    torch::nn::Linear fc241;
    torch::nn::Linear fc242;
    torch::nn::Linear fc243;
    torch::nn::Linear fc244;
    torch::nn::Linear fc245;
    torch::nn::Linear fc246;
    torch::nn::Linear fc247;
    torch::nn::Linear fc248;
    torch::nn::Linear fc249;
    torch::nn::Linear fc250;
    torch::nn::Linear fc251;
    torch::nn::Linear fc252;
    torch::nn::Linear fc253;
    torch::nn::Linear fc254;
    torch::nn::Linear fc255;
    torch::nn::Linear fc256;
    torch::nn::Linear fc257;
    torch::nn::Linear fc258;
    torch::nn::Linear fc259;
    torch::nn::Linear fc260;
    torch::nn::Linear fc261;
    torch::nn::Linear fc262;
    torch::nn::Linear fc263;
    torch::nn::Linear fc264;
    torch::nn::Linear fc265;
    torch::nn::Linear fc266;
    torch::nn::Linear fc267;
    torch::nn::Linear fc268;
    torch::nn::Linear fc269;
    torch::nn::Linear fc270;
    torch::nn::Linear fc271;
    torch::nn::Linear fc272;
    torch::nn::Linear fc273;
    torch::nn::Linear fc274;
    torch::nn::Linear fc275;
    torch::nn::Linear fc276;
    torch::nn::Linear fc277;
    torch::nn::Linear fc278;
    torch::nn::Linear fc279;
    torch::nn::Linear fc280;
    torch::nn::Linear fc281;
    torch::nn::Linear fc282;
    torch::nn::Linear fc283;
    torch::nn::Linear fc284;
    torch::nn::Linear fc285;
    torch::nn::Linear fc286;
    torch::nn::Linear fc287;
    torch::nn::Linear fc288;
    torch::nn::Linear fc289;
    torch::nn::Linear fc290;
    torch::nn::Linear fc291;
    torch::nn::Linear fc292;
    torch::nn::Linear fc293;
    torch::nn::Linear fc294;
    torch::nn::Linear fc295;
    torch::nn::Linear fc296;
    torch::nn::Linear fc297;
    torch::nn::Linear fc298;
    torch::nn::Linear fc299;
    torch::nn::Linear fc300;
    torch::nn::Linear fc301;
    torch::nn::Linear fc302;
    torch::nn::Linear fc303;
    torch::nn::Linear fc304;
    torch::nn::Linear fc305;
    torch::nn::Linear fc306;
    torch::nn::Linear fc307;
    torch::nn::Linear fc308;
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
        register_module("fc0", fc0);
        register_module("fc1", fc1);
        register_module("fc2", fc2);
        register_module("fc3", fc3);
        register_module("fc4", fc4);
        register_module("fc5", fc5);
        register_module("fc6", fc6);
        register_module("fc7", fc7);
        register_module("fc8", fc8);
        register_module("fc9", fc9);
        register_module("fc10", fc10);
        register_module("fc11", fc11);
        register_module("fc12", fc12);
        register_module("fc13", fc13);
        register_module("fc14", fc14);
        register_module("fc15", fc15);
        register_module("fc16", fc16);
        register_module("fc17", fc17);
        register_module("fc18", fc18);
        register_module("fc19", fc19);
        register_module("fc20", fc20);
        register_module("fc21", fc21);
        register_module("fc22", fc22);
        register_module("fc23", fc23);
        register_module("fc24", fc24);
        register_module("fc25", fc25);
        register_module("fc26", fc26);
        register_module("fc27", fc27);
        register_module("fc28", fc28);
        register_module("fc29", fc29);
        register_module("fc30", fc30);
        register_module("fc31", fc31);
        register_module("fc32", fc32);
        register_module("fc33", fc33);
        register_module("fc34", fc34);
        register_module("fc35", fc35);
        register_module("fc36", fc36);
        register_module("fc37", fc37);
        register_module("fc38", fc38);
        register_module("fc39", fc39);
        register_module("fc40", fc40);
        register_module("fc41", fc41);
        register_module("fc42", fc42);
        register_module("fc43", fc43);
        register_module("fc44", fc44);
        register_module("fc45", fc45);
        register_module("fc46", fc46);
        register_module("fc47", fc47);
        register_module("fc48", fc48);
        register_module("fc49", fc49);
        register_module("fc50", fc50);
        register_module("fc51", fc51);
        register_module("fc52", fc52);
        register_module("fc53", fc53);
        register_module("fc54", fc54);
        register_module("fc55", fc55);
        register_module("fc56", fc56);
        register_module("fc57", fc57);
        register_module("fc58", fc58);
        register_module("fc59", fc59);
        register_module("fc60", fc60);
        register_module("fc61", fc61);
        register_module("fc62", fc62);
        register_module("fc63", fc63);
        register_module("fc64", fc64);
        register_module("fc65", fc65);
        register_module("fc66", fc66);
        register_module("fc67", fc67);
        register_module("fc68", fc68);
        register_module("fc69", fc69);
        register_module("fc70", fc70);
        register_module("fc71", fc71);
        register_module("fc72", fc72);
        register_module("fc73", fc73);
        register_module("fc74", fc74);
        register_module("fc75", fc75);
        register_module("fc76", fc76);
        register_module("fc77", fc77);
        register_module("fc78", fc78);
        register_module("fc79", fc79);
        register_module("fc80", fc80);
        register_module("fc81", fc81);
        register_module("fc82", fc82);
        register_module("fc83", fc83);
        register_module("fc84", fc84);
        register_module("fc85", fc85);
        register_module("fc86", fc86);
        register_module("fc87", fc87);
        register_module("fc88", fc88);
        register_module("fc89", fc89);
        register_module("fc90", fc90);
        register_module("fc91", fc91);
        register_module("fc92", fc92);
        register_module("fc93", fc93);
        register_module("fc94", fc94);
        register_module("fc95", fc95);
        register_module("fc96", fc96);
        register_module("fc97", fc97);
        register_module("fc98", fc98);
        register_module("fc99", fc99);
        register_module("fc100", fc100);
        register_module("fc101", fc101);
        register_module("fc102", fc102);
        register_module("fc103", fc103);
        register_module("fc104", fc104);
        register_module("fc105", fc105);
        register_module("fc106", fc106);
        register_module("fc107", fc107);
        register_module("fc108", fc108);
        register_module("fc109", fc109);
        register_module("fc110", fc110);
        register_module("fc111", fc111);
        register_module("fc112", fc112);
        register_module("fc113", fc113);
        register_module("fc114", fc114);
        register_module("fc115", fc115);
        register_module("fc116", fc116);
        register_module("fc117", fc117);
        register_module("fc118", fc118);
        register_module("fc119", fc119);
        register_module("fc120", fc120);
        register_module("fc121", fc121);
        register_module("fc122", fc122);
        register_module("fc123", fc123);
        register_module("fc124", fc124);
        register_module("fc125", fc125);
        register_module("fc126", fc126);
        register_module("fc127", fc127);
        register_module("fc128", fc128);
        register_module("fc129", fc129);
        register_module("fc130", fc130);
        register_module("fc131", fc131);
        register_module("fc132", fc132);
        register_module("fc133", fc133);
        register_module("fc134", fc134);
        register_module("fc135", fc135);
        register_module("fc136", fc136);
        register_module("fc137", fc137);
        register_module("fc138", fc138);
        register_module("fc139", fc139);
        register_module("fc140", fc140);
        register_module("fc141", fc141);
        register_module("fc142", fc142);
        register_module("fc143", fc143);
        register_module("fc144", fc144);
        register_module("fc145", fc145);
        register_module("fc146", fc146);
        register_module("fc147", fc147);
        register_module("fc148", fc148);
        register_module("fc149", fc149);
        register_module("fc150", fc150);
        register_module("fc151", fc151);
        register_module("fc152", fc152);
        register_module("fc153", fc153);
        register_module("fc154", fc154);
        register_module("fc155", fc155);
        register_module("fc156", fc156);
        register_module("fc157", fc157);
        register_module("fc158", fc158);
        register_module("fc159", fc159);
        register_module("fc160", fc160);
        register_module("fc161", fc161);
        register_module("fc162", fc162);
        register_module("fc163", fc163);
        register_module("fc164", fc164);
        register_module("fc165", fc165);
        register_module("fc166", fc166);
        register_module("fc167", fc167);
        register_module("fc168", fc168);
        register_module("fc169", fc169);
        register_module("fc170", fc170);
        register_module("fc171", fc171);
        register_module("fc172", fc172);
        register_module("fc173", fc173);
        register_module("fc174", fc174);
        register_module("fc175", fc175);
        register_module("fc176", fc176);
        register_module("fc177", fc177);
        register_module("fc178", fc178);
        register_module("fc179", fc179);
        register_module("fc180", fc180);
        register_module("fc181", fc181);
        register_module("fc182", fc182);
        register_module("fc183", fc183);
        register_module("fc184", fc184);
        register_module("fc185", fc185);
        register_module("fc186", fc186);
        register_module("fc187", fc187);
        register_module("fc188", fc188);
        register_module("fc189", fc189);
        register_module("fc190", fc190);
        register_module("fc191", fc191);
        register_module("fc192", fc192);
        register_module("fc193", fc193);
        register_module("fc194", fc194);
        register_module("fc195", fc195);
        register_module("fc196", fc196);
        register_module("fc197", fc197);
        register_module("fc198", fc198);
        register_module("fc199", fc199);
        register_module("fc200", fc200);
        register_module("fc201", fc201);
        register_module("fc202", fc202);
        register_module("fc203", fc203);
        register_module("fc204", fc204);
        register_module("fc205", fc205);
        register_module("fc206", fc206);
        register_module("fc207", fc207);
        register_module("fc208", fc208);
        register_module("fc209", fc209);
        register_module("fc210", fc210);
        register_module("fc211", fc211);
        register_module("fc212", fc212);
        register_module("fc213", fc213);
        register_module("fc214", fc214);
        register_module("fc215", fc215);
        register_module("fc216", fc216);
        register_module("fc217", fc217);
        register_module("fc218", fc218);
        register_module("fc219", fc219);
        register_module("fc220", fc220);
        register_module("fc221", fc221);
        register_module("fc222", fc222);
        register_module("fc223", fc223);
        register_module("fc224", fc224);
        register_module("fc225", fc225);
        register_module("fc226", fc226);
        register_module("fc227", fc227);
        register_module("fc228", fc228);
        register_module("fc229", fc229);
        register_module("fc230", fc230);
        register_module("fc231", fc231);
        register_module("fc232", fc232);
        register_module("fc233", fc233);
        register_module("fc234", fc234);
        register_module("fc235", fc235);
        register_module("fc236", fc236);
        register_module("fc237", fc237);
        register_module("fc238", fc238);
        register_module("fc239", fc239);
        register_module("fc240", fc240);
        register_module("fc241", fc241);
        register_module("fc242", fc242);
        register_module("fc243", fc243);
        register_module("fc244", fc244);
        register_module("fc245", fc245);
        register_module("fc246", fc246);
        register_module("fc247", fc247);
        register_module("fc248", fc248);
        register_module("fc249", fc249);
        register_module("fc250", fc250);
        register_module("fc251", fc251);
        register_module("fc252", fc252);
        register_module("fc253", fc253);
        register_module("fc254", fc254);
        register_module("fc255", fc255);
        register_module("fc256", fc256);
        register_module("fc257", fc257);
        register_module("fc258", fc258);
        register_module("fc259", fc259);
        register_module("fc260", fc260);
        register_module("fc261", fc261);
        register_module("fc262", fc262);
        register_module("fc263", fc263);
        register_module("fc264", fc264);
        register_module("fc265", fc265);
        register_module("fc266", fc266);
        register_module("fc267", fc267);
        register_module("fc268", fc268);
        register_module("fc269", fc269);
        register_module("fc270", fc270);
        register_module("fc271", fc271);
        register_module("fc272", fc272);
        register_module("fc273", fc273);
        register_module("fc274", fc274);
        register_module("fc275", fc275);
        register_module("fc276", fc276);
        register_module("fc277", fc277);
        register_module("fc278", fc278);
        register_module("fc279", fc279);
        register_module("fc280", fc280);
        register_module("fc281", fc281);
        register_module("fc282", fc282);
        register_module("fc283", fc283);
        register_module("fc284", fc284);
        register_module("fc285", fc285);
        register_module("fc286", fc286);
        register_module("fc287", fc287);
        register_module("fc288", fc288);
        register_module("fc289", fc289);
        register_module("fc290", fc290);
        register_module("fc291", fc291);
        register_module("fc292", fc292);
        register_module("fc293", fc293);
        register_module("fc294", fc294);
        register_module("fc295", fc295);
        register_module("fc296", fc296);
        register_module("fc297", fc297);
        register_module("fc298", fc298);
        register_module("fc299", fc299);
        register_module("fc300", fc300);
        register_module("fc301", fc301);
        register_module("fc302", fc302);
        register_module("fc303", fc303);
        register_module("fc304", fc304);
        register_module("fc305", fc305);
        register_module("fc306", fc306);
        register_module("fc307", fc307);
        register_module("fc308", fc308);
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
        temp1 = fc0(src0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc1(src0);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc2(src0);
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
        temp1 = fc5(src0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc6(src0);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc7(src0);
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
        temp2 = fc11(src0);
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
        temp = fc14(temp);
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
        temp = F::relu(temp);
        temp = fc19(temp);
        src = src + temp;


        //encoder-layer-4


        src0 = src;
        nbatches = src0.size(0);
        temp1 = fc20(src0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc21(src0);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc22(src0);
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
        temp = fc23(src);
        temp = F::relu(temp);
        temp = fc24(temp);
        src = src + temp;


        //encoder-layer-5


        src0 = src;
        nbatches = src0.size(0);
        temp1 = fc25(src0);
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
        temp = F::relu(temp);
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
        temp = fc33(src);
        temp = F::relu(temp);
        temp = fc34(temp);
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
        temp = fc38(src);
        temp = F::relu(temp);
        temp = fc39(temp);
        src = src + temp;


        //encoder-layer-8


        src0 = src;
        nbatches = src0.size(0);
        temp1 = fc40(src0);
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
        temp = fc43(src);
        temp = F::relu(temp);
        temp = fc44(temp);
        src = src + temp;


        //encoder-layer-9


        src0 = src;
        nbatches = src0.size(0);
        temp1 = fc45(src0);
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
        temp = F::relu(temp);
        temp = fc49(temp);
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
        temp = fc53(src);
        temp = F::relu(temp);
        temp = fc54(temp);
        src = src + temp;


        //encoder-layer-11


        src0 = src;
        nbatches = src0.size(0);
        temp1 = fc55(src0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc56(src0);
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
        temp = fc58(src);
        temp = F::relu(temp);
        temp = fc59(temp);
        src = src + temp;


        //encoder-layer-12


        src0 = src;
        nbatches = src0.size(0);
        temp1 = fc60(src0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc61(src0);
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
        temp = fc63(src);
        temp = F::relu(temp);
        temp = fc64(temp);
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
        temp = fc68(src);
        temp = F::relu(temp);
        temp = fc69(temp);
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
        temp1 = fc75(src0);
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
        temp = F::relu(temp);
        temp = fc79(temp);
        src = src + temp;


        //encoder-layer-16


        src0 = src;
        nbatches = src0.size(0);
        temp1 = fc80(src0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc81(src0);
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
        temp = fc83(src);
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
        temp = fc89(temp);
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
        temp = fc94(temp);
        src = src + temp;


        //encoder-layer-19


        src0 = src;
        nbatches = src0.size(0);
        temp1 = fc95(src0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc96(src0);
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
        temp = fc103(src);
        temp = F::relu(temp);
        temp = fc104(temp);
        src = src + temp;


        //encoder-layer-21


        src0 = src;
        nbatches = src0.size(0);
        temp1 = fc105(src0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc106(src0);
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
        temp = fc109(temp);
        src = src + temp;


        //encoder-layer-22


        src0 = src;
        nbatches = src0.size(0);
        temp1 = fc110(src0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc111(src0);
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
        temp = fc118(src);
        temp = F::relu(temp);
        temp = fc119(temp);
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
        temp1 = fc123(tgt0);
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
        
        temp = fc126(tgt);
        temp = F::relu(temp);
        temp = fc127(temp);
        tgt = tgt + temp;


        //decoder-layer-1


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp1 = fc128(tgt0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc129(tgt0);
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
        temp1 = fc131(tgt0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc132(src);
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
        
        temp = fc134(tgt);
        temp = F::relu(temp);
        temp = fc135(temp);
        tgt = tgt + temp;


        //decoder-layer-2


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp1 = fc136(tgt0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc137(tgt0);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc138(tgt0);
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
        
        temp = fc142(tgt);
        temp = F::relu(temp);
        temp = fc143(temp);
        tgt = tgt + temp;


        //decoder-layer-3


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp1 = fc144(tgt0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc145(tgt0);
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
        temp = F::relu(temp);
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
        temp1 = fc155(tgt0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc156(src);
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
        temp2 = fc169(tgt0);
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
        temp2 = fc172(src);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
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
        
        temp = fc174(tgt);
        temp = F::relu(temp);
        temp = fc175(temp);
        tgt = tgt + temp;


        //decoder-layer-7


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp1 = fc176(tgt0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc177(tgt0);
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
        temp2 = fc180(src);
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
        temp = fc191(temp);
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
        
        temp = fc198(tgt);
        temp = F::relu(temp);
        temp = fc199(temp);
        tgt = tgt + temp;


        //decoder-layer-10


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp1 = fc200(tgt0);
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
        
        temp = fc206(tgt);
        temp = F::relu(temp);
        temp = fc207(temp);
        tgt = tgt + temp;


        //decoder-layer-11


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp1 = fc208(tgt0);
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
        
        temp = fc214(tgt);
        temp = F::relu(temp);
        temp = fc215(temp);
        tgt = tgt + temp;


        //decoder-layer-12


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp1 = fc216(tgt0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc217(tgt0);
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
        temp1 = fc219(tgt0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc220(src);
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
        temp = fc223(temp);
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
        temp1 = fc227(tgt0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc228(src);
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
        
        temp = fc230(tgt);
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
        
        temp = fc238(tgt);
        temp = F::relu(temp);
        temp = fc239(temp);
        tgt = tgt + temp;


        //decoder-layer-15


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp1 = fc240(tgt0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc241(tgt0);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc242(tgt0);
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
        temp2 = fc244(src);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc245(src);
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
        temp = F::relu(temp);
        temp = fc247(temp);
        tgt = tgt + temp;


        //decoder-layer-16


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp1 = fc248(tgt0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc249(tgt0);
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
        temp2 = fc257(tgt0);
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
        temp = F::relu(temp);
        temp = fc263(temp);
        tgt = tgt + temp;


        //decoder-layer-18


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp1 = fc264(tgt0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc265(tgt0);
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
        temp1 = fc267(tgt0);
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
        
        temp = fc270(tgt);
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
        temp2 = fc276(src);
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
        temp = fc279(temp);
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
        temp = F::relu(temp);
        temp = fc287(temp);
        tgt = tgt + temp;


        //decoder-layer-21


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp1 = fc288(tgt0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc289(tgt0);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
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
        temp1 = fc291(tgt0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc292(src);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc293(src);
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
        temp1 = fc296(tgt0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc297(tgt0);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc298(tgt0);
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
        
        temp = fc302(tgt);
        temp = F::relu(temp);
        temp = fc303(temp);
        tgt = tgt + temp;


        //decoder-layer-23


        tgt0 = tgt;
        nbatches = tgt0.size(0);
        temp1 = fc304(tgt0);
        temp1 = temp1.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp2 = fc305(tgt0);
        temp2 = temp2.view( {nbatches, -1, nheads, d_k} ).transpose(1, 2);
        temp3 = fc306(tgt0);
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
        
        temp = fc310(tgt);
        temp = F::relu(temp);
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
    // std::cout << "Read integer: " << d_model << std::endl;

    transformer model;
    std::cout << "Transformer - CPU version" << std::endl;
    std::cout << "Kernel Switch: 0" << std::endl;
    std::cout << "Associative Op: 360" << std::endl;
    std::cout << "Outsourced Op: 0" << std::endl;

    torch::Tensor src = torch::rand( {32, 10, d_model} );
    torch::Tensor tgt = torch::rand( {32, 20, d_model} );
    torch::Tensor out;

    int count = 100;
    int warmup = 1;
    struct timeval tvs, tve;

    // for (size_t i = 0; i < warmup; i++)
    // {
    //     out = model.forward(src, tgt);
    // }
    
    gettimeofday(&tvs, 0);
    for (size_t i = 0; i < count; i++)
    {
        out = model.forward(src, tgt);
    }
    gettimeofday(&tve, 0);
    float ms_time = (tve.tv_sec - tvs.tv_sec) * 1000 + (tve.tv_usec - tvs.tv_usec) / 1000;
    float latency;
    latency = ms_time/count;
    std::cout << "For " << count << " inferences..." << std::endl;
    std::cout << "Time elapsed: " << ms_time << " ms." << std::endl;
    std::cout << "Fetch here. Time consuming: " << latency << " ms per inference." << std::endl;
    std::cout << "completed." << std::endl;
    return 0;
}
