#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <torch/torch.h>
#include <iostream>
#include <random>
#include <time.h>
#include <sys/time.h>

namespace F = torch::nn::functional;

int total_para = 0;

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
};

struct resnet152eighty : public torch::nn::Module
{
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

    resnet152eighty():
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
        {
            cv_a_0.to(at::kCUDA);
            cv_a_1.to(at::kCUDA);
            cv_a_2.to(at::kCUDA);
            register_module("cv_a_3", cv_a_3);
            cv_a_4.to(at::kCUDA);
            cv_a_5.to(at::kCUDA);
            cv_a_6.to(at::kCUDA);
            cv_a_7.to(at::kCUDA);
            cv_a_8.to(at::kCUDA);
            cv_a_9.to(at::kCUDA);
            cv_a_10.to(at::kCUDA);
            cv_a_11.to(at::kCUDA);
            register_module("cv_a_12", cv_a_12);
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
            register_module("cv_a_24", cv_a_24);
            cv_a_25.to(at::kCUDA);
            register_module("cv_a_26", cv_a_26);
            cv_a_27.to(at::kCUDA);
            register_module("cv_a_28", cv_a_28);
            cv_a_29.to(at::kCUDA);
            register_module("cv_a_30", cv_a_30);
            cv_a_31.to(at::kCUDA);
            register_module("cv_a_32", cv_a_32);
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
            register_module("cv_a_46", cv_a_46);
            cv_a_47.to(at::kCUDA);
            cv_a_48.to(at::kCUDA);
            cv_a_49.to(at::kCUDA);
            cv_a_50.to(at::kCUDA);
            cv_a_51.to(at::kCUDA);
            cv_a_52.to(at::kCUDA);
            register_module("cv_a_53", cv_a_53);
            cv_b_0.to(at::kCUDA);
            cv_b_1.to(at::kCUDA);
            cv_b_2.to(at::kCUDA);
            cv_b_3.to(at::kCUDA);
            cv_c_0.to(at::kCUDA);
            cv_c_1.to(at::kCUDA);
            cv_c_2.to(at::kCUDA);
            cv_c_3.to(at::kCUDA);
            register_module("cv_c_4", cv_c_4);
            register_module("cv_c_5", cv_c_5);
            cv_c_6.to(at::kCUDA);
            cv_c_7.to(at::kCUDA);
            cv_c_8.to(at::kCUDA);
            cv_c_9.to(at::kCUDA);
            cv_c_10.to(at::kCUDA);
            cv_c_11.to(at::kCUDA);
            cv_c_12.to(at::kCUDA);
            register_module("cv_c_13", cv_c_13);
            cv_c_14.to(at::kCUDA);
            register_module("cv_c_15", cv_c_15);
            cv_c_16.to(at::kCUDA);
            register_module("cv_c_17", cv_c_17);
            register_module("cv_c_18", cv_c_18);
            cv_c_19.to(at::kCUDA);
            cv_c_20.to(at::kCUDA);
            cv_c_21.to(at::kCUDA);
            cv_c_22.to(at::kCUDA);
            register_module("cv_c_23", cv_c_23);
            cv_c_24.to(at::kCUDA);
            cv_c_25.to(at::kCUDA);
            cv_c_26.to(at::kCUDA);
            cv_c_27.to(at::kCUDA);
            register_module("cv_c_28", cv_c_28);
            cv_c_29.to(at::kCUDA);
            cv_c_30.to(at::kCUDA);
            cv_c_31.to(at::kCUDA);
            cv_c_32.to(at::kCUDA);
            cv_c_33.to(at::kCUDA);
            cv_c_34.to(at::kCUDA);
            cv_c_35.to(at::kCUDA);
            register_module("cv_c_36", cv_c_36);
            register_module("cv_c_37", cv_c_37);
            cv_c_38.to(at::kCUDA);
            cv_c_39.to(at::kCUDA);
            cv_c_40.to(at::kCUDA);
            register_module("cv_c_41", cv_c_41);
            cv_c_42.to(at::kCUDA);
            register_module("cv_c_43", cv_c_43);
            cv_c_44.to(at::kCUDA);
            cv_c_45.to(at::kCUDA);
            cv_c_46.to(at::kCUDA);
            register_module("cv_c_47", cv_c_47);
            register_module("cv_c_48", cv_c_48);
            cv_c_49.to(at::kCUDA);
            cv_c_50.to(at::kCUDA);
            cv_c_51.to(at::kCUDA);
            cv_c_52.to(at::kCUDA);
            cv_c_53.to(at::kCUDA);
            register_module("cv_c_54", cv_c_54);
            cv_c_55.to(at::kCUDA);
            cv_c_56.to(at::kCUDA);
            cv_c_57.to(at::kCUDA);
            cv_c_58.to(at::kCUDA);
            register_module("cv_c_59", cv_c_59);
            register_module("cv_c_60", cv_c_60);
            cv_c_61.to(at::kCUDA);
            cv_c_62.to(at::kCUDA);
            cv_c_63.to(at::kCUDA);
            register_module("cv_c_64", cv_c_64);
            cv_c_65.to(at::kCUDA);
            cv_c_66.to(at::kCUDA);
            register_module("cv_c_67", cv_c_67);
            cv_c_68.to(at::kCUDA);
            register_module("cv_c_69", cv_c_69);
            register_module("cv_c_70", cv_c_70);
            cv_c_71.to(at::kCUDA);
            register_module("cv_c_72", cv_c_72);
            cv_c_73.to(at::kCUDA);
            register_module("cv_c_74", cv_c_74);
            cv_c_75.to(at::kCUDA);
            register_module("cv_c_76", cv_c_76);
            cv_c_77.to(at::kCUDA);
            register_module("cv_c_78", cv_c_78);
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
            register_module("cv_c_91", cv_c_91);
            cv_c_92.to(at::kCUDA);
            cv_c_93.to(at::kCUDA);
            cv_c_94.to(at::kCUDA);
            cv_c_95.to(at::kCUDA);
            register_module("cv_c_96", cv_c_96);
            cv_c_97.to(at::kCUDA);
            cv_c_98.to(at::kCUDA);
            register_module("cv_c_99", cv_c_99);
            register_module("cv_c_100", cv_c_100);
            cv_c_101.to(at::kCUDA);
            cv_c_102.to(at::kCUDA);
            cv_c_103.to(at::kCUDA);
            cv_c_104.to(at::kCUDA);
            cv_c_105.to(at::kCUDA);
            register_module("cv_c_106", cv_c_106);
            register_module("cv_c_107", cv_c_107);
            register_module("c0", c0);
            relu.to(at::kCUDA);
            mxp2d0.to(at::kCUDA);
            avg0.to(at::kCUDA);
            linear0.to(at::kCUDA);
        }
    torch::Tensor forward(torch::Tensor x)
    {
        torch::Tensor temp;

        x = c0(x);
        x = F::relu(x);
        temp = x.to(at::kCUDA);
        temp = mxp2d0.forward(temp);
        x = temp.to(at::kCPU);


        //ATTENTION: following is BLOCK 1


        at::Tensor residual0(x.clone());
        temp = x.to(at::kCUDA);
        temp = cv_c_0.forward(temp);
        temp = relu.forward(temp);
        temp = cv_a_0.forward(temp);
        temp = relu.forward(temp);
        temp = cv_c_1.forward(temp);
        x = temp.to(at::kCPU);
        temp = residual0.to(at::kCUDA);
        temp = cv_b_0.forward(temp);
        residual0 = temp.to(at::kCPU);
        x += residual0;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);

        at::Tensor residual1(x.clone());
        temp = x.to(at::kCUDA);
        temp = cv_c_2.forward(temp);
        temp = relu.forward(temp);
        temp = cv_a_1.forward(temp);
        temp = relu.forward(temp);
        temp = cv_c_3.forward(temp);
        x = temp.to(at::kCPU);
        x += residual1;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);

        at::Tensor residual2(x.clone());
        x = cv_c_4(x);
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = cv_a_2.forward(temp);
        x = temp.to(at::kCPU);
        x = F::relu(x);
        x = cv_c_5(x);
        x += residual2;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);

        
        at::Tensor residual3(x.clone());
        temp = x.to(at::kCUDA);
        temp = cv_c_6.forward(temp);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);
        x = cv_a_3(x);
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = cv_c_7.forward(temp);
        x = temp.to(at::kCPU);
        x += residual3;
        x = F::relu(x);


        //ATTENTION: following is BLOCK 2
        at::Tensor residual4(x.clone());
        temp = x.to(at::kCUDA);
        temp = cv_c_8.forward(temp);
        temp = relu.forward(temp);
        temp = cv_a_4.forward(temp);
        temp = relu.forward(temp);
        temp = cv_c_9.forward(temp);
        x = temp.to(at::kCPU);
        temp = residual4.to(at::kCUDA);
        temp = cv_b_1.forward(temp);
        residual4 = temp.to(at::kCPU);
        x += residual4;
        x = F::relu(x);

        at::Tensor residual5(x.clone());
        temp = x.to(at::kCUDA);
        temp = cv_c_10.forward(temp);
        temp = relu.forward(temp);
        temp = cv_a_5.forward(temp);
        temp = relu.forward(temp);
        temp = cv_c_11.forward(temp);
        x = temp.to(at::kCPU);
        x += residual5;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);


        at::Tensor residual6(x.clone());
        temp = x.to(at::kCUDA);
        temp = cv_c_12.forward(temp);
        temp = relu.forward(temp);
        temp = cv_a_6.forward(temp);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);
        x = cv_c_13(x);
        x += residual6;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);

        at::Tensor residual7(x.clone());
        temp = x.to(at::kCUDA);
        temp = cv_c_14.forward(temp);
        temp = relu.forward(temp);
        temp = cv_a_7.forward(temp);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);
        x = cv_c_15(x);
        x += residual7;
        x = F::relu(x);

        at::Tensor residual8(x.clone());
        temp = x.to(at::kCUDA);
        temp = cv_c_16.forward(temp);
        x = temp.to(at::kCPU);
        x = F::relu(x);
        temp = x.to(at::kCUDA);
        temp = cv_a_8.forward(temp);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);
        x = cv_c_17(x);
        x += residual8;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);

        at::Tensor residual9(x.clone());
        x = cv_c_18(x);
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = cv_a_9.forward(temp);
        x = temp.to(at::kCPU);
        x = F::relu(x);
        temp = x.to(at::kCUDA);
        temp = cv_c_19.forward(temp);
        x = temp.to(at::kCPU);
        x += residual9;
        x = F::relu(x);
        at::Tensor residual10(x.clone());
        temp = x.to(at::kCUDA);
        temp = cv_c_20.forward(temp);
        temp = relu.forward(temp);
        temp = cv_a_10.forward(temp);
        x = temp.to(at::kCPU);
        x = F::relu(x);
        temp = x.to(at::kCUDA);
        temp = cv_c_21.forward(temp);
        x = temp.to(at::kCPU);
        x += residual10;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);
        at::Tensor residual11(x.clone());
        temp = x.to(at::kCUDA);
        temp = cv_c_22.forward(temp);
        temp = relu.forward(temp);
        temp = cv_a_11.forward(temp);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);
        x = cv_c_23(x);
        x += residual11;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);
        at::Tensor residual12(x.clone());
        temp = x.to(at::kCUDA);
        temp = cv_c_24.forward(temp);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);
        x = cv_a_12(x);
        x = F::relu(x);
        temp = x.to(at::kCUDA);
        temp = cv_c_25.forward(temp);
        x = temp.to(at::kCPU);
        x += residual12;
        x = F::relu(x);


        //ATTENTION: following is BLOCK 3


        at::Tensor residual13(x.clone());
        temp = x.to(at::kCUDA);
        temp = cv_c_26.forward(temp);
        temp = relu.forward(temp);
        temp = cv_a_13.forward(temp);
        temp = relu.forward(temp);
        temp = cv_c_27.forward(temp);
        x = temp.to(at::kCPU);
        temp = residual13.to(at::kCUDA);
        temp = cv_b_2.forward(temp);
        residual13 = temp.to(at::kCPU);
        x += residual13;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);
        
        at::Tensor residual14(x.clone());
        x = cv_c_28(x);
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = cv_a_14.forward(temp);
        temp = relu.forward(temp);
        temp = cv_c_29.forward(temp);
        x = temp.to(at::kCPU);
        x += residual14;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);
        at::Tensor residual15(x.clone());
        temp = x.to(at::kCUDA);
        temp = cv_c_30.forward(temp);
        temp = relu.forward(temp);
        temp = cv_a_15.forward(temp);
        temp = relu.forward(temp);
        temp = cv_c_31.forward(temp);
        x = temp.to(at::kCPU);
        x += residual15;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);
        at::Tensor residual16(x.clone());
        temp = x.to(at::kCUDA);
        temp = cv_c_32.forward(temp);
        temp = relu.forward(temp);
        temp = cv_a_16.forward(temp);
        temp = relu.forward(temp);
        temp = cv_c_33.forward(temp);
        x = temp.to(at::kCPU);
        x += residual16;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);
        at::Tensor residual17(x.clone());
        temp = x.to(at::kCUDA);
        temp = cv_c_34.forward(temp);
        temp = relu.forward(temp);
        temp = cv_a_17.forward(temp);
        x = temp.to(at::kCPU);
        x = F::relu(x);
        temp = x.to(at::kCUDA);
        temp = cv_c_35.forward(temp);
        x = temp.to(at::kCPU);
        x += residual17;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);
        at::Tensor residual18(x.clone());
        x = cv_c_36(x);
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = cv_a_18.forward(temp);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);
        x = cv_c_37(x);
        x += residual18;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);
        at::Tensor residual19(x.clone());
        temp = x.to(at::kCUDA);
        temp = cv_c_38.forward(temp);
        temp = relu.forward(temp);
        temp = cv_a_19.forward(temp);
        temp = relu.forward(temp);
        temp = cv_c_39.forward(temp);
        x = temp.to(at::kCPU);
        x += residual19;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);
        at::Tensor residual20(x.clone());
        temp = x.to(at::kCUDA);
        temp = cv_c_40.forward(temp);
        temp = relu.forward(temp);
        temp = cv_a_20.forward(temp);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);
        x = cv_c_41(x);
        x += residual20;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);
        at::Tensor residual21(x.clone());
        temp = x.to(at::kCUDA);
        temp = cv_c_42.forward(temp);
        x = temp.to(at::kCPU);
        x = F::relu(x);
        temp = x.to(at::kCUDA);
        temp = cv_a_21.forward(temp);
        x = temp.to(at::kCPU);
        x = F::relu(x);
        x = cv_c_43(x);
        x += residual21;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);
        at::Tensor residual22(x.clone());
        temp = x.to(at::kCUDA);
        temp = cv_c_44.forward(temp);
        temp = relu.forward(temp);
        temp = cv_a_22.forward(temp);
        temp = relu.forward(temp);
        temp = cv_c_45.forward(temp);
        x = temp.to(at::kCPU);
        x += residual22;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);
        at::Tensor residual23(x.clone());
        temp = x.to(at::kCUDA);
        temp = cv_c_46.forward(temp);
        temp = relu.forward(temp);
        temp = cv_a_23.forward(temp);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);
        x = cv_c_47(x);
        x += residual23;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);
        at::Tensor residual24(x.clone());
        x = cv_c_48(x);
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);
        x = cv_a_24(x);
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = cv_c_49.forward(temp);
        x = temp.to(at::kCPU);
        x += residual24;
        x = F::relu(x);
        at::Tensor residual25(x.clone());
        temp = x.to(at::kCUDA);
        temp = cv_c_50.forward(temp);
        x = temp.to(at::kCPU);
        x = F::relu(x);
        temp = x.to(at::kCUDA);
        temp = cv_a_25.forward(temp);
        temp = relu.forward(temp);
        temp = cv_c_51.forward(temp);
        x = temp.to(at::kCPU);
        x += residual25;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);
        at::Tensor residual26(x.clone());
        temp = x.to(at::kCUDA);
        temp = cv_c_52.forward(temp);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);
        x = cv_a_26(x);
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = cv_c_53.forward(temp);
        x = temp.to(at::kCPU);
        x += residual26;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);
        at::Tensor residual27(x.clone());
        x = cv_c_54(x);
        x = F::relu(x);
        temp = x.to(at::kCUDA);
        temp = cv_a_27.forward(temp);
        temp = relu.forward(temp);
        temp = cv_c_55.forward(temp);
        x = temp.to(at::kCPU);
        x += residual27;
        x = F::relu(x);
        at::Tensor residual28(x.clone());
        temp = x.to(at::kCUDA);
        temp = cv_c_56.forward(temp);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);
        x = cv_a_28(x);
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = cv_c_57.forward(temp);
        x = temp.to(at::kCPU);
        x += residual28;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);
        at::Tensor residual29(x.clone());
        temp = x.to(at::kCUDA);
        temp = cv_c_58.forward(temp);
        temp = relu.forward(temp);
        temp = cv_a_29.forward(temp);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);
        x = cv_c_59(x);
        x += residual29;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);
        at::Tensor residual30(x.clone());
        x = cv_c_60(x);
        x = F::relu(x);
        x = cv_a_30(x);
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = cv_c_61.forward(temp);
        x = temp.to(at::kCPU);
        x += residual30;
        x = F::relu(x);
        at::Tensor residual31(x.clone());
        temp = x.to(at::kCUDA);
        temp = cv_c_62.forward(temp);
        temp = relu.forward(temp);
        temp = cv_a_31.forward(temp);
        temp = relu.forward(temp);
        temp = cv_c_63.forward(temp);
        x = temp.to(at::kCPU);
        x += residual31;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);
        at::Tensor residual32(x.clone());
        x = cv_c_64(x);
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);
        x = cv_a_32(x);
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = cv_c_65.forward(temp);
        x = temp.to(at::kCPU);
        x += residual32;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);
        at::Tensor residual33(x.clone());
        temp = x.to(at::kCUDA);
        temp = cv_c_66.forward(temp);
        temp = relu.forward(temp);
        temp = cv_a_33.forward(temp);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);
        x = cv_c_67(x);
        x += residual33;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);
        at::Tensor residual34(x.clone());
        temp = x.to(at::kCUDA);
        temp = cv_c_68.forward(temp);
        temp = relu.forward(temp);
        temp = cv_a_34.forward(temp);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);
        x = cv_c_69(x);
        x += residual34;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);
        at::Tensor residual35(x.clone());
        x = cv_c_70(x);
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = cv_a_35.forward(temp);
        temp = relu.forward(temp);
        temp = cv_c_71.forward(temp);
        x = temp.to(at::kCPU);
        x += residual35;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);
        at::Tensor residual36(x.clone());
        x = cv_c_72(x);
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = cv_a_36.forward(temp);
        temp = relu.forward(temp);
        temp = cv_c_73.forward(temp);
        x = temp.to(at::kCPU);
        x += residual36;
        x = F::relu(x);
        at::Tensor residual37(x.clone());
        x = cv_c_74(x);
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = cv_a_37.forward(temp);
        temp = relu.forward(temp);
        temp = cv_c_75.forward(temp);
        x = temp.to(at::kCPU);
        x += residual37;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);
        at::Tensor residual38(x.clone());
        x = cv_c_76(x);
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = cv_a_38.forward(temp);
        x = temp.to(at::kCPU);
        x = F::relu(x);
        temp = x.to(at::kCUDA);
        temp = cv_c_77.forward(temp);
        x = temp.to(at::kCPU);
        x += residual38;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);
        at::Tensor residual39(x.clone());
        x = cv_c_78(x);
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = cv_a_39.forward(temp);
        temp = relu.forward(temp);
        temp = cv_c_79.forward(temp);
        x = temp.to(at::kCPU);
        x += residual39;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);
        at::Tensor residual40(x.clone());
        temp = x.to(at::kCUDA);
        temp = cv_c_80.forward(temp);
        temp = relu.forward(temp);
        temp = cv_a_40.forward(temp);
        temp = relu.forward(temp);
        temp = cv_c_81.forward(temp);
        x = temp.to(at::kCPU);
        x += residual40;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);
        at::Tensor residual41(x.clone());
        temp = x.to(at::kCUDA);
        temp = cv_c_82.forward(temp);
        temp = relu.forward(temp);
        temp = cv_a_41.forward(temp);
        temp = relu.forward(temp);
        temp = cv_c_83.forward(temp);
        x = temp.to(at::kCPU);
        x += residual41;
        x = F::relu(x);
        at::Tensor residual42(x.clone());
        temp = x.to(at::kCUDA);
        temp = cv_c_84.forward(temp);
        temp = relu.forward(temp);
        temp = cv_a_42.forward(temp);
        temp = relu.forward(temp);
        temp = cv_c_85.forward(temp);
        x = temp.to(at::kCPU);
        x += residual42;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);
        at::Tensor residual43(x.clone());
        temp = x.to(at::kCUDA);
        temp = cv_c_86.forward(temp);
        temp = relu.forward(temp);
        temp = cv_a_43.forward(temp);
        x = temp.to(at::kCPU);
        x = F::relu(x);
        temp = x.to(at::kCUDA);
        temp = cv_c_87.forward(temp);
        x = temp.to(at::kCPU);
        x += residual43;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);
        at::Tensor residual44(x.clone());
        temp = x.to(at::kCUDA);
        temp = cv_c_88.forward(temp);
        temp = relu.forward(temp);
        temp = cv_a_44.forward(temp);
        temp = relu.forward(temp);
        temp = cv_c_89.forward(temp);
        x = temp.to(at::kCPU);
        x += residual44;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);
        at::Tensor residual45(x.clone());
        temp = x.to(at::kCUDA);
        temp = cv_c_90.forward(temp);
        temp = relu.forward(temp);
        temp = cv_a_45.forward(temp);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);
        x = cv_c_91(x);
        x += residual45;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);
        at::Tensor residual46(x.clone());
        temp = x.to(at::kCUDA);
        temp = cv_c_92.forward(temp);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);
        x = cv_a_46(x);
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = cv_c_93.forward(temp);
        x = temp.to(at::kCPU);
        x += residual46;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);
        at::Tensor residual47(x.clone());
        temp = x.to(at::kCUDA);
        temp = cv_c_94.forward(temp);
        temp = relu.forward(temp);
        temp = cv_a_47.forward(temp);
        temp = relu.forward(temp);
        temp = cv_c_95.forward(temp);
        x = temp.to(at::kCPU);
        x += residual47;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);
        at::Tensor residual48(x.clone());
        x = cv_c_96(x);
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = cv_a_48.forward(temp);
        temp = relu.forward(temp);
        temp = cv_c_97.forward(temp);
        x = temp.to(at::kCPU);
        x += residual48;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);
        at::Tensor residual49(x.clone());
        temp = x.to(at::kCUDA);
        temp = cv_c_98.forward(temp);
        temp = relu.forward(temp);
        temp = cv_a_49.forward(temp);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);
        x = cv_c_99(x);
        x += residual49;
        x = F::relu(x);


        //ATTENTION: following is BLOCK 4


        at::Tensor residual50(x.clone());
        x = cv_c_100(x);
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = cv_a_50.forward(temp);
        temp = relu.forward(temp);
        temp = cv_c_101.forward(temp);
        x = temp.to(at::kCPU);
        temp = residual50.to(at::kCUDA);
        temp = cv_b_3.forward(temp);
        residual50 = temp.to(at::kCPU);
        x += residual50;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);
        at::Tensor residual51(x.clone());
        temp = x.to(at::kCUDA);
        temp = cv_c_102.forward(temp);
        temp = relu.forward(temp);
        temp = cv_a_51.forward(temp);
        temp = relu.forward(temp);
        temp = cv_c_103.forward(temp);
        x = temp.to(at::kCPU);
        x += residual51;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);
        at::Tensor residual52(x.clone());
        temp = x.to(at::kCUDA);
        temp = cv_c_104.forward(temp);
        temp = relu.forward(temp);
        temp = cv_a_52.forward(temp);
        temp = relu.forward(temp);
        temp = cv_c_105.forward(temp);
        x = temp.to(at::kCPU);
        x += residual52;
        x = F::relu(x);
        at::Tensor residual53(x.clone());
        x = cv_c_106(x);
        x = F::relu(x);
        x = cv_a_53(x);
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);
        x = cv_c_107(x);
        x += residual53;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);




        // end
        temp = x.to(at::kCUDA);
        temp = avg0.forward(temp);
        x = temp.to(at::kCPU);
        x = x.view({x.sizes()[0], -1});
        temp = x.to(at::kCUDA);
        temp = linear0.forward(temp);
        x = temp.to(at::kCPU);

        return x;
    }
};


int main()
{
    resnet152eighty model;
    std::cout << "ResNet152 - 80% outsourced to CUDA version" << std::endl;
    std::cout << "Kernel Switch: 152" << std::endl;
    std::cout << "Associative Op: 333" << std::endl;
    std::cout << "Outsourced Op: 265" << std::endl;

    torch::Tensor input = torch::ones({1, 3, 224, 224});
    torch::Tensor output;

    int count = 100;
    int warmup = 100;
    for (size_t i = 0; i < warmup; i++)
    {
        output = model.forward(input);
    }

    cudaEvent_t start, stop;
    float esp_time_gpu;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float total = 0;

    for (size_t i = 0; i < count; i++)
    {
        cudaEventRecord(start, 0);
        output = model.forward(input);
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


    // output = model.forward(input);
    // std::cout << "Medium activation size: " << total_para << std::endl;

    return 0;
}