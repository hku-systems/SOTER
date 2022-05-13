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

struct densenet121eighty : public torch::nn::Module
{
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
    densenet121eighty():
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
        //b1(256),
        c1(256, 128, 1, 1),
        //b2(512),
        c2(512,256, 1, 1),
        avg0(2, 2),
        //b3(1024),
        c3(torch::nn::Conv2d(torch::nn::Conv2dOptions(1024, 512, 1).stride(1))),
        //b4(1024),
        admaxp2d0(1, 1),
        linear0(1024, 1000),
        avg1(2, 2)
        {
            register_module("c0", c0);
            mxp2d0.to(at::kCUDA);
            relu.to(at::kCUDA);

            cv_a_0.to(at::kCUDA);
            register_module("cv_a_1", cv_a_1);
            cv_a_2.to(at::kCUDA);
            cv_a_3.to(at::kCUDA);
            cv_a_4.to(at::kCUDA);
            cv_a_5.to(at::kCUDA);
            cv_a_6.to(at::kCUDA);
            cv_a_7.to(at::kCUDA);
            register_module("cv_a_8", cv_a_8);
            cv_a_9.to(at::kCUDA);
            cv_a_10.to(at::kCUDA);
            cv_a_11.to(at::kCUDA);
            cv_a_12.to(at::kCUDA);
            cv_a_13.to(at::kCUDA);
            cv_a_14.to(at::kCUDA);
            cv_a_15.to(at::kCUDA);
            cv_a_16.to(at::kCUDA);
            register_module("cv_a_17", cv_a_17);
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
            register_module("cv_a_35", cv_a_35);
            cv_a_36.to(at::kCUDA);
            register_module("cv_a_37", cv_a_37);
            cv_a_38.to(at::kCUDA);
            cv_a_39.to(at::kCUDA);
            cv_a_40.to(at::kCUDA);
            cv_a_41.to(at::kCUDA);
            cv_a_42.to(at::kCUDA);
            register_module("cv_a_43", cv_a_43);
            cv_a_44.to(at::kCUDA);
            cv_a_45.to(at::kCUDA);
            cv_a_46.to(at::kCUDA);
            cv_a_47.to(at::kCUDA);
            cv_a_48.to(at::kCUDA);
            register_module("cv_a_49", cv_a_49);
            cv_a_50.to(at::kCUDA);
            cv_a_51.to(at::kCUDA);
            cv_a_52.to(at::kCUDA);
            cv_a_53.to(at::kCUDA);
            register_module("cv_a_54", cv_a_54);
            cv_a_55.to(at::kCUDA);
            cv_a_56.to(at::kCUDA);
            cv_a_57.to(at::kCUDA);
            cv_b_0.to(at::kCUDA);
            cv_b_1.to(at::kCUDA);
            cv_b_2.to(at::kCUDA);
            cv_b_3.to(at::kCUDA);
            cv_b_4.to(at::kCUDA);
            cv_b_5.to(at::kCUDA);
            register_module("cv_b_6", cv_b_6);
            cv_b_7.to(at::kCUDA);
            register_module("cv_b_8", cv_b_8);
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
            register_module("cv_b_19", cv_b_19);
            cv_b_20.to(at::kCUDA);
            cv_b_21.to(at::kCUDA);
            cv_b_22.to(at::kCUDA);
            cv_b_23.to(at::kCUDA);
            cv_b_24.to(at::kCUDA);
            register_module("cv_b_25", cv_b_25);
            cv_b_26.to(at::kCUDA);
            cv_b_27.to(at::kCUDA);
            cv_b_28.to(at::kCUDA);
            cv_b_29.to(at::kCUDA);
            cv_b_30.to(at::kCUDA);
            cv_b_31.to(at::kCUDA);
            register_module("cv_b_32", cv_b_32);
            cv_b_33.to(at::kCUDA);
            cv_b_34.to(at::kCUDA);
            cv_b_35.to(at::kCUDA);
            cv_b_36.to(at::kCUDA);
            register_module("cv_b_37", cv_b_37);
            cv_b_38.to(at::kCUDA);
            cv_b_39.to(at::kCUDA);
            cv_b_40.to(at::kCUDA);
            cv_b_41.to(at::kCUDA);
            cv_b_42.to(at::kCUDA);
            register_module("cv_b_43", cv_b_43);
            cv_b_44.to(at::kCUDA);
            cv_b_45.to(at::kCUDA);
            cv_b_46.to(at::kCUDA);
            cv_b_47.to(at::kCUDA);
            cv_b_48.to(at::kCUDA);
            cv_b_49.to(at::kCUDA);
            cv_b_50.to(at::kCUDA);
            register_module("cv_b_51", cv_b_51);
            cv_b_52.to(at::kCUDA);
            register_module("cv_b_53", cv_b_53);
            cv_b_54.to(at::kCUDA);
            register_module("cv_b_55", cv_b_55);
            register_module("cv_b_56", cv_b_56);
            cv_b_57.to(at::kCUDA);

            //register_module("b1", b1);
            c1.to(at::kCUDA);
            //register_module("b2", b2);
            c2.to(at::kCUDA);
            avg0.to(at::kCUDA);
            //register_module("b3", b3);
            register_module("c3", c3);
            //register_module("b4", b4);
            admaxp2d0.to(at::kCUDA);
            linear0.to(at::kCUDA);
        }

    torch::Tensor forward(torch::Tensor x)
    {
        torch::Tensor temp;
        torch::Tensor x_before;

        x = c0(x);
        //x = b0(x);
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = mxp2d0.forward(temp);
        x = temp.to(at::kCPU);

        // block 1
        x_before = x;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = cv_a_0.forward(temp);
        temp = relu.forward(temp);
        temp = cv_b_0.forward(temp);
        x = temp.to(at::kCPU);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);
        x = cv_a_1(x);
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = cv_b_1.forward(temp);
        x = temp.to(at::kCPU);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = cv_a_2.forward(temp);
        temp = relu.forward(temp);
        temp = cv_b_2.forward(temp);
        x = temp.to(at::kCPU);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = cv_a_3.forward(temp);
        temp = relu.forward(temp);
        temp = cv_b_3.forward(temp);
        x = temp.to(at::kCPU);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = F::relu(x);
        temp = x.to(at::kCUDA);
        temp = cv_a_4.forward(temp);
        temp = relu.forward(temp);
        temp = cv_b_4.forward(temp);
        x = temp.to(at::kCPU);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = cv_a_5.forward(temp);
        x = temp.to(at::kCPU);
        x = F::relu(x);
        temp = x.to(at::kCUDA);
        temp = cv_b_5.forward(temp);
        x = temp.to(at::kCPU);
        x = torch::cat({x_before, x}, 1);

        // transition 1
        //x = b1(x);
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = c1.forward(temp);
        x = temp.to(at::kCPU);
        x = avg1.forward(x);

        // block 2
        x_before = x;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = cv_a_6.forward(temp);
        x = temp.to(at::kCPU);
        x = F::relu(x);
        x = cv_b_6(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = cv_a_7.forward(temp);
        temp = relu.forward(temp);
        temp = cv_b_7.forward(temp);
        x = temp.to(at::kCPU);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = F::relu(x);
        x = cv_a_8(x);
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);
        x = cv_b_8(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = cv_a_9.forward(temp);
        x = temp.to(at::kCPU);
        x = F::relu(x);
        temp = x.to(at::kCUDA);
        temp = cv_b_9.forward(temp);
        x = temp.to(at::kCPU);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = cv_a_10.forward(temp);
        x = temp.to(at::kCPU);
        x = F::relu(x);
        temp = x.to(at::kCUDA);
        temp = cv_b_10.forward(temp);
        x = temp.to(at::kCPU);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = cv_a_11.forward(temp);
        x = temp.to(at::kCPU);
        x = F::relu(x);
        temp = x.to(at::kCUDA);
        temp = cv_b_11.forward(temp);
        x = temp.to(at::kCPU);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = cv_a_12.forward(temp);
        x = temp.to(at::kCPU);
        x = F::relu(x);
        temp = x.to(at::kCUDA);
        temp = cv_b_12.forward(temp);
        x = temp.to(at::kCPU);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = cv_a_13.forward(temp);
        temp = relu.forward(temp);
        temp = cv_b_13.forward(temp);
        x = temp.to(at::kCPU);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = cv_a_14.forward(temp);
        x = temp.to(at::kCPU);
        x = F::relu(x);
        temp = x.to(at::kCUDA);
        temp = cv_b_14.forward(temp);
        x = temp.to(at::kCPU);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = F::relu(x);
        temp = x.to(at::kCUDA);
        temp = cv_a_15.forward(temp);
        x = temp.to(at::kCPU);
        x = F::relu(x);
        temp = x.to(at::kCUDA);
        temp = cv_b_15.forward(temp);
        x = temp.to(at::kCPU);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = cv_a_16.forward(temp);
        temp = relu.forward(temp);
        temp = cv_b_16.forward(temp);
        x = temp.to(at::kCPU);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = F::relu(x);
        x = cv_a_17(x);
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = cv_b_17.forward(temp);
        x = temp.to(at::kCPU);
        x = torch::cat({x_before, x}, 1);

        // transition 2
        //x = b2(x);
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = c2.forward(temp);
        temp = avg0.forward(temp);
        x = temp.to(at::kCPU);

        // block 3
        x_before = x;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = cv_a_18.forward(temp);
        temp = relu.forward(temp);
        temp = cv_b_18.forward(temp);
        x = temp.to(at::kCPU);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = cv_a_19.forward(temp);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);
        x = cv_b_19(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = cv_a_20.forward(temp);
        temp = relu.forward(temp);
        temp = cv_b_20.forward(temp);
        x = temp.to(at::kCPU);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = cv_a_21.forward(temp);
        x = temp.to(at::kCPU);
        x = F::relu(x);
        temp = x.to(at::kCUDA);
        temp = cv_b_21.forward(temp);
        x = temp.to(at::kCPU);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = cv_a_22.forward(temp);
        temp = relu.forward(temp);
        temp = cv_b_22.forward(temp);
        x = temp.to(at::kCPU);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = F::relu(x);
        temp = x.to(at::kCUDA);
        temp = cv_a_23.forward(temp);
        temp = relu.forward(temp);
        temp = cv_b_23.forward(temp);
        x = temp.to(at::kCPU);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = cv_a_24.forward(temp);
        x = temp.to(at::kCPU);
        x = F::relu(x);
        temp = x.to(at::kCUDA);
        temp = cv_b_24.forward(temp);
        x = temp.to(at::kCPU);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = F::relu(x);
        temp = x.to(at::kCUDA);
        temp = cv_a_25.forward(temp);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);
        x = cv_b_25(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = cv_a_26.forward(temp);
        temp = relu.forward(temp);
        temp = cv_b_26.forward(temp);
        x = temp.to(at::kCPU);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = cv_a_27.forward(temp);
        temp = relu.forward(temp);
        temp = cv_b_27.forward(temp);
        x = temp.to(at::kCPU);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = cv_a_28.forward(temp);
        x = temp.to(at::kCPU);
        x = F::relu(x);
        temp = x.to(at::kCUDA);
        temp = cv_b_28.forward(temp);
        x = temp.to(at::kCPU);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = F::relu(x);
        temp = x.to(at::kCUDA);
        temp = cv_a_29.forward(temp);
        temp = relu.forward(temp);
        temp = cv_b_29.forward(temp);
        x = temp.to(at::kCPU);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = cv_a_30.forward(temp);
        temp = relu.forward(temp);
        temp = cv_b_30.forward(temp);
        x = temp.to(at::kCPU);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = cv_a_31.forward(temp);
        x = temp.to(at::kCPU);
        x = F::relu(x);
        temp = x.to(at::kCUDA);
        temp = cv_b_31.forward(temp);
        x = temp.to(at::kCPU);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = cv_a_32.forward(temp);
        x = temp.to(at::kCPU);
        x = F::relu(x);
        x = cv_b_32(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = cv_a_33.forward(temp);
        temp = relu.forward(temp);
        temp = cv_b_33.forward(temp);
        x = temp.to(at::kCPU);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = F::relu(x);
        temp = x.to(at::kCUDA);
        temp = cv_a_34.forward(temp);
        x = temp.to(at::kCPU);
        x = F::relu(x);
        temp = x.to(at::kCUDA);
        temp = cv_b_34.forward(temp);
        x = temp.to(at::kCPU);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);
        x = cv_a_35(x);
        x = F::relu(x);
        temp = x.to(at::kCUDA);
        temp = cv_b_35.forward(temp);
        x = temp.to(at::kCPU);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = cv_a_36.forward(temp);
        temp = relu.forward(temp);
        temp = cv_b_36.forward(temp);
        x = temp.to(at::kCPU);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);
        x = cv_a_37(x);
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);
        x = cv_b_37(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = cv_a_38.forward(temp);
        temp = relu.forward(temp);
        temp = cv_b_38.forward(temp);
        x = temp.to(at::kCPU);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = cv_a_39.forward(temp);
        temp = relu.forward(temp);
        temp = cv_b_39.forward(temp);
        x = temp.to(at::kCPU);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = cv_a_40.forward(temp);
        temp = relu.forward(temp);
        temp = cv_b_40.forward(temp);
        x = temp.to(at::kCPU);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = F::relu(x);
        temp = x.to(at::kCUDA);
        temp = cv_a_41.forward(temp);
        x = temp.to(at::kCPU);
        x = F::relu(x);
        temp = x.to(at::kCUDA);
        temp = cv_b_41.forward(temp);
        x = temp.to(at::kCPU);
        x = torch::cat({x_before, x}, 1);

        //x = b3(x);
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);
        x = c3(x);
        temp = x.to(at::kCUDA);
        temp = avg0.forward(temp);
        x = temp.to(at::kCPU);

        //block 4
        x_before = x;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = cv_a_42.forward(temp);
        temp = relu.forward(temp);
        temp = cv_b_42.forward(temp);
        x = temp.to(at::kCPU);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = F::relu(x);
        x = cv_a_43(x);
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);
        x = cv_b_43(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = cv_a_44.forward(temp);
        temp = relu.forward(temp);
        temp = cv_b_44.forward(temp);
        x = temp.to(at::kCPU);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = cv_a_45.forward(temp);
        x = temp.to(at::kCPU);
        x = F::relu(x);
        temp = x.to(at::kCUDA);
        temp = cv_b_45.forward(temp);
        x = temp.to(at::kCPU);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = cv_a_46.forward(temp);
        x = temp.to(at::kCPU);
        x = F::relu(x);
        temp = x.to(at::kCUDA);
        temp = cv_b_46.forward(temp);
        x = temp.to(at::kCPU);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        x = F::relu(x);
        temp = x.to(at::kCUDA);
        temp = cv_a_47.forward(temp);
        temp = relu.forward(temp);
        temp = cv_b_47.forward(temp);
        x = temp.to(at::kCPU);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = cv_a_48.forward(temp);
        x = temp.to(at::kCPU);
        x = F::relu(x);
        temp = x.to(at::kCUDA);
        temp = cv_b_48.forward(temp);
        x = temp.to(at::kCPU);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);
        x = cv_a_49(x);
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = cv_b_49.forward(temp);
        x = temp.to(at::kCPU);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = cv_a_50.forward(temp);
        temp = relu.forward(temp);
        temp = cv_b_50.forward(temp);
        x = temp.to(at::kCPU);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = cv_a_51.forward(temp);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);
        x = cv_b_51(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = cv_a_52.forward(temp);
        temp = relu.forward(temp);
        temp = cv_b_52.forward(temp);
        x = temp.to(at::kCPU);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = cv_a_53.forward(temp);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);
        x = cv_b_53(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);
        x = cv_a_54(x);
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = cv_b_54.forward(temp);
        x = temp.to(at::kCPU);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = cv_a_55.forward(temp);
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);
        x = cv_b_55(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = cv_a_56.forward(temp);
        x = temp.to(at::kCPU);
        x = F::relu(x);
        x = cv_b_56(x);
        x = torch::cat({x_before, x}, 1);
        x_before = x;
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = cv_a_57.forward(temp);
        temp = relu.forward(temp);
        temp = cv_b_57.forward(temp);
        x = temp.to(at::kCPU);
        x = torch::cat({x_before, x}, 1);

        //end
        //x = b4(x);
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);
        temp = admaxp2d0.forward(temp);
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
    densenet121eighty model;
    std::cout << "DenseNet121 - 80% outsourced to CUDA version" << std::endl;
    std::cout << "Kernel Switch: 79" << std::endl;
    std::cout << "Associative Op: 232" << std::endl;
    std::cout << "Outsourced Op: 182" << std::endl;

    torch::Tensor input = torch::ones({1, 3, 224, 224});
    torch::Tensor output;

    int count = 100;
    int warmup = 1000;
    
    output = model.forward(input);
    std::cout << "Medium activation size: " << total_para << std::endl;
    
    // for (size_t i = 0; i < warmup; i++)
    // {
    //     output = model.forward(input);
    // }

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
    
    return 0;
}