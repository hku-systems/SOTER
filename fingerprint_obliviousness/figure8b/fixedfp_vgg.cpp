#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <torch/torch.h>
#include <iostream>
#include <random>
#include <time.h>
#include <sys/time.h>
#include <string>
#include <math.h>
#include <fstream>

namespace F = torch::nn::functional;
int nbatches;
float D4reshape_fact = 30.0;
float D2reshape_fact = 10.0;

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
    operator2(torch::nn::Conv2dOptions a):
        conv(a)
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
        mxp2d(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(a).padding(c).stride(b)))
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

struct vgg19 : public torch::nn::Module
{
    int record_flag = 0;
    int total_fixed = 5; 
    int fptypenum = 6;  // totally 6 kernel switches for this partition
    // fingerprint
    torch::Tensor fp1_1;
    torch::Tensor fp1_2;
    torch::Tensor fp1_3;
    torch::Tensor fp1_4;
    torch::Tensor fp1_5;
    torch::Tensor fp2_1;
    torch::Tensor fp2_2;
    torch::Tensor fp2_3;
    torch::Tensor fp2_4;
    torch::Tensor fp2_5;
    torch::Tensor fp3_1;
    torch::Tensor fp3_2;
    torch::Tensor fp3_3;
    torch::Tensor fp3_4;
    torch::Tensor fp3_5;
    torch::Tensor fp4_1;
    torch::Tensor fp4_2;
    torch::Tensor fp4_3;
    torch::Tensor fp4_4;
    torch::Tensor fp4_5;
    torch::Tensor fp5_1;
    torch::Tensor fp5_2;
    torch::Tensor fp5_3;
    torch::Tensor fp5_4;
    torch::Tensor fp5_5;
    torch::Tensor fp6_1;
    torch::Tensor fp6_2;
    torch::Tensor fp6_3;
    torch::Tensor fp6_4;
    torch::Tensor fp6_5;
    
    // fingerprint proof
    torch::Tensor pf1_1;
    torch::Tensor pf1_2;
    torch::Tensor pf1_3;
    torch::Tensor pf1_4;
    torch::Tensor pf1_5;
    torch::Tensor pf2_1;
    torch::Tensor pf2_2;
    torch::Tensor pf2_3;
    torch::Tensor pf2_4;
    torch::Tensor pf2_5;
    torch::Tensor pf3_1;
    torch::Tensor pf3_2;
    torch::Tensor pf3_3;
    torch::Tensor pf3_4;
    torch::Tensor pf3_5;
    torch::Tensor pf4_1;
    torch::Tensor pf4_2;
    torch::Tensor pf4_3;
    torch::Tensor pf4_4;
    torch::Tensor pf4_5;
    torch::Tensor pf5_1;
    torch::Tensor pf5_2;
    torch::Tensor pf5_3;
    torch::Tensor pf5_4;
    torch::Tensor pf5_5;
    torch::Tensor pf6_1;
    torch::Tensor pf6_2;
    torch::Tensor pf6_3;
    torch::Tensor pf6_4;
    torch::Tensor pf6_5;

    operator1 relu;
    operator4 mxp2d0;
    torch::nn::Conv2d c;
    operator2 cv0;
    operator2 cv1;
    operator2 cv2;
    operator2 cv3;
    operator2 cv4;
    operator2 cv5;
    operator2 cv6;
    operator2 cv7;
    operator2 cv8;
    operator2 cv9;
    operator2 cv10;
    operator2 cv11;
    operator2 cv12;
    operator2 cv13;
    torch::nn::Conv2d cv14;
    operator7 fc0;
    torch::nn::Linear fc1;
    operator7 fc2;
    
    vgg19():
        mxp2d0(2, 2, 0),
        c(conv_options(3, 64, 3, 1, 1)),
        cv0(conv_options(64, 64, 3, 1, 1)),
        cv1(conv_options(64, 128, 3, 1, 1)),
        cv2(conv_options(128, 128, 3, 1, 1)),
        cv3(conv_options(128, 256, 3, 1, 1)),
        cv4(conv_options(256, 256, 3, 1, 1)),
        cv5(conv_options(256, 256, 3, 1, 1)),
        cv6(conv_options(256, 256, 3, 1, 1)),
        cv7(conv_options(256, 512, 3, 1, 1)),
        cv8(conv_options(512, 512, 3, 1, 1)),
        cv9(conv_options(512, 512, 3, 1, 1)),
        cv10(conv_options(512, 512, 3, 1, 1)),
        cv11(conv_options(512, 512, 3, 1, 1)),
        cv12(conv_options(512, 512, 3, 1, 1)),
        cv13(conv_options(512, 512, 3, 1, 1)),
        cv14(conv_options(512, 512, 3, 1, 1)),
        fc0(25088, 4096),
        fc1(4096, 4096),
        fc2(4096, 1000)
        {
            relu.to(at::kCUDA);
            mxp2d0.to(at::kCUDA);
            register_module("c", c);
            cv0.to(at::kCUDA);
            cv1.to(at::kCUDA);
            cv2.to(at::kCUDA);
            cv3.to(at::kCUDA);
            cv4.to(at::kCUDA);
            cv5.to(at::kCUDA);
            cv6.to(at::kCUDA);
            cv7.to(at::kCUDA);
            cv8.to(at::kCUDA);
            cv9.to(at::kCUDA);
            cv10.to(at::kCUDA);
            cv11.to(at::kCUDA);
            cv12.to(at::kCUDA);
            cv13.to(at::kCUDA);
	        register_module("cv14", cv14);
            fc0.to(at::kCUDA);
            register_module("fc1", fc1);
            fc2.to(at::kCUDA);
        }
    
    torch::Tensor forward(torch::Tensor x)
    {
        torch::Tensor temp;
	    torch::Tensor fp;
	    torch::Tensor proof;
	
        x = c(x);
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);

	    /*whenever kernel switch to GPU, send fingerprint (i.e., fp)*/
	    // preprocess: prepare fp
	    if(record_flag < total_fixed){
		    switch(record_flag) {
                case (0):
                    fp1_1 = pass_fingerprint(x, record_flag); 
                    // std::cout<<"fp11"<<fp1_1[0][0][0][0]<<std::endl;
                    pf1_1 = relu.forward(fp1_1.to(at::kCUDA));
                    // std::cout<<"pf1_1"<<pf1_1[0][0][0][0]<<std::endl;
                    break;
                case (1):
                    fp1_2 = pass_fingerprint(x, record_flag);
                    pf1_2 = relu.forward(fp1_2.to(at::kCUDA));
                    break;
                case (2):
                    fp1_3 = pass_fingerprint(x, record_flag);
                    pf1_3 = relu.forward(fp1_3.to(at::kCUDA));
                    break;  
                case (3):
                    fp1_4 = pass_fingerprint(x, record_flag);
                    pf1_4 = relu.forward(fp1_4.to(at::kCUDA));
                    break;  
                case (4):
                    fp1_5 = pass_fingerprint(x, record_flag);
                    pf1_5 = relu.forward(fp1_5.to(at::kCUDA));
                    break;  
            }
	    }
        if (record_flag >= total_fixed) {
            // online: send random fp
            int idx;
            fp = pass_fingerprint_withindex(x, record_flag, &idx, 1);
	        proof = relu.forward(fp.to(at::kCUDA));
	        integrity_check(1, idx, proof);
        }
 
        temp = cv0.forward(temp);
        temp = relu.forward(temp);
        temp = mxp2d0.forward(temp);
        temp = cv1.forward(temp);
        temp = relu.forward(temp);
        temp = cv2.forward(temp);
        temp = relu.forward(temp);
        temp = mxp2d0.forward(temp);
        temp = cv3.forward(temp);
        temp = relu.forward(temp);
        temp = cv4.forward(temp);
        temp = relu.forward(temp);
        temp = cv5.forward(temp);
        temp = relu.forward(temp);
        temp = cv6.forward(temp);
        temp = relu.forward(temp);
        temp = mxp2d0.forward(temp);
        temp = cv7.forward(temp);
        temp = relu.forward(temp);
        temp = cv8.forward(temp);
        x = temp.to(at::kCPU);
        x = F::relu(x);
        temp = x.to(at::kCUDA);
        temp = cv9.forward(temp);

	    // preprocess: prepare fp
	    if(record_flag < total_fixed){
            fp = pass_fingerprint(x, record_flag, 2);
		    switch(record_flag) {
                case (0):
                    fp2_1 = fp; 
                    pf2_1 = cv9.forward(fp2_1.to(at::kCUDA));
                    break;
                case (1):
                    fp2_2 = pass_fingerprint(x, record_flag);
                    pf2_2 = cv9.forward(fp2_2.to(at::kCUDA));
                    break;
                case (2):
                    fp2_3 = pass_fingerprint(x, record_flag);
                    pf2_3 = cv9.forward(fp2_3.to(at::kCUDA));
                    break;  
                case (3):
                    fp2_4 = pass_other_fixed_fp(fp, 1.18);
                    pf2_4 = cv9.forward(fp2_4.to(at::kCUDA));
                    break;  
                case (4):
                    fp2_5 = pass_fingerprint(x, record_flag);
                    pf2_5 = cv9.forward(fp2_5.to(at::kCUDA));
                    break;  
            }
	    }
        if (record_flag >= total_fixed) {
            int idx;
            fp = pass_fingerprint_withindex(x, record_flag, &idx, 2);
	        proof = cv9.forward(fp.to(at::kCUDA));
	        integrity_check(2, idx, proof);
        }

        x = temp.to(at::kCPU);
        x = F::relu(x);
        temp = x.to(at::kCUDA);
        temp = cv10.forward(temp);

        // preprocess: prepare fp
	    if(record_flag < total_fixed){
            fp = pass_fingerprint(x, record_flag, 3);
		    switch(record_flag) {
                case (0):
                    fp3_1 = fp; 
                    pf3_1 = cv10.forward(fp3_1.to(at::kCUDA));
                    // std::cout << "prepare fp11 proof: "<<pf1_1[0][0][0][0].item().to<float>()<< std::endl;
                    break;
                case (1):
                    fp3_2 = pass_fingerprint(x, record_flag);
                    pf3_2 = cv10.forward(fp3_2.to(at::kCUDA));
                    break;
                case (2):
                    fp3_3 = pass_other_fixed_fp(fp, 2.56);  
                    pf3_3 = cv10.forward(fp3_3.to(at::kCUDA));
                    break;  
                case (3):
                    fp3_4 = pass_other_fixed_fp(fp, 1.18); 
                    pf3_4 = cv10.forward(fp3_4.to(at::kCUDA));
                    break;  
                case (4):
                    fp3_5 = pass_fingerprint(x, record_flag);
                    pf3_5 = cv10.forward(fp3_5.to(at::kCUDA));
                    break;  
            }
	    }
        if (record_flag >= total_fixed) {
            int idx;
            fp = pass_fingerprint_withindex(x, record_flag,  &idx, 3);
	        proof = cv10.forward(fp.to(at::kCUDA));
	        integrity_check(3, idx, proof);
        }
        
        temp = relu.forward(temp);
        temp = mxp2d0.forward(temp);
        temp = cv11.forward(temp);
        temp = relu.forward(temp);
        temp = cv12.forward(temp);
        temp = relu.forward(temp);
        temp = cv13.forward(temp);
        x = temp.to(at::kCPU);
        x = F::relu(x);
        x = cv14(x);
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);

        // preprocess: prepare fp
	    if(record_flag < total_fixed){
            fp = pass_fingerprint(x, record_flag, 4);
		    switch(record_flag) {
                case (0):
                    fp4_1 = fp; 
                    pf4_1 = relu.forward(fp4_1.to(at::kCUDA));
                    // std::cout << "prepare fp11 proof: "<<pf1_1[0][0][0][0].item().to<float>()<< std::endl;
                    break;
                case (1):
                    fp4_2 = pass_other_fixed_fp(fp, 2.15);  
                    pf4_2 = relu.forward(fp4_2.to(at::kCUDA));
                    // std::cout << "fp4_2"<< std::endl;
                    break;
                case (2):
                    fp4_3 = pass_other_fixed_fp(fp, 5.05);   
                    pf4_3 = relu.forward(fp4_3.to(at::kCUDA));
                    break;  
                case (3):
                    fp4_4 = pass_other_fixed_fp(fp, 2.38);  
                    pf4_4 = relu.forward(fp4_4.to(at::kCUDA));
                    break;  
                case (4):
                    fp4_5 = pass_fingerprint(x, record_flag);
                    pf4_5 = relu.forward(fp4_5.to(at::kCUDA));
                    break;  
            }
	    }
        if (record_flag >= total_fixed) {
            int idx;
            fp = pass_fingerprint_withindex(x, record_flag,  &idx, 4);
	        proof = relu.forward(fp.to(at::kCUDA));
	        integrity_check(4, idx, proof);
        }
     
        temp = mxp2d0.forward(temp);
        x = temp.to(at::kCPU);
        x = x.view({ -1, num_flat_features(x)});
        temp = x.to(at::kCUDA);
        temp = fc0.forward(temp);

        // preprocess: prepare fp
	    if(record_flag < total_fixed){
            fp = pass_fingerprint(x, record_flag, 5);
		    switch(record_flag) {
                case (0):
                    fp5_1 = fp; 
                    pf5_1 = fc0.forward(fp5_1.to(at::kCUDA));
                    // std::cout<<"fp5_1 "<<fp5_1[0][0].item().to<float>()<<std::endl;
                    // std::cout<<"pf5_1 "<<pf5_1[0][0].item().to<float>()<<std::endl;
                    break;
                case (1):
                    fp5_2 = pass_other_fixed_fp_smallkernel(fp, 1.15); 
                    pf5_2 = fc0.forward(fp5_2.to(at::kCUDA));
                    // std::cout<<"fp5_2 "<<fp5_2[0][0].item().to<float>()<<std::endl;
                    // std::cout<<"pf5_2 "<<pf5_2[0][0].item().to<float>()<<std::endl;
                    break;
                case (2):
                    fp5_3 = pass_other_fixed_fp_smallkernel(fp, 1.5);  
                    pf5_3 = fc0.forward(fp5_3.to(at::kCUDA));
                    // std::cout<<"fp5_3 "<<fp5_3[0][0].item().to<float>()<<std::endl;
                    // std::cout<<"pf5_3 "<<pf5_3[0][0].item().to<float>()<<std::endl;
                    break;  
                case (3):
                    fp5_4 = pass_other_fixed_fp_smallkernel(fp, 0.7038); 
                    pf5_4 = fc0.forward(fp5_4.to(at::kCUDA));
                    // std::cout<<"fp5_4 "<<fp5_4[0][0].item().to<float>()<<std::endl;
                    // std::cout<<"pf5_4 "<<pf5_4[0][0].item().to<float>()<<std::endl;
                    break;  
                case (4):
                    fp5_5 = pass_fingerprint(x, record_flag);
                    pf5_5 = fc0.forward(fp5_5.to(at::kCUDA));
                    // std::cout<<"fp5_5 "<<fp5_5[0][0].item().to<float>()<<std::endl;
                    // std::cout<<"pf5_5 "<<pf5_5[0][0].item().to<float>()<<std::endl;
                    break;  
            }
	    }
        if (record_flag >= total_fixed) {
            int idx;
            fp = pass_fingerprint_withindex(x, record_flag,  &idx, 5);
	        proof = fc0.forward(fp.to(at::kCUDA));
            // std::cout<<"proof"<<proof[0][0].item().to<float>()<<std::endl;
	        integrity_check_smallkernel(5, idx, proof);
        }
        
        temp = relu.forward(temp);
        x = temp.to(at::kCPU);
        x = fc1(x);
        temp = x.to(at::kCUDA);
        temp = relu.forward(temp);

        // preprocess: prepare fp
	    if(record_flag < total_fixed){
            fp = pass_fingerprint(x, record_flag, 6);
		    switch(record_flag) {
                case (0):
                    fp6_1 = fp; 
                    pf6_1 = relu.forward(fp6_1.to(at::kCUDA));
                    record_flag ++;
                    // std::cout << "prepare fp11 proof: "<<pf1_1[0][0][0][0].item().to<float>()<< std::endl;
                    break;
                case (1):
                    fp6_2 = pass_other_fixed_fp_smallkernel(fp, 3.15);
                    pf6_2 = relu.forward(fp6_2.to(at::kCUDA));
                    // std::cout<<"fp6_2 "<<fp6_2[0][0].item().to<float>()<<std::endl;
                    // std::cout<<"pf6_2 "<<pf6_2[0][0].item().to<float>()<<std::endl;
                    record_flag ++;
                    break;
                case (2):
                    fp6_3 = pass_other_fixed_fp_smallkernel(fp, 4.96); 
                    pf6_3 = relu.forward(fp6_3.to(at::kCUDA));
                    // std::cout<<"fp6_3 "<<fp6_3[0][0].item().to<float>()<<std::endl;
                    // std::cout<<"pf6_3 "<<pf6_3[0][0].item().to<float>()<<std::endl;
                    record_flag ++;
                    break;  
                case (3):
                    fp6_4 = pass_other_fixed_fp_smallkernel(fp, 7.65);
                    pf6_4 = relu.forward(fp6_4.to(at::kCUDA));
                    record_flag ++;
                    break;  
                case (4):
                    fp6_5 = pass_fingerprint(x, record_flag);
                    pf6_5 = relu.forward(fp6_5.to(at::kCUDA));
                    record_flag ++;
                    break;  
            }
	    }
        if (record_flag >= total_fixed) {
            int idx;
            fp = pass_fingerprint_withindex(x, record_flag,  &idx, 6);
	        proof = relu.forward(fp.to(at::kCUDA));
	        integrity_check_smallkernel(6, idx, proof);
        }

        temp = fc2.forward(temp);
        x = temp.to(at::kCPU);

        return x;
    }

    int getDiffFPNum(){
        return fptypenum;
    }

    torch::Tensor pass_fingerprint_withindex(torch::Tensor t, int record_flag, int *idx, int fp_flag = 1){
        // random generate new fp
        if (record_flag < 5){
            torch::Tensor fp = torch::rand(t.sizes());
        	return fp;
        } else{
            // fixed at runtime, randomly select existing fp
            srand((unsigned)time(NULL)); 
            int index = (rand() % 5)+ 1;
            *idx = index;
            return getfp(fp_flag, index);
        }
    }
    
    torch::Tensor pass_fingerprint(torch::Tensor t, int record_flag, int fp_flag = 1){
        // random generate new fp
        if (record_flag < 5){
            torch::Tensor fp = torch::rand(t.sizes());
        	return fp;
        } else{
            // fixed at runtime, randomly select existing fp
            srand((unsigned)time(NULL)); 
            int index = (rand() % 5)+ 1;
            return getfp(fp_flag, index);
        }
    }

    void integrity_check(int fp_flag, int idx, torch::Tensor proof){
    	if (getpf(fp_flag, idx)[0][0][0][0].item().to<float>() != proof[0][0][0][0].item().to<float>()){
		    std::cout<<"fp"<<fp_flag<<"+"<<idx<<" Expect: "<<getpf(fp_flag, idx)[0][0][0][0].item().to<float>()<<" Actual: "<<proof[0][0][0][0].item().to<float>()<<" Detect breaches!"<<std::endl;
	    } 
    }

    void integrity_check_smallkernel(int fp_flag, int idx, torch::Tensor proof){
    	if (getpf(fp_flag, idx)[0][0].item().to<float>() != proof[0][0].item().to<float>()){
		     std::cout<<"fp"<<fp_flag<<"+"<<idx<<" Expect: "<<getpf(fp_flag, idx)[0][0].item().to<float>()<<" Actual: "<<proof[0][0].item().to<float>()<<" Detect breaches!"<<std::endl;
	    } 
    }

    torch::Tensor checkfp1(int i){
        switch(i) {
            case (1):
                return fp1_1;
            case (2):
                return fp1_2;
            case (3):
                return fp1_3; 
            case (4):
                return fp1_4; 
            case (5):
                return fp1_5; 
        }
    }

    torch::Tensor checkfp2(int i){
        switch(i) {
            case (1):
                return fp2_1;
            case (2):
                return fp2_2;
            case (3):
                return fp2_3; 
            case (4):
                return fp2_4; 
            case (5):
                return fp2_5; 
        }
    }

    torch::Tensor checkfp3(int i){
        switch(i) {
            case (1):
                return fp3_1;
            case (2):
                return fp3_2;
            case (3):
                return fp3_3; 
            case (4):
                return fp3_4; 
            case (5):
                return fp3_5; 
        }
    }

    torch::Tensor checkfp4(int i){
        switch(i) {
            case (1):
                return fp4_1;
            case (2):
                return fp4_2;
            case (3):
                return fp4_3; 
            case (4):
                return fp4_4; 
            case (5):
                return fp4_5; 
        }
    }

    torch::Tensor checkfp5(int i){
        switch(i) {
            case (1):
                return fp5_1;
            case (2):
                return fp5_2;
            case (3):
                return fp5_3; 
            case (4):
                return fp5_4; 
            case (5):
                return fp5_5; 
        }
    }

    torch::Tensor checkfp6(int i){
        switch(i) {
            case (1):
                return fp6_1;
            case (2):
                return fp6_2;
            case (3):
                return fp6_3; 
            case (4):
                return fp6_4; 
            case (5):
                return fp6_5; 
        }
    }

    torch::Tensor getfp(int i, int j){
        switch(i) {
            case (1):
                return checkfp1(j);
            case (2):
                return checkfp2(j);
            case (3):
                return checkfp3(j);
            case (4):
                return checkfp4(j);
            case (5):
                return checkfp5(j);
            case (6):
                return checkfp6(j);
        }
    }

    torch::Tensor checkpf1(int i){
        switch(i) {
            case (1):
                return pf1_1;
            case (2):
                return pf1_2;
            case (3):
                return pf1_3; 
            case (4):
                return pf1_4; 
            case (5):
                return pf1_5; 
        }
    }

    torch::Tensor checkpf2(int i){
        switch(i) {
            case (1):
                return pf2_1;
            case (2):
                return pf2_2;
            case (3):
                return pf2_3; 
            case (4):
                return pf2_4; 
            case (5):
                return pf2_5; 
        }
    }

    torch::Tensor checkpf3(int i){
        switch(i) {
            case (1):
                return pf3_1;
            case (2):
                return pf3_2;
            case (3):
                return pf3_3; 
            case (4):
                return pf3_4; 
            case (5):
                return pf3_5; 
        }
    }

    torch::Tensor checkpf4(int i){
        switch(i) {
            case (1):
                return pf4_1;
            case (2):
                return pf4_2;
            case (3):
                return pf4_3; 
            case (4):
                return pf4_4; 
            case (5):
                return pf4_5; 
        }
    }

    torch::Tensor checkpf5(int i){
        switch(i) {
            case (1):
                return pf5_1;
            case (2):
                return pf5_2;
            case (3):
                return pf5_3; 
            case (4):
                return pf5_4; 
            case (5):
                return pf5_5; 
        }
    }

    torch::Tensor checkpf6(int i){
        switch(i) {
            case (1):
                return pf6_1;
            case (2):
                return pf6_2;
            case (3):
                return pf6_3; 
            case (4):
                return pf6_4; 
            case (5):
                return pf6_5; 
        }
    }

    torch::Tensor getpf(int i, int j){
        switch(i) {
            case (1):
                return checkpf1(j);
            case (2):
                return checkpf2(j);
            case (3):
                return checkpf3(j);
            case (4):
                return checkpf4(j);
            case (5):
                return checkpf5(j);
            case (6):
                return checkpf6(j);
        }
    }

    torch::Tensor pass_other_fixed_fp(torch::Tensor t, float fact){
        // std::cout<<"fact:"<<fact<<std::endl;
        torch::Tensor x = t.clone();
        for (int i = 0 ; i < x.sizes()[0]; i++){
            for (int j = 0 ; j < x.sizes()[1]; j++){
                for (int m = 0 ; m < x.sizes()[2]; m++){
                    for (int n = 0 ; n < x.sizes()[3]; n++){
                        x[i][j][m][n] = x[i][j][m][n] * fact;
                    }
                }
            }
        }
        return x;
    }

    torch::Tensor pass_other_fixed_fp_smallkernel(torch::Tensor t, float fact){
        // std::cout<<"fact:"<<fact<<std::endl;
        torch::Tensor x = t.clone();
        for (int i = 0 ; i < x.sizes()[0]; i++){
            for (int j = 0 ; j < x.sizes()[1]; j++){
                x[i][j] = x[i][j] * fact;
            }
        }
        return x;
    }

    void checkfpsize(){
    	std::cout<<fp1_1.sizes()<<std::endl;
        std::cout<<fp2_1.sizes()<<std::endl;
        std::cout<<fp3_1.sizes()<<std::endl;
        std::cout<<fp4_1.sizes()<<std::endl;
        std::cout<<fp5_1.sizes()<<std::endl;
        std::cout<<fp6_1.sizes()<<std::endl;
    }

    long num_flat_features(torch::Tensor x)
    {
        auto size = x.sizes();
        auto num_features = 1;
        for (auto s : size)
        {
            num_features *= s;
        }
        num_features /= nbatches;
        return num_features;
    }

};

float l2_dist(torch::Tensor x, torch::Tensor y){
    float dist = 0;
    float temp = 0;
    for (int i = 0 ; i < x.sizes()[0]; i++){
        for (int j = 0 ; j < x.sizes()[1]; j++){
            for (int m = 0 ; m < x.sizes()[2]; m++){
                for (int n = 0 ; n < x.sizes()[3]; n++){
                    temp += pow((x[i][j][m][n].item().to<float>()-y[i][j][m][n].item().to<float>()),2);
                }
            }
        }
    }
    dist = sqrt(temp)/D4reshape_fact;
    return floor(dist);
}

float l2_dist_smallsize(torch::Tensor x, torch::Tensor y){
    float dist = 0;
    float temp = 0;
    for (int i = 0 ; i < x.sizes()[0]; i++){
        for (int j = 0 ; j < x.sizes()[1]; j++){
            temp += pow((x[i][j].item().to<float>()-y[i][j].item().to<float>()),2);
        }
    }
    dist = sqrt(temp)/D2reshape_fact;
    return floor(dist);
}

void attack_fingerprint_pattern(vgg19 model){
    // grab fixed fps (Note: the same grablist will be applied to oblivious fingerprint -
    // soterfp_vgg.cpp, to show soter's unobservable pattern)  
    // fp: 1-2, 1-3, 1-4, 2-3
    std::cout << "[Inference phase] Attack starts. This step takes around ** 3min20s ** !" << std::endl;
    int checkpattern_indexlist[] = { 5, 2, 5, 5,  16, 16, 12, 10,  16, 5, 5, 17,  10, 3, 10, 1,  10, 14, 1, 5,  17, 4, 7, 1};
    float temp = 0.0;

    std::ofstream outfile;
    outfile.open("../l2_dist_fixed_fp.dat");
    if (outfile) {
        std::cout << "Log file created." << std::endl;
    }

    // for each group of fingerprints belonging to a partitioned op
    for (int i = 0; i < model.getDiffFPNum() ; i++){
        for (int j = 0 ; j < 4 ; j++){
            // grab every element 
            //evaluate l2 dist with proper kernel size
            if(i < 4){
                temp = l2_dist(model.getfp((i+1),1), model.getfp((i+1),(j+2)));
            } else {
                temp = l2_dist_smallsize(model.getfp((i+1),1), model.getfp((i+1),(j+2)));
            }
            for (int k = 0 ; k < checkpattern_indexlist[(i*4)+j] ; k++){
                outfile << temp << std::endl;
            }
            std::cout << "Grab fp-content = "<< temp << ", Num = "<<checkpattern_indexlist[(i*4)+j] << " to log file"<<std::endl;
            // continue;
        }    
    }
    outfile.close();
}   

int main(int argc, char* argv[])
{
    nbatches = std::stoi(argv[1]); std::cout << "vgg19 {" << nbatches << " , 3, 224, 224} 80% securely outsourced to CUDA version\n" << std::endl;
    vgg19 model;
    model.eval();
    
    int count = 1000;
    int warmup = 5;

    torch::Tensor input = torch::rand({nbatches, 3, 224, 224},torch::dtype(torch::kFloat32));
    torch::Tensor output;
    std::cout << "[Preprocessing phase] This step takes around ** 30s ** !"<< std::endl;
    for (size_t i = 0; i < count; i++)
    {
        if(i < warmup) {
            std::cout << "[Preprocessing phase] Preparing for baseline fixed fingerprints ... (" <<(i+1)<<"/"<<warmup<<")"<< std::endl;
            output = model.forward(input);  
        } else {
            std::cout << "[Preprocessing phase] Preparing for inferences, GPU is warming up ..."<< std::endl;
            output = model.forward(input); 
        }
            
    }

    attack_fingerprint_pattern(model);
    
    return 0;
}
                                                                                                                                                                                                                                                                                                              