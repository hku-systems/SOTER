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
int count = 100; // number of inference tests

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

struct vgg19 : public torch::nn::Module
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
    std::vector<std::function<torch::Tensor(torch::Tensor)>> forwards;
        
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
            register_module("c", c);
	        register_module("cv14", cv14);
            register_module("fc1", fc1);
            forwards.push_back(std::bind(&vgg19::forward1_new, this, std::placeholders::_1));
            forwards.push_back(std::bind(&vgg19::forward2_new, this, std::placeholders::_1));
        }

    torch::Tensor forward1_new(torch::Tensor x) {
        // std::cout<<"forward1_new"<<std::endl;
        return x;
    }
    torch::Tensor forward2_new(torch::Tensor x) {
        // std::cout<<"forward2_new"<<std::endl;
        x = torch::max_pool2d(x, 2, 2, 0);
        return x.view({ -1, num_flat_features(x)});
    }

    torch::Tensor forward1(torch::Tensor x) {
        return c(x);
    }

    torch::Tensor forward2(torch::Tensor x) {
        x = x.to(at::kCPU);
        return F::relu(x);     
    }

    torch::Tensor forward3(torch::Tensor x) {
        x = x.to(at::kCPU);
        return F::relu(x);        
    }

    torch::Tensor forward4(torch::Tensor x) {
        x = x.to(at::kCPU);
        x = F::relu(x);
        return cv14(x);        
    }

    torch::Tensor forward5(torch::Tensor x) {
        x = x.to(at::kCPU);
        return x.view({ -1, num_flat_features(x)});
    }

    torch::Tensor forward6(torch::Tensor x) {
        x = x.to(at::kCPU);
        return fc1(x);
    }

    torch::Tensor forward7(torch::Tensor x) {
        return x.to(at::kCPU);
    }

    torch::Tensor forward(torch::Tensor x) {
        torch::Tensor intermedia;
        torch::Tensor tmp;
        torch::Tensor fp_check;
        torch::Tensor proof;
        torch::Tensor intercat;
        if (record_flag == 0){
            std::cout<<"[Preprocessing phase (2/3)] Preparing for cornerstone fingerprint ..."<<std::endl;   

            intermedia = torch::rand({6, 3, 224, 224});
            cfp11 = pass_fingerprint(intermedia, record_flag); 
            cfp12 = pass_fingerprint(intermedia, record_flag); 
            intermedia = torch::rand({1, 25088});
            cfp21 = pass_fingerprint(intermedia, record_flag); 
            cfp22 = pass_fingerprint(intermedia, record_flag); 
            record_flag ++;
            return x;
        } else if (record_flag == 1) {
            std::cout<<"[Preprocessing phase (3/3)] Preparing for oblivious fingerprint {input, output} ..."<<std::endl;

            fp1_1 = generate_obli_fp(1, 1.0, 6.6); // para: [cfp index ; random factor 1 ; random factor 2]
            // std::cout<<"fp1_1 "<<fp1_1[0][0][0][0]<<std::endl;
            std::stringstream ss1;
            torch::save(fp1_1, ss1);
            auto reply1 = request(ss1.str().data(), ss1.str().size(), record_flag);
            std::istringstream is1(reply1);
            torch::load(tmp, is1); 
            pf1_1 = tmp.clone();
            // std::cout<<"pf1_1 "<<pf1_1[0][0][0][0]<<std::endl;

            fp1_2 = generate_obli_fp(1, 2.1, 3.2); 
            // std::cout<<"fp1_2 "<<fp1_2[0][0][0][0]<<std::endl;
            std::stringstream ss2;
            torch::save(fp1_2, ss2);
            auto reply2 = request(ss2.str().data(), ss2.str().size(), record_flag);
            std::istringstream is2(reply2);
            torch::load(tmp, is2); 
            pf1_2 = tmp.clone();
            // std::cout<<"pf1_2 "<<pf1_2[0][0][0][0]<<std::endl;

            fp1_3 = generate_obli_fp(1, 1.1, 6.9); 
            // std::cout<<"fp1_3 "<<fp1_3[0][0][0][0]<<std::endl;
            std::stringstream ss3;
            torch::save(fp1_3, ss3);
            auto reply3 = request(ss3.str().data(), ss3.str().size(), record_flag);
            std::istringstream is3(reply3);
            torch::load(tmp, is3); 
            pf1_3 = tmp.clone();
            // std::cout<<"pf1_3 "<<pf1_3[0][0][0][0]<<std::endl;

            fp2_1 = generate_obli_fp_smallkernel(2, 1.5, 7.6); // para: [cfp index ; random factor 1 ; random factor 2]
            std::stringstream ss4;
            torch::save(fp2_1, ss4);
            auto reply4 = request(ss4.str().data(), ss4.str().size(), record_flag+1);
            std::istringstream is4(reply4);
            torch::load(tmp, is4); 
            pf2_1 = tmp.clone();
            fp2_2 = generate_obli_fp_smallkernel(2, 4.1, 1.3); 
            std::stringstream ss5;
            torch::save(fp2_2, ss5);
            auto reply5 = request(ss5.str().data(), ss5.str().size(), record_flag+1);
            std::istringstream is5(reply5);
            torch::load(tmp, is5); 
            pf2_2 = tmp.clone();
            fp2_3 = generate_obli_fp_smallkernel(2, 1.5, 14.2); 
            std::stringstream ss6;
            torch::save(fp2_3, ss6);
            auto reply6 = request(ss6.str().data(), ss6.str().size(), record_flag+1);
            std::istringstream is6(reply6);
            torch::load(tmp, is6); 
            pf2_3 = tmp.clone();
            record_flag ++;
            return x;
        } else {
            // online inference & fp check 
            std::cout<<"[Inference phase] Inference & integrity check ("<< (record_flag-1) << "/"<<count<<")" <<std::endl; 

            for (int i = 0; i < 2;i++) {
                // intercat = torch::cat({fp_check, intermedia},0);
                intermedia = forwards[i](x);
                std::stringstream ss;
                torch::save(intermedia, ss);
                auto reply = request(ss.str().data(), ss.str().size(), i + 1);
                std::istringstream is(reply);
                torch::load(tmp, is);             
                x = tmp/scalar; // restore 11ms

                // send fingerprints to untrusted GPU for integrity checking
                int idx;
                fp_check = pass_fingerprint_withindex(&idx, i+1);
                std::stringstream ssfp;
                torch::save(fp_check, ssfp);
                auto replyfp = request(ssfp.str().data(), ssfp.str().size(), i + 1); //56ms
                std::istringstream isfp(replyfp);
                torch::load(proof, isfp);
                if (i == 0){
                    integrity_check(i+1, idx, proof);
                } else {
                    integrity_check_smallkernel(i+1, idx, proof);
                }
            }
            
            record_flag ++;
            return x;
        }  
    }

    torch::Tensor pass_fingerprint_withindex(int *idx, int fp_flag){
        srand((unsigned)time(NULL)); 
        int index = (rand() % 2)+ 1;
        *idx = index;
        return getfp(fp_flag, index);
    }

    torch::Tensor generate_obli_fp(int fp_flag, float fact1, float fact2){
        torch::Tensor x1 = getcfp(fp_flag, 1).clone();
        torch::Tensor x2 = getcfp(fp_flag, 2).clone();

        for (int i = 0 ; i < x1.sizes()[0]; i++){
            for (int j = 0 ; j < x1.sizes()[1]; j++){
                for (int m = 0 ; m < x1.sizes()[2]; m++){
                    for (int n = 0 ; n < x1.sizes()[3]; n++){
                        x1[i][j][m][n] = x1[i][j][m][n] * fact1 + x2[i][j][m][n] * fact2;
                    }
                }
            }
        }
        return x1;
    }

    torch::Tensor generate_obli_fp_smallkernel(int fp_flag, float fact1, float fact2){
        torch::Tensor x1 = getcfp(fp_flag, 1).clone();
        torch::Tensor x2 = getcfp(fp_flag, 2).clone();

        for (int i = 0 ; i < x1.sizes()[0]; i++){
            for (int j = 0 ; j < x1.sizes()[1]; j++){
                x1[i][j] = x1[i][j] * fact1 + x2[i][j] * fact2;
            }
        }
        return x1;
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
        if (record_flag < 2){
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
        float left = getpf(fp_flag, idx)[0][0][0][0].item().to<float>();
        float right = proof[0][0][0][0].item().to<float>();
    	if (( (float)( (int)( (left+0.005)*100 ) ) )/100 != ( (float)( (int)( (right+0.005)*100 ) ) )/100){
		    std::cout<<"fp"<<fp_flag<<""<<idx<<" Expect: "<<right<<" Actual: "<<left<<" Detect breaches!"<<std::endl;
	    } 
    }

    void integrity_check_smallkernel(int fp_flag, int idx, torch::Tensor proof){
        float left = getpf(fp_flag, idx)[0][0].item().to<float>();
        float right = proof[0][0].item().to<float>();
    	if (( (float)( (int)( (left+0.005)*100 ) ) )/100 != ( (float)( (int)( (right+0.005)*100 ) ) )/100){
		     std::cout<<"fp"<<fp_flag<<"+"<<idx<<" Expect: "<<right<<" Actual: "<<left<<" Detect breaches!"<<std::endl;
	    } 
    }
    
    torch::Tensor checkcfp1(int i){
        switch(i) {
            case (1):
                return cfp11;
            case (2):
                return cfp12;
        }
    }

    torch::Tensor checkcfp2(int i){
        switch(i) {
            case (1):
                return cfp21;
            case (2):
                return cfp22;
        }
    }
 
    torch::Tensor getcfp(int i, int j){
        switch(i) {
            case (1):
                return checkcfp1(j);
            case (2):
                return checkcfp2(j);
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
        }
    }


    torch::Tensor getfp(int i, int j){
        switch(i) {
            case (1):
                return checkfp1(j);
            case (2):
                return checkfp2(j);
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
        }
    }

    torch::Tensor getpf(int i, int j){
        switch(i) {
            case (1):
                return checkpf1(j);
            case (2):
                return checkpf2(j);
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

    long num_flat_features(torch::Tensor x)
    {
        auto size = x.sizes();
        auto num_features = 1;
        for (auto s : size)
        {
            num_features *= s;
        }
        num_features /= 6;
        return num_features;
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

    vgg19 model;
    model.eval();

    ch_args.SetMaxReceiveMessageSize(28 * 1024 * 1024); 
    greeter = new GreeterClient(grpc::CreateCustomChannel(target_str, grpc::InsecureChannelCredentials(), ch_args));
    torch::Tensor input = torch::rand({6, 3, 224, 224},torch::dtype(torch::kFloat32));
    torch::Tensor output;

    // GPU warm up
    reply = greeter->SayHello(msg, 0);
    std::cout << "[Preprocessing phase (1/3)] Model partitioned & Parameter morphed!" <<std::endl;

    int prepare = 2;
    for (size_t i = 0; i < prepare; i++){
        output = model.forward(input); 
    }

    struct timeval tvs, tve;
    gettimeofday(&tvs, 0);
    for (size_t i = 0; i < count; i++){
        output = model.forward(input);
    }
    gettimeofday(&tve, 0);
    float ms_time = (tve.tv_sec - tvs.tv_sec) * 1000 + (tve.tv_usec - tvs.tv_usec) / 1000;
    std::cout << "For " << count << " inferences ..." << std::endl;
    std::cout << "Time elapsed: " << ms_time << " ms." << std::endl;
    std::cout << "Fetch here. Time consuming: " << ms_time/count << " ms per inference." << std::endl;
    std::cout << "Completed successfully !!!" << std::endl;

    return 0;
}
