#pragma once
#include <torch/extension.h>


//  incr activation kernels
void activation_increment(
    torch::Tensor &X,
    torch::Tensor const &in_incr,
    torch::Tensor &out_incr  // expect a zero tensor
);

void activation_increment_cuda_wrapper(
    torch::Tensor &X,
    torch::Tensor const &in_incr,
    torch::Tensor &out_incr  // expect a zero tensor
);




//  convolution kernels
void conv3x3_increment(
    torch::Tensor const &x_incr,
    torch::Tensor const &filter,
    torch::Tensor &out_incr  // expect a zero tensor
);


void conv3x3_increment_cuda_wrapper(
    torch::Tensor const &in_incr,
    torch::Tensor const &filter,
    torch::Tensor &out_incr  // expect a zero tensor
);




void conv3x3_increment_ext(
    torch::Tensor const &x_incr,
    torch::Tensor const &filter,
    torch::Tensor &out_incr  // expect a zero tensor
);


void conv3x3_increment_ext_cuda_wrapper(
    torch::Tensor const &in_incr,
    torch::Tensor const &filter,
    torch::Tensor &out_incr  // expect a zero tensor
);

