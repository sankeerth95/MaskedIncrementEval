#pragma once
#include <torch/extension.h>


// python exposed stuff!

//  incr activation kernels
void activation_increment(
    torch::Tensor &X,
    torch::Tensor const &in_incr,
    torch::Tensor &out_incr  // expect a zero tensor
);




void conv1x1_increment_ext(
    torch::Tensor const &x_incr,
    torch::Tensor const &mask,
    torch::Tensor const &filter,
    torch::Tensor &out_incr,
    int stride
);



void conv3x3_increment_ext(
    torch::Tensor const &x_incr,
    torch::Tensor const &mask,
    torch::Tensor const &filter,
    torch::Tensor &out_incr,
    int stride
);



void conv5x5_increment_ext(
    torch::Tensor const &x_incr,
    torch::Tensor const &mask,
    torch::Tensor const &filter,
    torch::Tensor &out_incr,
    int stride
);



