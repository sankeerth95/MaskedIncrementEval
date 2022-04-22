#pragma once
#include <torch/extension.h>

#define divup(a, b) (((a) + (b) - 1) / (b))

struct dim{
    int const C, H, W;
    dim(c10::IntArrayRef sizes): C(sizes[0]), H(sizes[1]), W(sizes[2])
    {

    }
};

template <typename scalar_t>
void activation_increment_cuda(
    torch::Tensor &X,
    torch::Tensor const &in_incr,
    torch::Tensor &out_incr  // expect a zero tensor
);




// cpu functions
void activation_increment_cuda_wrapper(
    torch::Tensor &X,
    torch::Tensor const &in_incr,
    torch::Tensor &out_incr  // expect a zero tensor
);

void activation_increment(
    torch::Tensor &X,
    torch::Tensor const &in_incr,
    torch::Tensor &out_incr  // expect a zero tensor
);





