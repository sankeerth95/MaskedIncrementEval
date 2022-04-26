#pragma once

#include <torch/extension.h>


#define divup(a, b) (((a) + (b) - 1) / (b))



// trivially copyable structs
struct dim{
    int const C, H, W;
    dim(c10::IntArrayRef sizes): C(sizes[0]), H(sizes[1]), W(sizes[2])
    {
    }
};


