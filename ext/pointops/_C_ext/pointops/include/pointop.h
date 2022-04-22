#pragma once
#include <torch/extension.h>

torch::Tensor pointwise_add(torch::Tensor const &self, torch::Tensor const &input_incr_values, \
    torch::Tensor const &input_incr_indices);

torch::Tensor pointwise_add_cuda(c10::DeviceIndex const device_idx, torch::Tensor const &X ,\
    torch::Tensor const &input_incr_values, torch::Tensor const &input_incr_indices);


