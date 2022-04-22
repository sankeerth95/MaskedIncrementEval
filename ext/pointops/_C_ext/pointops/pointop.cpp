#include <c10/cuda/CUDAStream.h>

#include "pointop.h"
#include "checks.h"



torch::Tensor pointwise_add(torch::Tensor const &self, torch::Tensor const &input_incr_values, torch::Tensor const &input_incr_indices){
    CHECK_INPUT(self);
    CHECK_INPUT(input_incr_values);
    CHECK_INPUT(input_incr_indices);

    return pointwise_add_cuda(
      static_cast<c10::DeviceIndex>(0),
      self,
      input_incr_values,
      input_incr_indices
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pointwise_add", &pointwise_add, "Pointwise add Forward");
}


