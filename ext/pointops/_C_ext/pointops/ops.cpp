#include <c10/cuda/CUDAStream.h>

#include "ops.h"
#include "utils.h"
#include "checks.h"


//dispatch cuda kernel
void activation_increment(
    torch::Tensor &X,
    torch::Tensor const &in_incr,
    torch::Tensor &out_incr  // expect a zero tensor
)
{
    CHECK_INPUT(X);
    CHECK_INPUT(in_incr);
    CHECK_INPUT(out_incr);

    activation_increment_cuda_wrapper(
      X,
      in_incr,
      out_incr  // expect a zero tensor
    );

}


void conv3x3_increment(
    torch::Tensor const &x_incr,
    torch::Tensor const &filter,
    torch::Tensor &out_incr  // expect a zero tensor
){

    CHECK_INPUT(x_incr);
    CHECK_INPUT(filter);
    CHECK_INPUT(out_incr);

    conv3x3_increment_cuda_wrapper(
      x_incr,
      filter,
      out_incr
    );

}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("activation_increment", &activation_increment, "Activate and increment x tensor;");
  m.def("conv3x3_increment", &conv3x3_increment, "convolution 3x3 kernel;");
}


