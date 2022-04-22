#include <c10/cuda/CUDAStream.h>

#include "pointop.h"
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("activation_increment", &activation_increment, "Acitvate and increment x tensor;j");
}


