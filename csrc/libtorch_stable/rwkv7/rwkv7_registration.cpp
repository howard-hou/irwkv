#include "core/registration.h"

#ifdef IRWKV_RWKV_RUNTIME_OPS
  #include "../ops.h"

  #include <torch/csrc/stable/library.h>

STABLE_TORCH_LIBRARY_FRAGMENT(_RWKV_C, ops) {
  ops.def("get_cuda_view_from_cpu_tensor(Tensor cpu_tensor) -> Tensor");
  ops.def("weak_ref_tensor(Tensor input) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(_RWKV_C, CPU, ops) {
  ops.impl("get_cuda_view_from_cpu_tensor",
           TORCH_BOX(&get_cuda_view_from_cpu_tensor));
}
STABLE_TORCH_LIBRARY_IMPL(_RWKV_C, CUDA, ops) {
  ops.impl("weak_ref_tensor", TORCH_BOX(&weak_ref_tensor));
}
#endif

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
