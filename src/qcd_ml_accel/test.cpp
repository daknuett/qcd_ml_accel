// XXX: This is just for testing purposes.
// The code is taken from the PyTorch tutorials.
#include <torch/extension.h>

#include <vector>

namespace extension_cpp {


    at::Tensor mymuladd_cpu(at::Tensor a, const at::Tensor& b, double c) {
      TORCH_CHECK(a.sizes() == b.sizes());
      TORCH_CHECK(a.dtype() == at::kFloat);
      TORCH_CHECK(b.dtype() == at::kFloat);
      TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CPU);
      TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CPU);
      at::Tensor a_contig = a.contiguous();
      at::Tensor b_contig = b.contiguous();
      at::Tensor result = torch::empty(a_contig.sizes(), a_contig.options());
      const float* a_ptr = a_contig.data_ptr<float>();
      const float* b_ptr = b_contig.data_ptr<float>();
      float* result_ptr = result.data_ptr<float>();
      for (int64_t i = 0; i < result.numel(); i++) {
        result_ptr[i] = a_ptr[i] * b_ptr[i] + c;
      }
      return result;
    }

    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

    TORCH_LIBRARY(extension_cpp, m) {
       // Note that "float" in the schema corresponds to the C++ double type
       // and the Python float type.
       m.def("mymuladd(Tensor a, Tensor b, float c) -> Tensor");
     }



    TORCH_LIBRARY_IMPL(extension_cpp, CPU, m) {
      m.impl("mymuladd", &mymuladd_cpu);
    }
}
