from setuptools import setup, find_packages
from torch.utils import cpp_extension

setup(name='qcd_ml_accel',
      ext_modules=[cpp_extension.CppExtension('qcd_ml_accel._C', ['src/qcd_ml_accel/pool4d.cpp']
                                              , extra_compile_args={'cxx': ['-fopenmp', '-O3', '-DAT_PARALLEL_OPENMP']}
                                              , extra_link_args=['-lgomp'])],
      packages=["qcd_ml_accel"],
      package_dir={"qcd_ml_accel": "./src/qcd_ml_accel"},
      url="https://github.com/daknuett/qcd_ml_accel",
      version="0.0.1",
      description="C++ accelerator for QCD ML",
      cmdclass={'build_ext': cpp_extension.BuildExtension})
