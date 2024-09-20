from setuptools import setup, find_packages
from torch.utils import cpp_extension

setup(name='qcd_ml_accel',
      ext_modules=[cpp_extension.CppExtension('_C', ['src/qcd_ml_accel/pool4d.cpp'])],
      packages=find_packages(),
      url="https://github.com/daknuett/qcd_ml_accel",
      version="0.0.1",
      description="C++ accelerator for QCD ML",
      cmdclass={'build_ext': cpp_extension.BuildExtension})
