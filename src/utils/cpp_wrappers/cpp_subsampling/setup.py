from distutils.core import setup, Extension
import numpy.distutils.misc_util

# Adding OpenCV to project
# ************************

# Adding sources of the project
# *****************************

module = Extension(
    "grid_subsampling",
    sources=[
        "../cpp_utils/cloud/cloud.cpp", "src/grid_subsampling.cpp",
        "wrapper.cpp"
    ],
    extra_compile_args=['-std=c++11', '-D_GLIBCXX_USE_CXX11_ABI=0'])

setup(name='cpp_wrappers',
      ext_modules=[module],
      include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs())
