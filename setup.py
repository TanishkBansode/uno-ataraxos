from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import platform

sys_name = platform.system()
mach_name = platform.machine().lower()

extra_compile_args = ["-O3", "-std=c++17"]
extra_link_args = []

if sys_name == "Darwin" and mach_name in ("arm64", "aarch64"):
    # macOS Apple Silicon ARM64 (M1/M2/M3/M4/M5)
    extra_compile_args.extend(["-mcpu=native", "-stdlib=libc++"])
elif sys_name == "Linux" and mach_name in ("x86_64", "amd64"):
    # Linux x86_64 (Kaggle / Colab / Cloud GPU instances)
    # NEVER pass -mcpu=native to GCC x86_64!
    extra_compile_args.extend(["-march=native", "-pthread"])
    extra_link_args.extend(["-pthread"])
else:
    # Generic fallback
    extra_compile_args.extend(["-pthread"])
    extra_link_args.extend(["-pthread"])

ext_modules = [
    Pybind11Extension(
        "uno_ataraxos._cpp",
        ["cpp/vec_uno_env.cpp"],
        cxx_std=17,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

setup(
    name="uno-ataraxos",
    version="0.1.0",
    packages=["uno_ataraxos"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
