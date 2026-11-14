"""setup.py: the extension module dflimit._engine (the metadata is in pyproject.toml).

The extension is the gateway src/dflimit/_engine.c and the C engine in csrc/ (copied from DF-Limit's c/ by
tools/sync-engine.sh): C11 (dfl_*.c, tables.c) and the C++ of the Faddeeva package (Faddeeva.cc, faddeeva_shim.cpp).
The two languages need different flags (-std=c11 on a C++ file is an error under clang), so build_ext below compiles
the engine's sources itself, each with the flags of its language, and links the objects into the extension, with the
C++ driver (language="c++": the C++ runtime) and -pthread.  The flags are those of DF-Limit's c/build.sh (gcc, clang)
and of .github/workflows/c.yml and mex.yml (MSVC: cl /O2 /std:c11); -DDFL_QUIET compiles out the debugging output of
dfl_gen.c.  The compiler is the one setuptools configures from the running Python (and ARCHFLAGS, CC, CXX, CFLAGS on
macOS and Linux), so the architecture flags of a wheel build apply to the engine as well.

The limited API of CPython 3.9 (Py_LIMITED_API): one abi3 wheel per platform for every CPython from 3.9.  The
free-threaded CPython has no limited API; there the extension is built for the running version only.
"""
import sysconfig

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

CSRC = "csrc"
ENGINE_C = ["dfl_special.c", "dfl_mgf.c", "dfl_single.c", "dfl_gen.c", "dfl_moments.c", "dfl_double.c", "dfl_api.c",
            "dfl_threads.c", "tables.c"]
ENGINE_CXX = ["Faddeeva.cc", "faddeeva_shim.cpp"]
ENGINE_H = ["dflimit.h", "dfl_lock.h", "dfl_vmath.h", "Faddeeva.h"]
MACROS = [("DFL_QUIET", None)]

LIMITED_API = not sysconfig.get_config_var("Py_GIL_DISABLED")


def c_flags(compiler_type, src=None):
    """The flags of the C files (the engine's and the gateway's)."""
    if compiler_type == "msvc":
        return ["/O2", "/std:c11"]
    # -O3 for dfl_gen.c as in c/build.sh: the l-rule of the noncentral kernel is written to be vectorized;
    # -Wno-unused-function as in c/build.sh (after the -Wall of Python's CFLAGS)
    return ["-std=c11", "-O3" if src == "dfl_gen.c" else "-O2", "-pthread", "-Wno-unused-function"]


def cxx_flags(compiler_type):
    """The flags of the C++ files (setuptools adds /EHsc for MSVC)."""
    return ["/O2"] if compiler_type == "msvc" else ["-std=c++11", "-O2"]


class build_ext_engine(build_ext):
    """build_ext that compiles the engine's C and C++ sources with the flags of their language."""

    def build_extension(self, ext):
        ct = self.compiler.compiler_type
        objects = []
        for f in ENGINE_C:
            objects += self.compiler.compile([f"{CSRC}/{f}"], output_dir=self.build_temp, macros=MACROS,
                                             include_dirs=[CSRC], extra_postargs=c_flags(ct, f), debug=self.debug)
        for f in ENGINE_CXX:
            objects += self.compiler.compile([f"{CSRC}/{f}"], output_dir=self.build_temp, macros=MACROS,
                                             include_dirs=[CSRC], extra_postargs=cxx_flags(ct), debug=self.debug)
        ext.extra_objects = objects + list(ext.extra_objects or [])
        ext.extra_compile_args = c_flags(ct) + list(ext.extra_compile_args or [])
        if ct != "msvc":
            # (after the objects on the command line: the engine needs pthreads and libm)
            ext.extra_link_args = list(ext.extra_link_args or []) + ["-pthread", "-lm"]
        super().build_extension(ext)


engine = Extension(
    "dflimit._engine",
    sources=["src/dflimit/_engine.c"],
    include_dirs=[CSRC],
    depends=[f"{CSRC}/{f}" for f in ENGINE_C + ENGINE_CXX + ENGINE_H],
    define_macros=MACROS + ([("Py_LIMITED_API", "0x03090000")] if LIMITED_API else []),
    py_limited_api=LIMITED_API,
    language="c++",
)

setup(
    ext_modules=[engine],
    cmdclass={"build_ext": build_ext_engine},
    options={"bdist_wheel": {"py_limited_api": "cp39"}} if LIMITED_API else {},
)
