"""Build vllm_tbccl._C against an INSTALLED TBCCL prefix (TBCCL_ROOT).

Never reaches into the TBCCL source tree: only <TBCCL_ROOT>/include/tbccl
and the installed library are used.
"""
import glob
import os
import re
import sys

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

HERE = os.path.dirname(os.path.abspath(__file__))
SUPPORTED_C_ABI = (1,)
TESTED_WIRE_PROTOCOL = (4,)


def fail(msg):
    sys.exit(f"vllm-tbccl: {msg}")


def check_compatibility(root, inc):
    """Refuse a TBCCL prefix whose C ABI this package does not support (the extension is linked statically, so the headers decide what runs)."""
    hdr = os.path.join(inc, "tbccl", "tbccl.h")
    m = os.path.isfile(hdr) and re.search(r"#\s*define\s+TBCCL_C_ABI_VERSION\s+(\d+)", open(hdr).read())
    if not m:
        fail(f"{hdr} does not define TBCCL_C_ABI_VERSION; {root} is not a TBCCL prefix this package can build against (needs TBCCL >= 0.6.0)")
    abi = int(m.group(1))
    if abi not in SUPPORTED_C_ABI:
        fail(f"the TBCCL prefix {root} has C ABI {abi}; this vllm-tbccl supports C ABI {list(SUPPORTED_C_ABI)}. "
             "Point TBCCL_ROOT at a compatible prefix (or use a vllm-tbccl release that supports this ABI).")
    w = re.search(r"kWireProtocolVersion\s*=\s*(\d+)", open(os.path.join(inc, "tbccl", "rank_directory.hpp")).read())
    if not w:
        fail(f"cannot read the wire protocol version from {inc}/tbccl/rank_directory.hpp")
    if int(w.group(1)) not in TESTED_WIRE_PROTOCOL:
        print(f"vllm-tbccl: warning: wire protocol {w.group(1)} of this prefix has not been tested (tested: {list(TESTED_WIRE_PROTOCOL)})")
    return abi, int(w.group(1))


def find_tbccl():
    root = os.environ.get("TBCCL_ROOT")
    if not root:
        fail("TBCCL_ROOT is not set. Point it at an installed TBCCL prefix, e.g.\n"
             "  TBCCL_ROOT=/path/to/tbccl-install uv pip install -e . --no-build-isolation")
    root = os.path.abspath(root)
    inc = os.path.join(root, "include")
    if not os.path.isfile(os.path.join(inc, "tbccl", "communicator.hpp")):
        fail(f"{inc}/tbccl/communicator.hpp not found; TBCCL_ROOT is not an installed TBCCL prefix")

    _, wire = check_compatibility(root, inc)

    libs = []
    for libdir in ("lib", "lib64"):
        for name in ("libtbccl_c.a", "libtbccl.a"):  # static link order: the C ABI layer, then the core it wraps
            p = os.path.join(root, libdir, name)
            if os.path.isfile(p):
                libs.append(p)
        if len(libs) == 2:
            break
        libs = []
    if not libs:
        fail(f"libtbccl_c.a and libtbccl.a not both found under {root}/lib or lib64 (vllm-tbccl links the static libraries)")

    version = "unknown"
    for f in glob.glob(os.path.join(root, "lib*", "cmake", "TBCCL", "TBCCLConfigVersion.cmake")):
        m = re.search(r'set\(PACKAGE_VERSION "([^"]+)"\)', open(f).read())
        if m:
            version = m.group(1)
    cuda_lib = None
    for libdir in ("lib", "lib64"):
        p = os.path.join(root, libdir, "libtbccl_cuda.a")
        if os.path.isfile(p):
            cuda_lib = p
    return root, inc, libs, version, cuda_lib, wire


def find_cudart():
    """Headers from a CUDA toolkit (CUDA_HOME); the runtime library from the copy torch
    itself ships/loads when present, so there is exactly one cudart in the process."""
    import torch
    from torch.utils.cpp_extension import CUDA_HOME

    if torch.version.cuda is None:
        return None
    site = os.path.dirname(os.path.dirname(torch.__file__))
    pip_inc = pip_lib = None
    for inc in sorted(glob.glob(os.path.join(site, "nvidia", "cu*", "include"))):
        libdir = os.path.join(os.path.dirname(inc), "lib")
        if glob.glob(os.path.join(libdir, "libcudart.so*")):
            pip_inc, pip_lib = inc, libdir
    inc = None
    for cand in (os.path.join(CUDA_HOME, "include") if CUDA_HOME else None, pip_inc):
        if cand and os.path.isfile(os.path.join(cand, "crt", "host_defines.h")):
            inc = cand
            break
    libdir = pip_lib or (os.path.join(CUDA_HOME, "lib64") if CUDA_HOME else None)
    return (inc, libdir) if inc and libdir else None


# Metadata-only commands (sdist, egg_info, dist_info) need the source list but no TBCCL: the sdist can be produced anywhere and built where a prefix exists.
METADATA_ONLY = any(c in sys.argv for c in ("sdist", "egg_info", "dist_info"))


def configure():
    root, inc, libs, tbccl_version, cuda_lib, wire_protocol = find_tbccl()
    print("vllm-tbccl: using TBCCL %s from %s (%s)" % (tbccl_version, root, ", ".join(os.path.basename(l) for l in libs)))

    include_dirs = [inc]
    library_dirs = []
    libraries = []
    objects = list(libs)  # C ABI layer first, then the core it wraps
    macros = [("VLLM_TBCCL_LINKED_TBCCL_VERSION", f'"{tbccl_version}"'), ("VLLM_TBCCL_WIRE_PROTOCOL", str(wire_protocol))]
    link_args = ["-pthread"]

    cudart = find_cudart() if cuda_lib else None
    if cuda_lib and cudart:
        cuda_inc, cuda_libdir = cudart
        print(f"vllm-tbccl: CUDA enabled (tbccl_cuda + cudart from {cuda_libdir})")
        include_dirs.append(cuda_inc)
        library_dirs.append(cuda_libdir)
        cudart_so = sorted(glob.glob(os.path.join(cuda_libdir, "libcudart.so.*")))[0]
        objects = [libs[0], cuda_lib, *libs[1:]]  # tbccl_cuda depends on the core
        libraries += ["c10_cuda", "torch_cuda"]
        # The runtime path must not name the build directory: relative to the installed package when cudart comes from the pip nvidia/ tree next to it
        # (the layout every torch wheel uses), else the absolute system CUDA directory. The extension is linked by distutils with an argument list, no shell, so `$ORIGIN` is literal.
        site = os.path.dirname(os.path.dirname(__import__("torch").__file__))
        if os.path.commonpath([cuda_libdir, site]) == site:
            rpath = "$ORIGIN/" + os.path.relpath(cuda_libdir, os.path.join(site, "vllm_tbccl"))
        else:
            rpath = cuda_libdir
        link_args += [f"-Wl,-rpath,{rpath}", cudart_so]
        macros.append(("VLLM_TBCCL_WITH_CUDA", "1"))
    else:
        print("vllm-tbccl: CUDA disabled (needs TBCCL's tbccl_cuda component and a CUDA-enabled torch)")

    sanitize = os.environ.get("VLLM_TBCCL_SANITIZE")  # e.g. "address,undefined" or "thread"
    compile_args = ["-O1", "-g", "-fno-omit-frame-pointer", f"-fsanitize={sanitize}"] if sanitize else ["-O2"]
    # Keep the build machine's directory layout out of the binary (assert/__FILE__ strings): sources, the torch headers and the TBCCL prefix get neutral prefixes.
    _torch_dir = os.path.dirname(os.path.abspath(__import__("torch").__file__))
    compile_args += [f"-ffile-prefix-map={HERE}=vllm-tbccl", f"-ffile-prefix-map={_torch_dir}=torch", f"-ffile-prefix-map={root}=tbccl-prefix",
                     f"-ffile-prefix-map={__import__('sysconfig').get_paths()['include']}=python"]
    if sanitize:
        link_args.append(f"-fsanitize={sanitize}")
        print(f"vllm-tbccl: sanitizer build ({sanitize})")

    if sys.platform == "darwin":
        link_args.append("-Wl,-S")  # no debug map: ld would record the absolute path of every object file in the build directory

    return dict(include_dirs=include_dirs, library_dirs=library_dirs, libraries=libraries, extra_objects=objects, define_macros=macros,
                extra_compile_args=compile_args + ["-Wall"], extra_link_args=link_args)


sources = sorted(glob.glob(os.path.join("csrc", "*.cpp")))
ext = CppExtension(name="vllm_tbccl._C", sources=sources, **({} if METADATA_ONLY else configure()))

setup(ext_modules=[ext], cmdclass={"build_ext": BuildExtension})
