// Native module surface for vllm_tbccl: the private "tbccl" c10d backend factory plus build/ABI information. No libtbccl class is exposed to Python.
#include <torch/extension.h>
#include <torch/version.h>

#include <pybind11/chrono.h>

#include <tbccl/tbccl.h>

#include "process_group.hpp"

#ifndef VLLM_TBCCL_LINKED_TBCCL_VERSION
#define VLLM_TBCCL_LINKED_TBCCL_VERSION "unknown"
#endif
#ifndef VLLM_TBCCL_WIRE_PROTOCOL
#define VLLM_TBCCL_WIRE_PROTOCOL 0
#endif

namespace
{

std::string runtime_version()
{
    uint32_t major = 0, minor = 0, patch = 0;
    vllm_tbccl::check_call("package version", tbcclGetPackageVersion(&major, &minor, &patch));
    return std::to_string(major) + "." + std::to_string(minor) + "." + std::to_string(patch);
}

// The libtbccl C ABI the headers declared at build time and the one the statically linked library reports at run time.
unsigned built_c_abi_version() { return TBCCL_C_ABI_VERSION; }

unsigned c_abi_version()
{
    uint32_t abi = 0;
    vllm_tbccl::check_call("abi version", tbcclGetAbiVersion(&abi));
    return abi;
}

// Recorded from the installed TBCCL headers by setup.py (the C ABI has no wire-protocol query).
unsigned wire_protocol_version() { return VLLM_TBCCL_WIRE_PROTOCOL; }

std::string built_with_torch()
{
    return std::to_string(TORCH_VERSION_MAJOR) + "." + std::to_string(TORCH_VERSION_MINOR) + "." + std::to_string(TORCH_VERSION_PATCH);
}

pybind11::dict compiled_features()
{
    pybind11::dict d;
    d["cpu"] = true;
#ifdef VLLM_TBCCL_WITH_CUDA
    d["cuda"] = true;
#else
    d["cuda"] = false;
#endif
    return d;
}

// Signature expected by torch.distributed.Backend.register_backend (extended_api=False): (store, rank, world_size, timeout).
c10::intrusive_ptr<c10d::Backend> create_backend(
    const c10::intrusive_ptr<c10d::Store> &store, int rank, int world_size, const std::chrono::duration<float> &timeout)
{
    return c10::make_intrusive<vllm_tbccl::ProcessGroupTBCCL>(
        store, rank, world_size, std::chrono::duration_cast<std::chrono::milliseconds>(timeout));
}

pybind11::dict op_stats()
{
    pybind11::dict d;
    for (const auto &[name, entry] : vllm_tbccl::op_stats())
    {
        pybind11::dict e;
        e["count"] = entry.first;
        e["bytes"] = entry.second;
        d[pybind11::str(name)] = e;
    }
    return d;
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.doc() = "vllm-tbccl native backend (PyTorch c10d <-> libtbccl C ABI)";
    m.def("runtime_version", &runtime_version, "Version of the libtbccl runtime linked into this build");
    m.def("c_abi_version", &c_abi_version, "libtbccl C ABI version reported by the linked library");
    m.def("built_c_abi_version", &built_c_abi_version, "libtbccl C ABI version of the headers this extension was compiled against");
    m.def("wire_protocol_version", &wire_protocol_version, "libtbccl wire-protocol version of the headers this extension was compiled against");
    m.def("built_with_torch", &built_with_torch, "Version of the torch headers this extension was compiled against");
    m.def("compiled_features", &compiled_features, "Devices this build can serve");
    m.def("op_stats", &op_stats, "Per-operation count and bytes submitted to libtbccl by this process");
    m.def("reset_op_stats", &vllm_tbccl::reset_op_stats);
    // GIL released: bootstrap blocks on the Store and the network.
    m.def("create_backend", &create_backend, py::arg("store"), py::arg("rank"), py::arg("world_size"), py::arg("timeout"), py::call_guard<py::gil_scoped_release>());
}
