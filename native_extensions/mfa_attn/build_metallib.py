#!/usr/bin/env python3
"""Compile MFA's generated Metal source on a toolchain that rejects its
`__asm("air.simdgroup_async_copy...")` declarations: strip the asm labels,
emit LLVM IR, point the calls at the air intrinsics, assemble a metallib."""
import re, subprocess, sys, pathlib
src, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
text = "#include <metal_stdlib>\n" + src.read_text()
text = re.sub(r'\s*__asm\("air\.[^"]+"\)', '', text)
for old, new in [("__metal_simdgroup_async_copy_1d", "mfa_async_copy_1d"),
                 ("__metal_simdgroup_async_copy_2d", "mfa_async_copy_2d"),
                 ("__metal_wait_simdgroup_events", "mfa_wait_events")]:
    text = text.replace(old, new)
stage = out.with_suffix("")
m = stage.with_suffix(".metal"); m.write_text(text)
ll = stage.with_suffix(".ll")
subprocess.run(["xcrun", "-sdk", "macosx", "metal", "-std=metal3.1", "-O3", "-S", "-emit-llvm", str(m), "-o", str(ll)], check=True)
ir = ll.read_text()
def air_name(name, params):
    if "mfa_wait_events" in name:
        return "@air.wait_simdgroup_events"
    dim = "1d" if "async_copy_1d" in name else "2d"
    spaces = re.findall(r"addrspace\((\d)\)", params)
    first, second = spaces[0], spaces[1]
    return f"@air.simdgroup_async_copy_{dim}.p{first}i8.p{second}i8"
mapping = {}
for mm in re.finditer(r"declare [^@]*(@_Z\d+mfa_\w+)\((.*)$", ir, re.M):
    mapping[mm.group(1)] = air_name(mm.group(1), mm.group(2))
for k, v in mapping.items():
    ir = ir.replace(k + "(", v + "(")
# Call-site types must match the declaration's; the rename keeps them identical.
ll.write_text(ir)
air = stage.with_suffix(".air")
subprocess.run(["xcrun", "-sdk", "macosx", "metal", "-c", str(ll), "-o", str(air)], check=True)
subprocess.run(["xcrun", "-sdk", "macosx", "metallib", str(air), "-o", str(out)], check=True)
print(" ".join(f"{k}->{v}" for k, v in mapping.items()), file=sys.stderr)
