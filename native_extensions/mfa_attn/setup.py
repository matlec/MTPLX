from setuptools import setup

from mlx import extension


if __name__ == "__main__":
    setup(
        name="mtplx_mfa_attn",
        version="0.0.0",
        description="metal-flash-attention forward for MTPLX on M1 (D=256).",
        ext_modules=[extension.CMakeExtension("mtplx_mfa_attn._ext")],
        cmdclass={"build_ext": extension.CMakeBuild},
        packages=["mtplx_mfa_attn"],
        package_data={"mtplx_mfa_attn": ["*.so", "*.dylib", "*.metallib"]},
        zip_safe=False,
        python_requires=">=3.11",
    )
