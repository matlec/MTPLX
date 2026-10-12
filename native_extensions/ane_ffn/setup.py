from setuptools import setup

from mlx import extension

if __name__ == "__main__":
    setup(
        name="mtplx_ane_ffn",
        version="0.0.0",
        description="Neural Engine share of the prefill FFN for MTPLX on M1.",
        ext_modules=[extension.CMakeExtension("mtplx_ane_ffn._ext")],
        cmdclass={"build_ext": extension.CMakeBuild},
        packages=["mtplx_ane_ffn"],
        package_data={"mtplx_ane_ffn": ["*.so", "*.dylib"]},
        zip_safe=False,
        python_requires=">=3.11",
    )
