from setuptools import setup 


setup(
        name="tsbrca",
        py_modules=["bin"],
        install_requires=["blobfile>=1.0.5", "einops", "omegaconf"],
)