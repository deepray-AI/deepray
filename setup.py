import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deepray",
    version="0.1.0",
    author="Hailin Fu",
    author_email="hailinfufu@outlook.com",
    description="A small example package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fuhailin/deepray",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
