from setuptools import setup, find_packages

setup(
    classifiers=
    ['Programming Language :: Python :: 3.7+', ],
    name='udl',
    description="unified pytorch framework for vision task",
    author="XiaoXiao-Woo",
    author_email="wxwsx1997@gmail.com",
    url='https://github.com/XiaoXiao-Woo/PanCollection',
    version='0.1',
    packages=find_packages(),
    license='GPLv3',
    python_requires='>=3.7',
    install_requires=[
        "psutil",
        "opencv-python",
        "numpy",
        "matplotlib",
        "tensorboard",
        "addict",
        "yapf",
        "imageio",
        "colorlog",
        "scipy",
        "timm"
    ],
)