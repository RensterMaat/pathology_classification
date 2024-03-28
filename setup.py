from setuptools import find_packages, setup

setup(
    name="source",
    packages=find_packages(),
    version="0.1.0",
    install_requires=[
        "aiohttp==3.8.4",
        "aiosignal==1.3.1",
        "albumentations==1.3.1",
        "appdirs==1.4.4",
        "asttokens==2.2.1",
        "async-timeout==4.0.2",
        "attrs==22.2.0",
        "backcall==0.2.0",
        "backports.functools-lru-cache==1.6.4",
        "black==23.3.0",
        "blackdoc==0.3.8",
        "braceexpand==0.1.7",
        "certifi==2022.12.7",
        "cffi==1.16.0",
        "charset-normalizer==3.1.0",
        "click==8.1.3",
        "cmake==3.26.1",
        "comm==0.1.4",
        "contourpy==1.0.7",
        "cycler==0.11.0",
        "debugpy==1.5.1",
        "decorator==5.1.1",
        "docker-pycreds==0.4.0",
        "einops==0.6.1",
        "et-xmlfile==1.1.0",
        "executing==1.2.0",
        "filelock==3.11.0",
        "fonttools==4.39.3",
        "frozenlist==1.3.3",
        "fsspec==2023.12.2",
        "gitdb==4.0.10",
        "GitPython==3.1.31",
        "h5py==3.8.0",
        "huggingface-hub==0.20.2",
        "idna==3.4",
        "imageio==2.31.1",
        "importlib-metadata==6.6.0",
        "iniconfig==2.0.0",
        "ipykernel==6.15.0",
        "ipython==8.13.1",
        "ipywidgets==8.1.0",
        "jedi==0.18.2",
        "Jinja2==3.1.2",
        "joblib==1.2.0",
        "jupyter_client==8.2.0",
        "jupyter_core==5.3.0",
        "jupyterlab-widgets==3.0.8",
        "kiwisolver==1.4.4",
        "lazy_loader==0.3",
        "lightning-utilities==0.8.0",
        "lit==16.0.0",
        "llvmlite==0.41.1",
        "markdown-it-py==3.0.0",
        "MarkupSafe==2.1.2",
        "matplotlib==3.7.1",
        "matplotlib-inline==0.1.6",
        "mdurl==0.1.2",
        "more-itertools==10.0.0",
        "mpmath==1.3.0",
        "multidict==6.0.4",
        "mypy-extensions==1.0.0",
        "nest-asyncio==1.5.6",
        "networkx==3.1",
        "numba==0.58.1",
        "numpy==1.24.2",
        "nvidia-cublas-cu11==11.10.3.66",
        "nvidia-cuda-cupti-cu11==11.7.101",
        "nvidia-cuda-nvrtc-cu11==11.7.99",
        "nvidia-cuda-runtime-cu11==11.7.99",
        "nvidia-cudnn-cu11==8.5.0.96",
        "nvidia-cufft-cu11==10.9.0.58",
        "nvidia-curand-cu11==10.2.10.91",
        "nvidia-cusolver-cu11==11.4.0.1",
        "nvidia-cusparse-cu11==11.7.4.91",
        "nvidia-nccl-cu11==2.14.3",
        "nvidia-nvtx-cu11==11.7.91",
        "opencv-python==4.8.0.74",
        "opencv-python-headless==4.8.1.78",
        "openpyxl==3.1.2",
        "openslide-python==1.2.0",
        "packaging==23.1",
        "pandas==2.0.0",
        "parso==0.8.3",
        "pathspec==0.11.1",
        "pathtools==0.1.2",
        "patsy==0.5.6",
        "pexpect==4.8.0",
        "pickleshare==0.7.5",
        "Pillow==10.1.0",
        "pip==23.0.1",
        "platformdirs==3.5.0",
        "pluggy==1.0.0",
        "prompt-toolkit==3.0.38",
        "protobuf==4.22.3",
        "psutil==5.9.0",
        "ptyprocess==0.7.0",
        "pure-eval==0.2.2",
        "pycparser==2.21",
        "Pygments==2.15.1",
        "pynndescent==0.5.10",
        "pyparsing==3.0.9",
        "pytest==7.2.2",
        "python-dateutil==2.8.2",
        "pytorch-lightning==2.0.6",
        "pytz==2023.3",
        "PyWavelets==1.4.1",
        "PyYAML==6.0",
        "pyzmq==25.0.2",
        "qudida==0.0.4",
        "regex==2023.12.25",
        "requests==2.28.2",
        "rich==13.5.1",
        "rpy2==3.5.15",
        "safetensors==0.4.1",
        "scikit-image==0.21.0",
        "scikit-learn==1.2.2",
        "scikit-misc==0.2.0",
        "scipy==1.11.1",
        "seaborn==0.12.2",
        "sentry-sdk==1.21.1",
        "setproctitle==1.3.2",
        "setuptools==68.0.0",
        "six==1.16.0",
        "smmap==5.0.0",
        "stack-data==0.6.2",
        "statsmodels==0.14.1",
        "sympy==1.11.1",
        "tableone==0.8.0",
        "tabulate==0.9.0",
        "tbb==2021.10.0",
        "threadpoolctl==3.1.0",
        "tifffile==2023.7.18",
        "tokenizers==0.15.0",
        "toml==0.10.2",
        "tomli==2.0.1",
        "torch==2.0.0",
        "torchmetrics==0.11.4",
        "torchvision==0.15.1",
        "tornado==6.2",
        "tqdm==4.65.0",
        "traitlets==5.9.0",
        "transformers==4.36.2",
        "triton==2.0.0",
        "tuna==0.5.11",
        "typing_extensions==4.5.0",
        "tzdata==2023.3",
        "tzlocal==5.2",
        "urllib3==1.26.15",
        "vision-transformer-pytorch==1.0.3",
        "vulture==2.7",
        "wandb==0.15.4",
        "wcwidth==0.2.6",
        "webdataset==0.2.48",
        "wheel==0.38.4",
        "widgetsnbextension==4.0.8",
        "yarl==1.8.2",
        "zipp==3.15.0",
    ],
)
