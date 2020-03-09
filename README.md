# 2IMV10
Visual computing project

## Data

This project uses the [HDR+ Burst Photography Dataset](http://www.hdrplusdata.org/dataset.html).
To download the subset of bursts used in this project, download the [Google Cloud SDK](https://cloud.google.com/sdk/docs/#install_the_latest_cloud_sdk_version) and use the following command:
```
gsutil -m cp -r gs://hdrplusdata/20171106_subset .
```

## Setup (MacOS / Ubuntu)

### Prerequisites

* [LLVM](http://llvm.org/releases/download.html)
* [Halide](https://github.com/halide/Halide)
* [python_bindings](https://github.com/halide/Halide/tree/master/python_bindings)

`pip install -r requirements.txt`
