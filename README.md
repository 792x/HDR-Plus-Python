# HDR Plus Python
Implementation with GUI for desktop of Google's HDR+ in Python using Halide bindings. This repository is provided as is and is not maintained.

For the original paper see [https://www.hdrplusdata.org/hdrplus.pdf](https://www.hdrplusdata.org/hdrplus.pdf)


![Image Banner](https://github.com/792x/HDR-Plus-Python/blob/master/Examples/Banner.png)

![Image GUI](https://github.com/792x/HDR-Plus-Python/blob/master/Examples/empty_gui_v2.png)



## Data
This project uses the [HDR+ Burst Photography Dataset](http://www.hdrplusdata.org/dataset.html).
To download the subset of bursts used in this project, download the [Google Cloud SDK](https://cloud.google.com/sdk/docs/#install_the_latest_cloud_sdk_version) and use the following command:
```
gsutil -m cp -r gs://hdrplusdata/20171106_subset .
```

## Prerequisites
* Linux or MacOS
* [LLVM](http://llvm.org/releases/download.html)
* [Halide](https://github.com/halide/Halide)
* [python_bindings](https://github.com/halide/Halide/tree/master/python_bindings)

`pip install -r requirements.txt`

## Footnote
This project was heavily inspired by [https://github.com/timothybrooks/hdr-plus](https://github.com/timothybrooks/hdr-plus)
