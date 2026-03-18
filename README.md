<div align="center">
<h2> S-VGGT: Structure-Aware Subscene Decomposition for Scalable 3D Foundation Models </h2>
<p align="center">
  <a href=""><img src="https://img.shields.io/badge/arXiv-S_VGGT-red?logo=arxiv" alt="Paper PDF (Coming Soon)"></a>
  <a href="/"><img src="https://img.shields.io/badge/Project_Page-SVGGT-yellow" alt="Project Page"></a>
<p align="center">
  Xinze Li<sup>1</sup>, Pengxu Chen<sup>1,2</sup>, Yiyuan Wang<sup>1,3</sup>, 
  Weifeng Su<sup>1</sup>, Wentao Cheng<sup>1†</sup>
</p>
<p align="center">
  <sup>1</sup>Beijing Normal University–Hong Kong Baptist University &nbsp;&nbsp;
  <sup>2</sup>Jilin University &nbsp;&nbsp;
  <sup>3</sup>Hong Kong Baptist University
</p>
<p align="center">
  <sup>†</sup>Corresponding Author
</p>
</div>


## 📰 News

- [Mar 18, 2026] Code Release



## 🔭 Overview

S-VGGT identifies structural redundancy across frames and reorganizes dense scenes into balanced subscenes with a shared reference frame, enabling highly efficient parallel 3D reconstruction with strong acceleration and no loss in fidelity.

<img src="assets/SVGGT.pdf" alt="S-VGGT Overview" width="">


## ⚙️ Environment Setup

First, create a virtual environment using Conda, clone this repository to your local machine, and install the required dependencies.

```bash
conda create -n svggt python=3.10
conda activate svggt
git clone 
cd svggt
pip install -r requirements.txt
```

Next, prepare the ScanNet dataset: http://www.scan-net.org/ScanNet/

We follow the preprocessing method to ScanNet by choosing 50 scenes following [FastVGGT](https://github.com/mystorm16/FastVGGT), where we implement a tool to assist you. Please refer to [Xinze Li: ScanNet-Process-Python3](https://github.com/Powertony102/ScanNet-Process-Python3) for more details.

Finally, configure the dataset path. For example:

```bash
parser.add_argument(
    "--data_dir", type=Path,
    default="/home/jovyan/shared/xinzeli/scannetv2/process_scannet"
)
parser.add_argument(
    "--gt_ply_dir",
    type=Path,
    default="/home/jovyan/shared/xinzeli/scannetv2/scannet",
)
```

Note that, we build our implementation based on [FastVGGT](https://github.com/mystorm16/FastVGGT), where we observe that the default value of `--depth_conf_thresh`  is too high for VGGT output. We suggest to configure it by a value which is lower that 1.0.



## 🤖 Usage

### ScanNet

Evaluate S-VGGT on the ScanNet dataset with various input sequence lengths. 

- The `--merging` parameter specifies the block index at which the merging strategy is applied, which is a parameter from [FastVGGT](https://github.com/mystorm16/FastVGGT). You can leave it with its default value `None` to inference and evaluate S-VGGT only, or you can use `--merging 0` to use token merging from Block 0 to evaluate the orthogonality of S-VGGT and token-level acceleration method.
- The `--num_groups` parameter overrides the number of subscene groups; `None` uses auto selection 



```bash
python eval/eval_scannet.py --input_frame 1000
```

### 7 Scenes & NRGBD
Evaluate across two datasets, sampling keyframes every 3 frames:
```bash
python eval/eval_7andN.py --kf 3
```



## 🍺 Acknowledgements

- Thanks to these great repositories: [VGGT](https://github.com/facebookresearch/vggt), [Dust3r](https://github.com/naver/dust3r),  [Fast3R](https://github.com/facebookresearch/fast3r), [CUT3R](https://github.com/CUT3R/CUT3R), [MV-DUSt3R+](https://github.com/facebookresearch/mvdust3r), [StreamVGGT](https://github.com/wzzheng/StreamVGGT), [VGGT-Long](https://github.com/DengKaiCQ/VGGT-Long),  [FastVGGT](https://github.com/mystorm16/FastVGGT) and many other inspiring works in the community.

- Special thanks to our supervisor [Dr. Wentao Cheng](https://wtchengcv.github.io/) for consistent suggestions and efforts to this work.

## ⚖️ License
See the [LICENSE](./LICENSE.txt) file for details about the license under which this code is made available.

