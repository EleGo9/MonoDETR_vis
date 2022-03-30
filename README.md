# MonoDETR: Depth-aware Transformer for Monocular 3D Object Detection
Official implementation of the paper ['MonoDETR: Depth-aware Transformer for Monocular 3D Object Detection'](https://arxiv.org/pdf/2203.13310.pdf).

## Introduction
MonoDETR is the first DETR-based model for monocular 3D detection **without additional depth supervision, anchors or NMS**, which achieves leading performance on KITTI *val* and *test* set. We enable the vanilla transformer in DETR to be depth-aware and enforce the whole detection process guided by depth. In this way, each object estimates its 3D attributes adaptively from the depth-informative regions on the image, not limited by center-around features.
<div align="center">
  <img src="pipeline.jpg"/>
</div>


## Installation
Comming soon!

## Acknowlegment
This repo benefits from the excellent [MonoDLE](https://github.com/xinzhuma/monodle) and [GUPNet](https://github.com/SuperMHP/GUPNet).

## Contact
If you have any question about this project, please feel free to contact zhangrenrui@pjlab.org.cn.
