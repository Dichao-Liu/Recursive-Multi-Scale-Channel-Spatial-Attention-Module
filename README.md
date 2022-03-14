# Recursive Multi-Scale Channel-Spatial Attention Module (RMCSAM)


Official PyTorch implementation of the Recursive Multi-Scale Channel-Spatial Attention Module (RMCSAM). RMCSAM is a lightweight and insertable attention module designed for the fine-grained image classification (FGIC) task. RMCSAM can be easily embedded into various backbone convolutional neural networks (CNNs) to improve the classification accuracy for the FGIC task. You may check more details in [our paper published in *IEICE Transactions on Information and Systems*](https://www.jstage.jst.go.jp/article/transinf/E105.D/3/E105.D_2021EDP7166/_pdf) if you are interested in our work.

![The overview of RMCSAM.](https://github.com/Dichao-Liu/Recursive-Multi-Scale-Channel-Spatial-Attention-Module/blob/main/IEICE2022_.Attention_Module.jpg)

## Requirements

 - Linux and Windows are supported.
 - Python >= 3.7.0, PyTorch >= 1.4.0
 - Timm = 0.1.26

## Usage

The script `attention_module.py` provides the source code of the proposed RMCSAM. The script `example_insert_into_backbone.py` provides an example of embedding the RMCSAM into a backbone CNN. We recommend training the networks with the tool of [PyTorch Image Models](https://github.com/rwightman/pytorch-image-models).

The folder `example_insert_into_state_of_the_art_PMG` provides the source code of embedding RMCSAM with the [Progressive Multi Granularity (PMG)](https://github.com/PRIS-CV/PMG-Progressive-Multi-Granularity-Training) framework. 

### Citation
 
Please cite our paper if you use RMCSAM in your work.
```
@article{liu2022recursive,
  title={Recursive Multi-Scale Channel-Spatial Attention for Fine-Grained Image Classification},
  author={LIU, Dichao and WANG, Yu and MASE, Kenji and KATO, Jien},
  journal={IEICE TRANSACTIONS on Information and Systems},
  volume={105},
  number={3},
  pages={713--726},
  year={2022},
  publisher={The Institute of Electronics, Information and Communication Engineers}
}

```
