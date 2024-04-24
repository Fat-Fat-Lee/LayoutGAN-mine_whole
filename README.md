# LayoutGAN+X:Sketch-based Generation and Beautification of Front-end Page Lay-out

This repository provides the official code for the paper "LayoutGAN+X:Sketch-based Generation and Beautification of Front-end Page Lay-out".

## Installation

1. Clone this repository

   ```bash
   git clone https://github.com/Fat-Fat-Lee/LayoutGAN-mine_whole.git
   cd LayoutGAN-mine_whole
   ```

2. Create a new [conda](https://docs.conda.io/en/latest/miniconda.html) environment (Python 3.8)

   ```bash
   conda create -n CREATE_NAME python=3.8
   conda activate CREATE_NAME
   ```

3. Install [PyTorch 1.8.1](https://pytorch.org/get-started/previous-versions/#v181) and [PyTorch Geometric 1.7.2](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#installation-via-pip-wheels). An example of the PyG installation command is shown below.

   ```bash
   pip install torch-scatter==2.0.7 -f https://data.pyg.org/whl/torch-1.8.1+cu111.html
   pip install torch-sparse==0.6.10 -f https://data.pyg.org/whl/torch-1.8.1+cu111.html
   pip install torch-geometric==1.7.2
   ```

4. Install the other dependent libraries

   ```bash
   pip install -r requirements.txt
   ```

5. Build Cascade-RCNN using MMDetection

   1. Install MMEngine and MMCV using MIM

      ```bash
      pip install -U openmim
      mim install mmengine
      mim install "mmcv>=2.0.0"
      ```

   2. Install MMDetection

      ```bash
      mim install mmdet
      ```

6. Prepare data

   Download three needed datasets: [RICO](http://www.interactionmining.org/rico.html), [Screen2words](https://github.com/google-research-datasets/screen2words) and [Synz](https://github.com/vinothpandian/synz)

   Preprocess data: run /Screen2words-main/jsonCorrect.py

## Train and inference

To train the layout generation and optimization model, run:

```bash
python train.py --dataset ricoTest --batch_size 64 --iteration 200000 --latent_size 8 --lr 5e-06 --G_d_model 256 --G_nhead 4 --G_num_layers 8 --D_d_model 256 --D_nhead 4 --D_num_layers 8
```

To evaluate the layout generation and optimization model, run:

```bash
python eval.py ricoTest output/generated_layouts_ricoTest.pkl
```

