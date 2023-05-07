import json
import os
import pickle
from pathlib import Path

import torch
from torch_geometric.data import Data

from data.base import BaseDataset
from data.ricoTest import get_dataset_

if __name__ == '__main__':
    file_path=r".\data\dataset\Synz\results.pkl"#file_path为草图布局
    in_ricoLabel={}#in_ricoLabel是草图布局的功能属性序列，需要由用户进行输入
    #in_ricoLabel={'0_50380.jpg':'Calendar Page','0_50381.jpg':'Calculator Page','0_50470.j[g':'other'}#默认形式

    get_dataset_(file_path=file_path,in_ricoLabel=in_ricoLabel)
