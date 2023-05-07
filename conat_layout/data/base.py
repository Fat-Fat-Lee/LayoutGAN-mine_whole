import torch
import seaborn as sns
from pathlib import Path
from torch_geometric.data import InMemoryDataset


class BaseDataset(InMemoryDataset):
    labels = []
    ricoLabels=[]#add by ljw 20221103
    _label2index = None
    _index2label = None
    _ricoLabel2index = None#add by ljw 20221103
    _index2ricoLabel = None#add by ljw 20221103
    _colors = None

    def __init__(self, name, split, transform):
        assert split in ['train', 'val', 'test','randn','gen']#by ljw 20230222加入randn数据集，测试const
        super().__init__(f'data/dataset/{name}/', transform)
        idx = self.processed_file_names.index('{}.pt'.format(split))
        self.data, self.slices = torch.load(self.processed_paths[idx])

    @property
    def label2index(self):
        if self._label2index is None:
            self._label2index = dict()
            for idx, label in enumerate(self.labels):
                self._label2index[label] = idx
        return self._label2index


    @property
    def index2label(self):
        if self._index2label is None:
            self._index2label = dict()
            for idx, label in enumerate(self.labels):
                self._index2label[idx] = label
        return self._index2label

    #add by ljw 20221103
    @property
    def ricoLabel2index(self):
        if self._ricoLabel2index is None:
            self._ricoLabel2index = dict()
            for idx, label in enumerate(self.ricoLabels):
                self._ricoLabel2index[label] = idx
        return self._ricoLabel2index
    #add by ljw 20221103
    @property
    def index2ricoLabel(self):
        if self._index2ricoLabel is None:
            self._index2ricoLabel = dict()
            for idx, label in enumerate(self.ricoLabels):
                self._index2ricoLabel[idx] = label
        return self._index2ricoLabel


    @property
    def colors(self):
        if self._colors is None:
            n_colors = self.num_classes
            colors = sns.color_palette('husl', n_colors=n_colors)
            self._colors = [tuple(map(lambda x: int(x * 255), c))
                            for c in colors]
        return self._colors

    @property
    def raw_file_names(self):
        raw_dir = Path(self.raw_dir)
        if not raw_dir.exists():
            return []
        return [p.name for p in raw_dir.iterdir()]

    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt','randn.pt','gen.pt']#by ljw 20230222加入randn数据集，测试const

    def download(self):
        raise FileNotFoundError('See data/README.md to prepare dataset')

    def process(self):
        raise NotImplementedError
