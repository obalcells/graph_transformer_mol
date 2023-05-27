from ogb.lsc import PCQM4Mv2Dataset
from ..features.graph_preprocessing import preprocess_2d_graph

class MyPygPCQM4Mv2Dataset(PygPCQM4Mv2Dataset):
    def download(self):
        super(MyPygPCQM4MDataset, self).download()

    def process(self):
        super(MyPygPCQM4MDataset, self).process()

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        item = self.get(self.indices()[idx])
        item.idx = idx
        return preprocess_2d_graph(item)

if __name__ == "__main__":
    dataset = PygPCQM4Mv2Dataset()
    print(len(dataset))