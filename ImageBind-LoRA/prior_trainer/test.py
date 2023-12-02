import torch
from torch.utils.data import IterableDataset

class MyIterableDataset(IterableDataset):
    def __init__(self, data):
        self.data = data

    def __iter__(self):
        for item in self.data:
            yield item

# 데이터 생성
data = [1, 2, 3, 4, 5]
dataset = MyIterableDataset(data)

# DataLoader 사용
data_loader = torch.utils.data.DataLoader(dataset, batch_size=2)

for batch in data_loader:
    print(batch)