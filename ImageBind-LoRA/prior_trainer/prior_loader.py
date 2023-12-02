

from math import ceil
from clip import tokenize
from embedding_reader import EmbeddingReader
from torch import from_numpy
from torch.utils.data import IterableDataset, DataLoader



# Use this as ShapetalkEmbeddingDataset : Shape, Text embedding pair dataset with generator?
class PriorEmbeddingDataset(IterableDataset):
        

    def __init__(
        self,
        text_conditioned: bool,
        batch_size: int,
        start: int,
        stop: int,
        image_reader,
        text_reader: EmbeddingReader = None,
    ) -> None:
        super(PriorEmbeddingDataset).__init__()

        self.text_conditioned = text_conditioned


    
        pass

    def __iter__(self):
        pass

    
    def __next__(self):
        try:
            return self.get_sample()
        except StopIteration:
            raise StopIteration
    

    def get_sample(self):
        """
        pre-proocess data from either reader into a common format
        """
        if self.text_conditioned:
            image_embedding, caption = next(self.loader)

            image_embedding = from_numpy(image_embedding)
            tokenized_caption = tokenize(caption["caption"].to_list(), truncate=True)

            return image_embedding, tokenized_caption

        else:
            (image_embedding, _), (text_embedding, _) = next(self.loader)

            image_embedding = from_numpy(image_embedding)
            text_embedding = from_numpy(text_embedding)

            return image_embedding, text_embedding