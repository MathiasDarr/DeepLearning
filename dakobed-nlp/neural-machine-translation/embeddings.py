import torch.nn as nn


class Embeddings(nn.Module):
    def __init__(self, embed_size, vocab):
        """
        Initialize the embedding layers
        Parameters
        ----------
        embed_size
        vocab
        """

        super(Embeddings, self).__init__()
        self.embed_size = embed_size
        self.source = None
        self.target = None

