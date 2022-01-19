import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, vocab

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

