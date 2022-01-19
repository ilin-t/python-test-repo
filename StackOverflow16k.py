import math
import os.path

import numpy as np
import torch
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import pandas as pd
from torch.utils.data import Dataset


class StackOverflow16k(Dataset):

    def __init__(self, root, transform=None):

        self.item_lookup = pd.read_csv("../annotations.csv")
        self.root = root
        self.transform = transform
        self.labels = {0: "csharp", 1: "java", 2: "javascript", 3: "python"}

    def __len__(self):
        dirlist = [item for item in os.listdir(self.root)]
        count = 0
        for label in dirlist:
            count = count + len(os.listdir(os.path.join(self.root, label)))

        return count

    def __getitem__(self, idx):

        # if torch.is_tensor(item):
        #     item = item.toList()
        # label = self.labels[math.floor(int(idx/2000))]
        file_name = os.path.join(self.root, "csharp", str(self.item_lookup.iloc[idx, 1]) + ".txt")
        # txt_name = os.path.join(self.root, item)
        txt_file = open(file_name, "r")
        txt = txt_file.read()
        txt_file.close()
        if self.transform:
            txt = self.transform(txt)

        return txt


if __name__ == '__main__':
    dataset = StackOverflow16k(root="stack_overflow_16k/train")
    df = pd.DataFrame(columns=["post", "label"], data=np.ones(shape=[8000, 2]))
    dirList = [name for name in os.listdir(dataset.root)]

    for key in dataset.labels.keys():
        # print("Dir: " + dir + " Post: " + post)
        df.iloc[key * 2000:(key + 1) * 2000, 0] = range(0, len(os.listdir(
            os.path.join(dataset.root, dataset.labels[key]))))
        df.iloc[key * 2000:(key + 1) * 2000, 1] = dataset.labels[key]

    df["post"] = df["post"].astype(int)
    df.to_csv("annotations.csv")

    # for i in range(10):
    #     sample = dataset["csharp/0.txt"]
    #
    #     print(len([name for name in os.listdir(dataset.root) if os.path.isfile(name)]))
    #
    #     print(i, len(sample))
    #
    #     print("Sample:", dataset.__getitem__("csharp/0.txt"))
