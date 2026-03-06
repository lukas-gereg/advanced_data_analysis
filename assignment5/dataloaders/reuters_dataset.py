import torch
import numpy as np
from torch.utils.data import Dataset
from tensorflow.keras.datasets import reuters
from sklearn.utils.class_weight import compute_class_weight

SEED = 1337
MAX_LEN = 256


def load_reuters_texts(num_words=20_000, test_split=0.2, seed=SEED):
    (x_tr, y_tr), (x_te, y_te) = reuters.load_data(num_words=num_words, test_split=test_split, seed=seed)
    word_index = reuters.get_word_index()
    idx2w = {(idx + 3): w for w, idx in word_index.items()}
    idx2w[0], idx2w[1], idx2w[2], idx2w[3] = "<PAD>", "<START>", "<UNK>", "<UNUSED>"

    def decode(seq): return " ".join(idx2w.get(i, "<UNK>") for i in seq)

    texts_tr = [decode(s) for s in x_tr]
    texts_te = [decode(s) for s in x_te]
    num_labels = int(max(y_tr.max(), y_te.max())) + 1

    class_ids = np.arange(num_labels)

    # inverse freq (balanced) then dampen with sqrt and cap the max a bit
    raw_w = compute_class_weight("balanced", classes=class_ids, y=y_tr)
    mild_w = np.sqrt(raw_w)  # soften
    mild_w = np.minimum(mild_w, 3.0)  # cap (tune 2.5–4.0)

    class_weights = torch.tensor(mild_w, dtype=torch.float)
    return (texts_tr, y_tr), (texts_te, y_te), num_labels, class_weights


class ReutersDS(Dataset):
    def __init__(self, texts, labels, tok, max_len=MAX_LEN):
        self.texts, self.labels, self.tok, self.max_len = texts, labels, tok, max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        enc = self.tok(self.texts[i], truncation=True, max_length=self.max_len, padding=False, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(int(self.labels[i]), dtype=torch.long)
        return item