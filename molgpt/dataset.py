import torch
from torch.utils.data import Dataset
from .utils import SmilesEnumerator
import numpy as np
import re
import math  # 添加 math

class SmileDataset(Dataset):
    def __init__(self, args, data, content, block_size, aug_prob=0.5, prop=None, scaffold=None, scaffold_maxlen=None):
        chars = sorted(list(set(content)))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d smiles, %d unique characters.' % (data_size, vocab_size))
    
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.max_len = block_size
        self.vocab_size = vocab_size
        self.data = data
        self.prop = prop
        self.sca = scaffold
        self.scaf_max_len = scaffold_maxlen
        self.debug = getattr(args, 'debug', False)
        self.tfm = SmilesEnumerator()
        self.aug_prob = aug_prob
        
        # 🔑 使用与生成脚本完全一致的正则（双反斜杠）
        self.pattern = r"(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        self.regex = re.compile(self.pattern)

    def __len__(self):
        return len(self.data)  # ✅ 直接返回样本数

    def __getitem__(self, idx):
        smiles = self.data[idx].strip()
        
        # Augmentation
        if not self.debug and np.random.uniform() < self.aug_prob:
            aug_smi = self.tfm.randomize_smiles(smiles)
            if aug_smi is not None:
                smiles = aug_smi

        tokens = self.regex.findall(smiles)
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
        else:
            tokens += ['<'] * (self.max_len - len(tokens))
        
        dix = [self.stoi.get(t, self.stoi['<']) for t in tokens]
        x = torch.tensor(dix, dtype=torch.long)
        y = x.clone()  # GPT 通常用 (x, x) 或 (x[:-1], x[1:])，这里保持与 Trainer 一致

        # 🔑 处理 prop（可为 None）
        prop_tensor = None
        if self.prop is not None:
            prop_val = self.prop[idx]
            if not isinstance(prop_val, (list, tuple, np.ndarray)):
                prop_val = [prop_val]
            prop_tensor = torch.tensor(prop_val, dtype=torch.float)

        # 🔑 处理 scaffold（可为 None）
        sca_tensor = None
        if self.sca is not None:
            scaffold = self.sca[idx].strip()
            sca_tokens = self.regex.findall(scaffold)
            if len(sca_tokens) > self.scaf_max_len:
                sca_tokens = sca_tokens[:self.scaf_max_len]
            else:
                sca_tokens += ['<'] * (self.scaf_max_len - len(sca_tokens))
            sca_dix = [self.stoi.get(t, self.stoi['<']) for t in sca_tokens]
            sca_tensor = torch.tensor(sca_dix, dtype=torch.long)

        # 🔑 动态返回：确保 Trainer 能正确解包
        if prop_tensor is not None and sca_tensor is not None:
            return x, y, prop_tensor, sca_tensor
        elif prop_tensor is not None:
            return x, y, prop_tensor
        elif sca_tensor is not None:
            return x, y, sca_tensor
        else:
            return x, y