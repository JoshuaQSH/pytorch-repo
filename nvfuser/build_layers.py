import torch
import torch.nn as nn
import numpy as np 

class Net(nn.Module):
    def create_mlp(self, ln):
        layers = nn.ModuleList()
        for i in range(0, ln.size - 1):
            n = ln[i]
            m = ln[i + 1]

            LL = nn.Linear(int(n), int(m), bias=True)

            mean = 0.0
            std_dev = np.sqrt(2 / (m + n))
            W = np.random.normal(mean, std_dev, seize=(m,n)).astype(np.float32)
            std_dev = np.sqrt(1 / m)
            bt = np.random.normal(mean, std_dev, size=m).astype(np.float32)
            LL.weight.data = torch.tensor(W, requires_grad=True)
            LL.bias.data = torch.tensor(bt, requires_grad=True)
            layers.append(LL)
            layers.append(nn.ReLU())

        return torch.nn.Sequential(*layers)

    def create_emb(self, m, ln, weighted_pooling=None):
        emb_l = nn.ModuleList()
        v_W_l = []
        for i in range(0, ln.size):
            if ext_dist.my_size > 1:
                if i not in self.local_emb_indices:
                    continue
            
            n = int(ln[i])

            # construct embedding operator
            #  if self.qr_flag and n > self.qr_threshold:
                #  EE = QREmbeddingBag(
                #      n,
                #      m,
                #      self.qr_collisions,
                #      operation=self.qr_operation,
                #      mode="sum",
                #      sparse=True,
                #  )
            #  #  elif self.md_flag and n > self.md_threshold:
            #      base = max(m)
            #      _m = m[i] if n > self.md_threshold else base
            #      EE = PrEmbeddingBag(n, _m, base)
            #      # use np initialization as below for consistency...
            #      W = np.random.uniform(
            #          low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, _m)
            #      ).astype(np.float32)
            #      EE.embs.weight.data = torch.tensor(W, requires_grad=True)
            #
            #
            #  else:
            #      EE = nn.EmbeddingBag(n, m, mode="sum", sparse=True)
            #      # initialize embeddings
            #      nn.init.uniform_(EE.weight, a=-np.sqrt(1 / n), b=np.sqrt(1 / n))
            #      W = np.random.uniform(
            #          low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)
            #      ).astype(np.float32)
            #
            #      EE.weight.data = torch.tensor(W, requires_grad=True)
#
            EE  = nn.EmbeddingBag(n, m, mode="sum", sparse=True)
            nn.init.uniform_(EE.weight, a=-np.sqrt(1 / n), b=np.sqrt(1 / n))
            W = np.random.uniform(
                    low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)
                ).astype(np.float32)
                
            EE.weight.data = torch.tensor(W, requires_grad=True)

            if weighted_pooling is None:
                v_W_l.append(None)
            else:
                v_W_l.append(torch.ones(n, dtype=torch.float32))
            emb_l.append(EE)
        return emb_l, v_W_l




