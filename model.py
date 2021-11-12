from sentence_transformers import SentenceTransformer
import torch
from torch import nn

lencoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')#, device = 'cuda')
lencoder.max_seq_len = 512

def get_emb(lyrics):
    return lencoder.encode([lyrics])

class Model(nn.Module):
  def __init__(self):
    super(Model, self).__init__()

    self.encoder = nn.Sequential(nn.Linear(384,256),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(256,100),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(100, 25),
                                 nn.ReLU(),
                                 nn.Linear(25,5))
                                #  nn.LogSoftmax(dim=1))
    
    self.Classifier = nn.Sequential(nn.Linear(20, 5),
                                    nn.ReLU(),
                                    nn.Linear(5, 1))
    

    
  def forward(self, emb, features):
    emb = self.encoder(emb)
    emb = torch.cat((emb,features),1)
    res = self.Classifier(emb)
    return res

