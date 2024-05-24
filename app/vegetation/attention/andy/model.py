
from torch import nn
import torch
import math



class InputFeature(nn.Module):
    def __init__(self, nTup, nxc, nh):
        super().__init__()
        self.nh = nh
        self.lnXc = nn.Sequential(nn.Linear(nxc, nh), nn.ReLU(), nn.Linear(nh, nh))
        self.lnLst = nn.ModuleList()
        for n in nTup:
            self.lnLst.append(
                nn.Sequential(nn.Linear(n, nh), nn.ReLU(), nn.Linear(nh, nh))
            )

    def getPos(self, pos):
        nh = self.nh
        P = torch.zeros([pos.shape[0], pos.shape[1], nh], dtype=torch.float32)
        for i in range(int(nh / 2)):
            P[:, :, 2 * i] = torch.sin(pos / (i + 1) * torch.pi)
            P[:, :, 2 * i + 1] = torch.cos(pos / (i + 1) * torch.pi)
        return P

    def forward(self, xTup, pTup, xc):
        outLst = list()
        for k in range(len(xTup)):
            x = self.lnLst[k](xTup[k]) + self.getPos(pTup[k])
            outLst.append(x)
        outC = self.lnXc(xc)
        out = torch.cat(outLst + [outC[:, None, :]], dim=1)
        return out


class AttentionLayer(nn.Module):
    def __init__(self, nx, nh):
        super().__init__()
        self.nh = nh
        self.W_k = nn.Linear(nx, nh, bias=False)
        self.W_q = nn.Linear(nx, nh, bias=False)
        self.W_v = nn.Linear(nx, nh, bias=False)
        self.W_o = nn.Linear(nh, nh, bias=False)

    def forward(self, x):
        # pdb.set_trace()
        q, k, v = self.W_q(x), self.W_k(x), self.W_v(x)
        d = q.shape[1]
        score = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(d)
        attention_mask = torch.ones(score.shape)
        attention = torch.softmax(score * attention_mask, dim=-1)
        out = torch.bmm(attention, v)
        out = self.W_o(out)
        return out


class PositionWiseFFN(nn.Module):
    def __init__(self, nh, ny):
        super().__init__()
        self.dense1 = nn.Linear(nh, nh)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(nh, ny)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))


class AddNorm(nn.Module):
    def __init__(self, norm_shape, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(norm_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


class FinalModel(nn.Module):
    def __init__(self, nTup, nxc, nh):
        super().__init__()
        self.nTup = nTup
        self.nxc = nxc
        self.encoder = InputFeature(nTup, nxc, nh)
        self.atten = AttentionLayer(nh, nh)
        self.addnorm1 = AddNorm(nh, DROPOUT)
        self.addnorm2 = AddNorm(nh, DROPOUT)
        self.ffn1 = PositionWiseFFN(nh, nh)
        self.ffn2 = PositionWiseFFN(nh, 1)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, pos, xcT, lTup):
        xIn = self.encoder(x, pos, xcT)
        out = self.atten(xIn)
        out = self.addnorm1(xIn, out)
        out = self.ffn1(out)
        out = self.addnorm2(xIn, out)
        out = self.ffn2(out)
        out = out.squeeze(-1)
        k = 0
        temp = 0
        for i in lTup:
            temp = temp + out[:, k : i + k].mean(-1)
            k = k + i
        temp = temp + out[:, k:].mean(-1)
        return temp