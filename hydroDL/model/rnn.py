import math
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from .dropout import DropMask, createMask
from . import cnn


class LSTMcell_untied(torch.nn.Module):
    def __init__(self,
                 *,
                 inputSize,
                 hiddenSize,
                 train=True,
                 dr=0.5,
                 drMethod='gal+sem',
                 gpu=0):
        super(LSTMcell_untied, self).__init__()
        self.inputSize = inputSize
        self.hiddenSize = inputSize
        self.dr = dr

        self.w_xi = Parameter(torch.Tensor(hiddenSize, inputSize))
        self.w_xf = Parameter(torch.Tensor(hiddenSize, inputSize))
        self.w_xo = Parameter(torch.Tensor(hiddenSize, inputSize))
        self.w_xc = Parameter(torch.Tensor(hiddenSize, inputSize))

        self.w_hi = Parameter(torch.Tensor(hiddenSize, hiddenSize))
        self.w_hf = Parameter(torch.Tensor(hiddenSize, hiddenSize))
        self.w_ho = Parameter(torch.Tensor(hiddenSize, hiddenSize))
        self.w_hc = Parameter(torch.Tensor(hiddenSize, hiddenSize))

        self.b_i = Parameter(torch.Tensor(hiddenSize))
        self.b_f = Parameter(torch.Tensor(hiddenSize))
        self.b_o = Parameter(torch.Tensor(hiddenSize))
        self.b_c = Parameter(torch.Tensor(hiddenSize))

        self.drMethod = drMethod.split('+')
        self.gpu = gpu
        self.train = train
        if gpu >= 0:
            self = self.cuda(gpu)
            self.is_cuda = True
        else:
            self.is_cuda = False
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hiddenSize)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def init_mask(self, x, h, c):
        self.maskX_i = createMask(x, self.dr)
        self.maskX_f = createMask(x, self.dr)
        self.maskX_c = createMask(x, self.dr)
        self.maskX_o = createMask(x, self.dr)

        self.maskH_i = createMask(h, self.dr)
        self.maskH_f = createMask(h, self.dr)
        self.maskH_c = createMask(h, self.dr)
        self.maskH_o = createMask(h, self.dr)

        self.maskC = createMask(c, self.dr)

        self.maskW_xi = createMask(self.w_xi, self.dr)
        self.maskW_xf = createMask(self.w_xf, self.dr)
        self.maskW_xc = createMask(self.w_xc, self.dr)
        self.maskW_xo = createMask(self.w_xo, self.dr)
        self.maskW_hi = createMask(self.w_hi, self.dr)
        self.maskW_hf = createMask(self.w_hf, self.dr)
        self.maskW_hc = createMask(self.w_hc, self.dr)
        self.maskW_ho = createMask(self.w_ho, self.dr)

    def forward(self, x, hidden):
        h0, c0 = hidden
        doDrop = self.training and self.dr > 0.0

        if doDrop:
            self.init_mask(x, h0, c0)

        if doDrop and 'drH' in self.drMethod:
            h0_i = h0.mul(self.maskH_i)
            h0_f = h0.mul(self.maskH_f)
            h0_c = h0.mul(self.maskH_c)
            h0_o = h0.mul(self.maskH_o)
        else:
            h0_i = h0
            h0_f = h0
            h0_c = h0
            h0_o = h0

        if doDrop and 'drX' in self.drMethod:
            x_i = x.mul(self.maskX_i)
            x_f = x.mul(self.maskX_f)
            x_c = x.mul(self.maskX_c)
            x_o = x.mul(self.maskX_o)
        else:
            x_i = x
            x_f = x
            x_c = x
            x_o = x

        if doDrop and 'drW' in self.drMethod:
            w_xi = self.w_xi.mul(self.maskW_xi)
            w_xf = self.w_xf.mul(self.maskW_xf)
            w_xc = self.w_xc.mul(self.maskW_xc)
            w_xo = self.w_xo.mul(self.maskW_xo)
            w_hi = self.w_hi.mul(self.maskW_hi)
            w_hf = self.w_hf.mul(self.maskW_hf)
            w_hc = self.w_hc.mul(self.maskW_hc)
            w_ho = self.w_ho.mul(self.maskW_ho)
        else:
            w_xi = self.w_xi
            w_xf = self.w_xf
            w_xc = self.w_xc
            w_xo = self.w_xo
            w_hi = self.w_hi
            w_hf = self.w_hf
            w_hc = self.w_hc
            w_ho = self.w_ho

        gate_i = F.linear(x_i, w_xi) + F.linear(h0_i, w_hi) + self.b_i
        gate_f = F.linear(x_f, w_xf) + F.linear(h0_f, w_hf) + self.b_f
        gate_c = F.linear(x_c, w_xc) + F.linear(h0_c, w_hc) + self.b_c
        gate_o = F.linear(x_o, w_xo) + F.linear(h0_o, w_ho) + self.b_o

        gate_i = F.sigmoid(gate_i)
        gate_f = F.sigmoid(gate_f)
        gate_c = F.tanh(gate_c)
        gate_o = F.sigmoid(gate_o)

        if doDrop and 'drC' in self.drMethod:
            gate_c = gate_c.mul(self.maskC)

        c1 = (gate_f * c0) + (gate_i * gate_c)
        h1 = gate_o * F.tanh(c1)

        return h1, c1


class LSTMcell_tied(torch.nn.Module):
    def __init__(self,
                 *,
                 inputSize,
                 hiddenSize,
                 mode='train',
                 dr=0.5,
                 drMethod='drX+drW+drC',
                 gpu=1):
        super(LSTMcell_tied, self).__init__()

        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.dr = dr

        self.w_ih = Parameter(torch.Tensor(hiddenSize * 4, inputSize))
        self.w_hh = Parameter(torch.Tensor(hiddenSize * 4, hiddenSize))
        self.b_ih = Parameter(torch.Tensor(hiddenSize * 4))
        self.b_hh = Parameter(torch.Tensor(hiddenSize * 4))

        self.drMethod = drMethod.split('+')
        self.gpu = gpu
        self.mode = mode
        if mode == 'train':
            self.train(mode=True)
        elif mode == 'test':
            self.train(mode=False)
        elif mode == 'drMC':
            self.train(mode=False)

        if gpu >= 0:
            self = self.cuda()
            self.is_cuda = True
        else:
            self.is_cuda = False
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hiddenSize)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def reset_mask(self, x, h, c):
        self.maskX = createMask(x, self.dr)
        self.maskH = createMask(h, self.dr)
        self.maskC = createMask(c, self.dr)
        self.maskW_ih = createMask(self.w_ih, self.dr)
        self.maskW_hh = createMask(self.w_hh, self.dr)

    def forward(self, x, hidden, *, resetMask=True, doDropMC=False):
        if self.dr > 0 and (doDropMC is True or self.training is True):
            doDrop = True
        else:
            doDrop = False

        batchSize = x.size(0)
        h0, c0 = hidden
        if h0 is None:
            h0 = x.new_zeros(batchSize, self.hiddenSize, requires_grad=False)
        if c0 is None:
            c0 = x.new_zeros(batchSize, self.hiddenSize, requires_grad=False)

        if self.dr > 0 and self.training is True and resetMask is True:
            self.reset_mask(x, h0, c0)

        if doDrop is True and 'drH' in self.drMethod:
            h0 = DropMask.apply(h0, self.maskH, True)

        if doDrop is True and 'drX' in self.drMethod:
            x = DropMask.apply(x, self.maskX, True)

        if doDrop is True and 'drW' in self.drMethod:
            w_ih = DropMask.apply(self.w_ih, self.maskW_ih, True)
            w_hh = DropMask.apply(self.w_hh, self.maskW_hh, True)
        else:
            # self.w are parameters, while w are not
            w_ih = self.w_ih
            w_hh = self.w_hh

        gates = F.linear(x, w_ih, self.b_ih) + \
            F.linear(h0, w_hh, self.b_hh)
        gate_i, gate_f, gate_c, gate_o = gates.chunk(4, 1)

        gate_i = torch.sigmoid(gate_i)
        gate_f = torch.sigmoid(gate_f)
        gate_c = torch.tanh(gate_c)
        gate_o = torch.sigmoid(gate_o)

        if self.training is True and 'drC' in self.drMethod:
            gate_c = gate_c.mul(self.maskC)

        c1 = (gate_f * c0) + (gate_i * gate_c)
        h1 = gate_o * torch.tanh(c1)

        return h1, c1


class CudnnLstm(torch.nn.Module):
    def __init__(self, *, inputSize, hiddenSize, dr=0.5, drMethod='drW',
                 gpu=0):
        super(CudnnLstm, self).__init__()
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.dr = dr
        self.w_ih = Parameter(torch.Tensor(hiddenSize * 4, inputSize))
        self.w_hh = Parameter(torch.Tensor(hiddenSize * 4, hiddenSize))
        self.b_ih = Parameter(torch.Tensor(hiddenSize * 4))
        self.b_hh = Parameter(torch.Tensor(hiddenSize * 4))
        self._all_weights = [['w_ih', 'w_hh', 'b_ih', 'b_hh']]
        self.cuda()

        self.reset_mask()
        self.reset_parameters()

    def _apply(self, fn):
        ret = super(CudnnLstm, self)._apply(fn)
        return ret

    def __setstate__(self, d):
        super(CudnnLstm, self).__setstate__(d)
        self.__dict__.setdefault('_data_ptrs', [])
        if 'all_weights' in d:
            self._all_weights = d['all_weights']
        if isinstance(self._all_weights[0][0], str):
            return
        self._all_weights = [['w_ih', 'w_hh', 'b_ih', 'b_hh']]

    def reset_mask(self):
        self.maskW_ih = createMask(self.w_ih, self.dr)
        self.maskW_hh = createMask(self.w_hh, self.dr)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hiddenSize)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hx=None, cx=None, doDropMC=False):
        if self.dr > 0 and (doDropMC is True or self.training is True):
            doDrop = True
        else:
            doDrop = False

        batchSize = input.size(1)

        if hx is None:
            hx = input.new_zeros(
                1, batchSize, self.hiddenSize, requires_grad=False)
        if cx is None:
            cx = input.new_zeros(
                1, batchSize, self.hiddenSize, requires_grad=False)

        # cuDNN backend - disabled flat weight
        handle = torch.backends.cudnn.get_handle()
        if doDrop is True:
            self.reset_mask()
            weight = [
                DropMask.apply(self.w_ih, self.maskW_ih, True),
                DropMask.apply(self.w_hh, self.maskW_hh, True), self.b_ih,
                self.b_hh
            ]
        else:
            weight = [self.w_ih, self.w_hh, self.b_ih, self.b_hh]

        output, hy, cy, reserve, new_weight_buf = torch._cudnn_rnn(
            input, weight, 4, None, hx, cx, torch.backends.cudnn.CUDNN_LSTM,
            self.hiddenSize, 1, False, 0, self.training, False, (), None)

        return output, (hy, cy)

    @property
    def all_weights(self):
        return [[getattr(self, weight) for weight in weights]
                for weights in self._all_weights]


class CudnnLstmModel(torch.nn.Module):
    def __init__(self, *, nx, ny, hiddenSize, dr=0.5):
        super(CudnnLstmModel, self).__init__()
        self.nx = nx
        self.ny = ny
        self.hiddenSize = hiddenSize
        self.ct = 0
        self.nLayer = 1
        self.linearIn = torch.nn.Linear(nx, hiddenSize)
        self.lstm = CudnnLstm(
            inputSize=hiddenSize, hiddenSize=hiddenSize, dr=dr)
        self.linearOut = torch.nn.Linear(hiddenSize, ny)
        self.gpu = 1

    def forward(self, x, doDropMC=False):
        x0 = F.relu(self.linearIn(x))
        outLSTM, (hn, cn) = self.lstm(x0, doDropMC=doDropMC)
        out = self.linearOut(outLSTM)
        # out = rho/time * batchsize * Ntargetvar
        return out


class LstmCloseModel(torch.nn.Module):
    def __init__(self, *, nx, ny, hiddenSize, dr=0.5, fillObs=True):
        super(LstmCloseModel, self).__init__()
        self.nx = nx
        self.ny = ny
        self.hiddenSize = hiddenSize
        self.ct = 0
        self.nLayer = 1
        self.linearIn = torch.nn.Linear(nx + 1, hiddenSize)
        # self.lstm = CudnnLstm(
        #     inputSize=hiddenSize, hiddenSize=hiddenSize, dr=dr)
        self.lstm = LSTMcell_tied(
            inputSize=hiddenSize, hiddenSize=hiddenSize, dr=dr, drMethod='drW')
        self.linearOut = torch.nn.Linear(hiddenSize, ny)
        self.gpu = 1
        self.fillObs = fillObs

    def forward(self, x, y=None):
        nt, ngrid, nx = x.shape
        yt = torch.zeros(ngrid, 1).cuda()
        out = torch.zeros(nt, ngrid, self.ny).cuda()
        ht = None
        ct = None
        resetMask = True
        for t in range(nt):
            if self.fillObs is True:
                ytObs = y[t, :, :]
                mask = ytObs == ytObs
                yt[mask] = ytObs[mask]
            xt = torch.cat((x[t, :, :], yt), 1)
            x0 = F.relu(self.linearIn(xt))
            ht, ct = self.lstm(x0, hidden=(ht, ct), resetMask=resetMask)
            yTemp = self.linearOut(ht)
            yt = yTemp[:, 0]
            resetMask = False
            out[t, :, :] = yTemp
        return out


class AnnModel(torch.nn.Module):
    def __init__(self, *, nx, ny, hiddenSize):
        super(AnnModel, self).__init__()
        self.hiddenSize = hiddenSize
        self.i2h = nn.Linear(nx, hiddenSize)
        self.h2h = nn.Linear(hiddenSize, hiddenSize)
        self.h2o = nn.Linear(hiddenSize, ny)
        self.ny = ny

    def forward(self, x, y=None):
        nt, ngrid, nx = x.shape
        yt = torch.zeros(ngrid, 1).cuda()
        out = torch.zeros(nt, ngrid, self.ny).cuda()
        for t in range(nt):
            xt = x[t, :, :]
            ht = F.relu(self.i2h(xt))
            ht2 = self.h2h(ht)
            yt = self.h2o(ht2)
            out[t, :, :] = yt
        return out


class AnnCloseModel(torch.nn.Module):
    def __init__(self, *, nx, ny, hiddenSize, fillObs=True):
        super(AnnCloseModel, self).__init__()
        self.hiddenSize = hiddenSize
        self.i2h = nn.Linear(nx + 1, hiddenSize)
        self.h2h = nn.Linear(hiddenSize, hiddenSize)
        self.h2o = nn.Linear(hiddenSize, ny)
        self.fillObs = fillObs
        self.ny = ny

    def forward(self, x, y=None):
        nt, ngrid, nx = x.shape
        yt = torch.zeros(ngrid, 1).cuda()
        out = torch.zeros(nt, ngrid, self.ny).cuda()
        for t in range(nt):
            if self.fillObs is True:
                ytObs = y[t, :, :]
                mask = ytObs == ytObs
                yt[mask] = ytObs[mask]
            xt = torch.cat((x[t, :, :], yt), 1)
            ht = F.relu(self.i2h(xt))
            ht2 = self.h2h(ht)
            yt = self.h2o(ht2)
            out[t, :, :] = yt
        return out


class LstmCnnCond(nn.Module):
    def __init__(self,
                 *,
                 nx,
                 ny,
                 ct,
                 opt=1,
                 hiddenSize=64,
                 cnnSize=32,
                 cp1=(64, 3, 2),
                 cp2=(128, 5, 2),
                 dr=0.5):
        super(LstmCnnCond, self).__init__()

        # opt == 1: cnn output as initial state of LSTM (h0)
        # opt == 2: cnn output as additional output of LSTM
        # opt == 3: cnn output as constant input of LSTM

        if opt == 1:
            cnnSize = hiddenSize

        self.nx = nx
        self.ny = ny
        self.ct = ct
        self.ctRm = False
        self.hiddenSize = hiddenSize
        self.opt = opt

        self.cnn = cnn.Cnn1d(nx=nx, nt=ct, cnnSize=cnnSize, cp1=cp1, cp2=cp2)

        self.lstm = CudnnLstm(
            inputSize=hiddenSize, hiddenSize=hiddenSize, dr=dr)
        if opt == 3:
            self.linearIn = torch.nn.Linear(nx + cnnSize, hiddenSize)
        else:
            self.linearIn = torch.nn.Linear(nx, hiddenSize)
        if opt == 2:
            self.linearOut = torch.nn.Linear(hiddenSize + cnnSize, ny)
        else:
            self.linearOut = torch.nn.Linear(hiddenSize, ny)

    def forward(self, x, xc):
        # x- [nt,ngrid,nx]
        x1 = xc
        x1 = self.cnn(x1)
        x2 = x
        if self.opt == 1:
            x2 = F.relu(self.linearIn(x2))
            x2, (hn, cn) = self.lstm(x2, hx=x1[None, :, :])
            x2 = self.linearOut(x2)
        elif self.opt == 2:
            x1 = x1[None, :, :].repeat(x2.shape[0], 1, 1)
            x2 = F.relu(self.linearIn(x2))
            x2, (hn, cn) = self.lstm(x2)
            x2 = self.linearOut(torch.cat([x2, x1], 2))
        elif self.opt == 3:
            x1 = x1[None, :, :].repeat(x2.shape[0], 1, 1)
            x2 = torch.cat([x2, x1], 2)
            x2 = F.relu(self.linearIn(x2))
            x2, (hn, cn) = self.lstm(x2)
            x2 = self.linearOut(x2)

        return x2


class LstmCnnForcast(nn.Module):
    def __init__(self,
                 *,
                 nx,
                 ny,
                 ct,
                 opt=1,
                 hiddenSize=64,
                 cnnSize=32,
                 cp1=(64, 3, 2),
                 cp2=(128, 5, 2),
                 dr=0.5):
        super(LstmCnnForcast, self).__init__()

        if opt == 1:
            cnnSize = hiddenSize

        self.nx = nx
        self.ny = ny
        self.ct = ct
        self.ctRm = True
        self.hiddenSize = hiddenSize
        self.opt = opt
        self.cnnSize = cnnSize

        if opt == 1:
            self.cnn = cnn.Cnn1d(
                nx=nx + 1, nt=ct, cnnSize=cnnSize, cp1=cp1, cp2=cp2)
        if opt == 2:
            self.cnn = cnn.Cnn1d(
                nx=1, nt=ct, cnnSize=cnnSize, cp1=cp1, cp2=cp2)

        self.lstm = CudnnLstm(
            inputSize=hiddenSize, hiddenSize=hiddenSize, dr=dr)
        self.linearIn = torch.nn.Linear(nx + cnnSize, hiddenSize)
        self.linearOut = torch.nn.Linear(hiddenSize, ny)

    def forward(self, x, y):
        # x- [nt,ngrid,nx]
        nt, ngrid, nx = x.shape
        ct = self.ct
        pt = nt - ct

        if self.opt == 1:
            x1 = torch.cat((y, x), dim=2)
        elif self.opt == 2:
            x1 = y

        x1out = torch.zeros([pt, ngrid, self.cnnSize]).cuda()
        for k in range(pt):
            x1out[k, :, :] = self.cnn(x1[k:k + ct, :, :])

        x2 = x[ct:nt, :, :]
        x2 = torch.cat([x2, x1out], 2)
        x2 = F.relu(self.linearIn(x2))
        x2, (hn, cn) = self.lstm(x2)
        x2 = self.linearOut(x2)

        return x2


# class RNN(nn.Module):
#     def __init__(self, *, nx, nh):
#         super(RNN, self).__init__()
#         self.nh = nh
#         self.i2h = F.sigmoid(nn.Linear(nh + nh, nh))
#         self.h2o = F.relu(nn.Linear(nh, 1))


class modelTest(nn.Module):
    def __init__(self, *, nx, nh, rho):
        super(modelTest, self).__init__()
        self.rho = rho
        self.nh = nh
        self.x2i = nn.Linear(nx, nh)
        self.i2h = nn.Linear(nh + nh, nh)
        # self.h2h = nn.Linear(nh + nh, nh)
        self.h2s = nn.Linear(nh, rho)
        self.h2o = nn.Linear(nh, 2)
        # self.h2o = nn.Linear(nh, 1)

    def forward(self, x):
        nt, ns, nx = x.shape
        nh = self.nh
        rho = self.rho
        rhoF = nt-rho
        h1 = torch.zeros(ns, nh)
        zO = torch.zeros(rhoF, ns, 1)
        if self.training is False:
            sO = torch.zeros(rhoF, ns, rho)
            bO = torch.zeros(rhoF, ns, 2)
        if next(self.parameters()).is_cuda:
            h1 = h1.cuda()
            zO = zO.cuda()
            if self.training is False:
                bO = bO.cuda()
                sO = sO.cuda()
        for k in range(rho):
            hx = self.x2i(x[k, :])
            h1 = self.i2h(torch.cat((hx, h1), 1))
            h1 = F.sigmoid(h1)
            # h2 = self.i2h(torch.cat((hx, h1), 1))
            # h2 = F.sigmoid(h2)
        for k in range(rhoF):
            hx = self.x2i(x[k+rho, :])
            h1 = self.i2h(torch.cat((hx, h1), 1))
            h1 = F.sigmoid(h1)
            # h2 = self.i2h(torch.cat((hx, h1), 1))
            # h2 = F.sigmoid(h2)
            s = self.h2s(h1)
            s = F.tanh(s)
            z1 = x[k:k+rho, :, 0].transpose(0, 1).mul(s).sum(dim=1)
            b = self.h2o(h1)
            b = F.sigmoid(b)
            zO[k, :, 0] = z1.mul(b[:, 1])+b[:, 0]
            # zO[k, :, 0] = z1+b[:, 0]
            if self.training is False:
                sO[k, :, :] = s
                bO[k, :, :] = b
        if self.training:
            return zO
        else:
            return zO, sO, bO


class AgeLSTM(torch.nn.Module):
    def __init__(self, *, nx, ny, nyc, nh, dr=0.5):
        super(AgeLSTM, self).__init__()
        self.nx = nx
        self.ny = ny
        self.nyc = nyc
        self.nh = nh
        self.nLayer = 1
        self.linearIn = torch.nn.Linear(nx, nh)
        self.lstm = CudnnLstm(
            inputSize=nh, hiddenSize=nh, dr=dr)
        self.linearOut = torch.nn.Linear(nh, 2)
        self.b = Parameter(torch.zeros(3, nyc, requires_grad=True).float())

    def forward(self, x, doDropMC=False):
        nt, ns, nx = x.shape
        ny = self.ny
        nyc = self.nyc
        b = self.b
        nh = self.nh
        x0 = self.linearIn(x)
        outLSTM, (hn, cn) = self.lstm(x0, doDropMC=doDropMC)
        outLSTM = F.relu(outLSTM)
        y = outLSTM.sum(dim=-1)
        t = torch.arange(nh).float()/nh
        if self.training:
            yc = torch.zeros([ns, nyc]).float()
            hh = outLSTM[-1, :, :]
        else:
            yc = torch.zeros([nt, ns, nyc]).float()
            hh = outLSTM
        if next(self.parameters()).is_cuda:
            yc = yc.cuda()
            b = b.cuda()
            t = t.cuda()
        for k in range(nyc):
            c0 = b[0, k]
            c1 = b[1, k]
            r = 10**b[2, k]
            gate = c0 * torch.exp(-r*t)*r +\
                c1*(1-torch.exp(-r*t))
            if self.training:
                yc[:, k] = hh.mul(gate).sum(dim=1)
            else:
                yc[:, :, k] = hh.mul(gate).sum(dim=2)
        if self.training:
            return y[:, :, None], yc
        else:
            return y[:, :, None], yc, hh


class AgeLSTM2(torch.nn.Module):
    def __init__(self, *, nx, ny, nyc, nh, rho, dr=0.5):
        super(AgeLSTM2, self).__init__()
        self.nx = nx
        self.ny = ny
        self.nyc = nyc
        self.nh = nh
        self.rho = rho
        self.nLayer = 1
        self.linearIn = torch.nn.Linear(nx, nh)
        self.lstm = CudnnLstm(
            inputSize=nh, hiddenSize=nh, dr=dr)
        self.linearOut = torch.nn.Linear(nh, rho)
        self.linearB = torch.nn.Linear(nh, 2)
        self.r = Parameter(torch.zeros(3, nyc, requires_grad=True).float())

    def forward(self, x, doDropMC=False):
        nt, ns, nx = x.shape
        ny = self.ny
        nyc = self.nyc
        r = self.r
        nh = self.nh
        rho = self.rho
        x0 = self.linearIn(x)
        outLSTM, (hn, cn) = self.lstm(x0)
        t = torch.arange(rho).flip(0).float()/nh
        if self.training:
            gate = F.relu(self.linearOut(hn)[0, :, :])
            b = F.relu(self.linearB(hn)[0, :, :])
            out = torch.zeros([ns, ny+nyc]).float()
            if next(self.parameters()).is_cuda:
                out = out.cuda()
            p = x[:, :, 0]
            yt = p.transpose(0, 1).mul(gate)
            out[:, 0] = yt.sum(dim=1)*b[:, 0]
        else:
            gate = F.relu(self.linearOut(outLSTM))
            b = F.relu(self.linearB(outLSTM))
            yt = torch.zeros([nt, ns, rho]).float()
            out = torch.zeros([nt, ns, ny+nyc]).float()
            if next(self.parameters()).is_cuda:
                out = out.cuda()
                yt = yt.cuda()
            for k in range(rho, nt):
                p = x[k-rho:k, :, 0]
                yt[k, :, :] = p.transpose(0, 1).mul(gate[k, :, :])
                out[k, :, 0] = yt[k, :, :].sum(dim=1)*b[k, :, 0]

        if next(self.parameters()).is_cuda:
            r = r.cuda()
            t = t.cuda()
        for j in range(nyc):
            c0 = r[0, j]
            c1 = r[1, j]
            rr = 10**r[2, j]
            # func = c0 * torch.exp(-rr*t)*rr + c1*(1-torch.exp(-rr*t))
            func = c1*(1-torch.exp(-rr*t))
            if self.training:
                out[:, ny+j] = yt.mul(func).sum(dim=1)/yt.sum(dim=1)
                # out[:, ny+j] = yt.mul(func).sum(dim=1)
            else:
                for k in range(rho, nt):
                    out[k, :, ny+j] = yt[k, :, :].mul(func).sum(dim=1)/yt[k, :, :].sum(dim=1)
                    # out[k, :, ny+j] = yt[k, :,:].mul(func).sum(dim=1)                    
        if self.training:
            return out
        else:
            return out, b, gate


class LstmModel(torch.nn.Module):
    def __init__(self, *, nx, ny, hiddenSize, dr=0.5):
        super(LstmModel, self).__init__()
        self.nx = nx
        self.ny = ny
        self.hiddenSize = hiddenSize
        self.ct = 0
        self.nLayer = 1
        self.linearIn = torch.nn.Linear(nx, hiddenSize)
        self.lstm = CudnnLstm(
            inputSize=hiddenSize, hiddenSize=hiddenSize, dr=dr)
        self.lstm2 = CudnnLstm(
            inputSize=hiddenSize, hiddenSize=hiddenSize, dr=dr)
        self.linearOut = torch.nn.Linear(hiddenSize, ny)
        self.gpu = 1

    def forward(self, x, doDropMC=False):
        x0 = F.relu(self.linearIn(x))
        outLSTM, (hn, cn) = self.lstm(x0)
        outLSTM2, (hn, cn) = self.lstm2(outLSTM)
        out = self.linearOut(outLSTM2)
        # out = rho/time * batchsize * Ntargetvar
        return out
