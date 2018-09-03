import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RWA(nn.Module):
    """
    Recurrent Weighted Average
    https://arxiv.org/pdf/1703.01253.pdf
    """
    def __init__(self, input_dim, output_dim, activation=None):
        super().__init__()
        self.activation = nn.Tanh() if activation is None else activation
        self.input_dim = input_dim
        self.output_dim = output_dim
        self._u = nn.Linear(input_dim, output_dim)
        self._g = nn.Linear(input_dim+output_dim, output_dim)
        self._a = nn.Linear(input_dim+output_dim, output_dim, bias=False)
        self.s0 = nn.parameter.Parameter(torch.Tensor(output_dim,))
        # ----------- initialize weights
        self.s0 = nn.init.normal_(self.s0, 0, 1)
        self._u.bias.data.fill_(0)
        self._g.bias.data.fill_(0)

        low = - math.sqrt(6 / (input_dim + output_dim))
        high = - low
        self._u.weight.data = nn.init.uniform_(self._u.weight.data, low, high)
        self._g.weight.data = nn.init.uniform_(self._g.weight.data, low, high)
        self._a.weight.data = nn.init.uniform_(self._a.weight.data, low, high)

    def forward(self, x, shape_mode='blc'):
        if shape_mode == 'bcl':
            x = x.permute(0, 2, 1)  # BLC
        elif shape_mode == 'blc':
            pass  # All ok
        else:
            raise Exception('shape_mode should be "blc" or "bcl"')
        s0 = torch.stack([self.s0]*x.size()[0], dim=0)
        # ------------keep track of these
        last_h = F.tanh(s0)
        numerator = torch.zeros((x.size()[0], self.output_dim))
        denominator = torch.zeros((x.size()[0], self.output_dim))
        last_a_max = torch.ones((x.size()[0], self.output_dim)) * 1e-38
        # ------------initialization done
        U = self._u(x)
        outputs = []
        for idx in range(x.size()[1]):
            xi = x[:, idx, :]
            ui = U[:, idx, :]
            xh = torch.cat([xi, last_h], dim=1)
            # ----- calculate Z and A
            z = ui * F.tanh(self._g(xh))
            a = self._a(xh)
            a_max, _ = torch.max(torch.stack([a, last_a_max], dim=1), dim=1)
            # ----- calculate  num and den
            e_to_a = torch.exp(a_max - last_a_max)
            numerator = numerator + z * e_to_a
            denominator = denominator + e_to_a
            last_h = self.activation(numerator / denominator)
            last_a_max = a_max
            outputs.append(last_h)
        return torch.stack(outputs, dim=1)


class CorefGRU(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super().__init__()
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.w_r = nn.Linear(inp_dim, out_dim)
        self.u_r = nn.Linear(out_dim, out_dim, bias=False)
        self.w_z = nn.Linear(inp_dim, out_dim)
        self.u_z = nn.Linear(out_dim, out_dim, bias=False)
        self.w_h = nn.Linear(inp_dim, out_dim)
        self.u_h = nn.Linear(out_dim, out_dim, bias=False)
        self.k1k2 = nn.Linear(inp_dim, 2, bias=False)

    def forward(self, x, cor, shape_mode='blc'):
        if shape_mode == 'bcl':
            x = x.permute(0, 2, 1)  # BLC
        elif shape_mode == 'blc':
            pass  # All ok
        else:
            raise Exception('shape_mode should be "blc" or "bcl"')
        B, L, _ = x.size()
        hid_states = [torch.zeros((B, self.out_dim))]
        for t in range(L):
            # ---------- xt, h_tm1, h_y
            xt = x[:, t]
            h_tm1 = hid_states[-1]
            h_y = torch.stack([hid_states[i if i > 0 else -1][idx]
                               for idx, i in enumerate(cor[:, t])])
            # ---------- a calculation
            m_t1 = h_tm1[:, :self.out_dim//2]
            m_t2 = h_y[:, self.out_dim//2:]
            a = torch.unsqueeze(F.softmax(self.k1k2(xt), dim=1)[:, 0],
                                dim=1)
            m_t = torch.cat([a * m_t1, (1 - a) * m_t2], dim=1)
            # ---------- equations
            r_t = F.sigmoid(self.w_r(xt) + self.u_r(m_t))
            z_t = F.sigmoid(self.w_z(xt) + self.u_z(m_t))
            h_ = F.tanh(self.w_h(xt) + r_t * self.u_h(m_t))
            h = (1 - z_t) * m_t + z_t * h_
            # ---------- done
            hid_states.append(h)
        return torch.stack(hid_states[1:], dim=1)
