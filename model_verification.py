import torch
from model.models import *
from util import *
from activation_bound import *
from copy import deepcopy
from domain import DOMAIN, DOMNEW, DONEW_BAT
import torch.optim as opt


class VModel:
    def __init__(self, model, mean, std, device, activation='sigmoid'):
        self.model = model.to(device)
        paras = model.state_dict()
        p_keys = list(paras.keys())
        self.layer_num = len(p_keys) // 2
        self.weights = [paras[p_keys[2 * i]].to(device) for i in range(self.layer_num)]
        self.biases = [paras[p_keys[2 * i + 1]].to(device) for i in range(self.layer_num)]
        self.mean = mean
        self.std = std
        self.activation_fn = activation
        self.hidden_bounds = [None] * (self.layer_num - 1)
        self.saved_slops = [None] * (self.layer_num - 1)
        self.saved_intercepts = [None] * (self.layer_num - 1)
        self.device = device

    def reset(self):
        self.hidden_bounds = [None] * (self.layer_num - 1)
        self.saved_slops = [None] * (self.layer_num - 1)
        self.saved_intercepts = [None] * (self.layer_num - 1)

    def check(self, pce, c):
        pred = self.model((pce - self.mean) / self.std)
        if pce.dim() == 2:
            return torch.sum(c * pred.detach(), dim=1)
        else:
            return torch.sum(c * pred.detach())

    def _bound(self, weight, bias, layer_idx, region_min, region_max):
        upper_slop = weight
        upper_intercept = bias
        lower_slop = weight
        lower_intercept = bias

        for i in range(layer_idx - 1, -1, -1):
            upper_intercept = torch.matmul(torch.maximum(upper_slop, torch.zeros_like(upper_slop)),
                                           self.saved_intercepts[i][1]) \
                              + torch.matmul(torch.minimum(upper_slop, torch.zeros_like(upper_slop)),
                                             self.saved_intercepts[i][0]) \
                              + upper_intercept
            upper_slop = torch.maximum(upper_slop, torch.zeros_like(upper_slop)) * self.saved_slops[i][1] \
                         + torch.minimum(upper_slop, torch.zeros_like(upper_slop)) * self.saved_slops[i][1]
            lower_intercept = torch.matmul(torch.maximum(lower_slop, torch.zeros_like(lower_slop)),
                                           self.saved_intercepts[i][0]) \
                              + torch.matmul(torch.minimum(lower_slop, torch.zeros_like(upper_slop)),
                                             self.saved_intercepts[i][1]) \
                              + lower_intercept
            lower_slop = torch.maximum(lower_slop, torch.zeros_like(lower_slop)) * self.saved_slops[i][0] \
                         + torch.minimum(lower_slop, torch.zeros_like(lower_slop)) * self.saved_slops[i][1]

            upper_intercept = torch.matmul(upper_slop, self.biases[i]) + upper_intercept
            upper_slop = torch.matmul(upper_slop, self.weights[i])
            lower_intercept = torch.matmul(lower_slop, self.biases[i]) + lower_intercept
            lower_slop = torch.matmul(lower_slop, self.weights[i])

        upper_bound = evaluate_inf_max(region_min, region_max, upper_slop) + upper_intercept
        lower_bound = evaluate_inf_min(region_min, region_max, lower_slop) + lower_intercept

        if self.hidden_bounds[layer_idx] is None:
            self.hidden_bounds[layer_idx] = [lower_bound, upper_bound]

        al_slop, al_intercept, au_slop, au_intercept = sigmoid_parallel(lower_bound, upper_bound)
        if self.saved_slops[layer_idx] is None:
            self.saved_slops[layer_idx] = [al_slop, au_slop]
            self.saved_intercepts[layer_idx] = [al_intercept, au_intercept]

    def _hidden_layers_bounds(self, region_min, region_max):
        for i in range(self.layer_num - 1):
            self._bound(self.weights[i], self.biases[i], i, region_min, region_max)

    def get_neuron_bound(self, idx):
        layer_id, neuron_id = idx
        return self.hidden_bounds[layer_id][0][neuron_id], self.hidden_bounds[layer_id][1][neuron_id]

    def beta_init(self, domain: DOMAIN):
        l = len(domain.idxes)
        betas = [None] * l
        for i in range(l):
            pass

    def backward_propagation(self, x, eps, c, b, norm='inf'):
        # find the lower bound of the minimum value
        x_min = torch.max(x - eps, torch.zeros_like(x))
        x_max = torch.min(x + eps, torch.ones_like(x))

        # perform normalization
        x_min = (x_min - self.mean) / self.std
        x_max = (x_max - self.mean) / self.std

        self._hidden_layers_bounds(x_min, x_max)

        coeff = torch.matmul(c, self.weights[-1])
        const = torch.matmul(c, self.biases[-1]) + b
        for i in range(self.layer_num - 2, -1, -1):
            const = torch.matmul(torch.maximum(coeff, torch.zeros_like(coeff)), self.saved_intercepts[i][0]) \
                    + torch.matmul(torch.minimum(coeff, torch.zeros_like(coeff)), self.saved_intercepts[i][1]) \
                    + const
            coeff = torch.maximum(coeff, torch.zeros_like(coeff)) * self.saved_slops[i][0] \
                    + torch.minimum(coeff, torch.zeros_like(coeff)) * self.saved_slops[i][1]
            const = torch.matmul(coeff, self.biases[i]) + const
            coeff = torch.matmul(coeff, self.weights[i])

        # pce for pesudo-counterexample
        val, pce = evaluate_inf_min_arg(x_min, x_max, coeff)
        val = val + const
        return val, pce

    def backward_with_betas(self, c, b, betas, dom: DOMAIN, slops, intercepts, region_min, region_max):
        coeff = torch.matmul(c, self.weights[-1])
        const = torch.matmul(c, self.biases[-1]) + b
        for i in range(self.layer_num - 2, -1, -1):
            const = torch.matmul(torch.maximum(coeff, torch.zeros_like(coeff)), intercepts[i][0]) \
                    + torch.matmul(torch.minimum(coeff, torch.zeros_like(coeff)), intercepts[i][1]) \
                    + const
            coeff = torch.maximum(coeff, torch.zeros_like(coeff)) * slops[i][0] \
                    + torch.minimum(coeff, torch.zeros_like(coeff)) * slops[i][1]

            if dom.acting[i]:
                const = torch.matmul(betas[i][0], dom.additional_bounds[i][0]) \
                        - torch.matmul(betas[i][1], dom.additional_bounds[i][1]) \
                        + const
                coeff[dom.idxes[i]] = coeff[dom.idxes[i]] - betas[i][0] + betas[i][1]

            const = torch.matmul(coeff, self.biases[i]) + const
            coeff = torch.matmul(coeff, self.weights[i])

        val, pce = evaluate_inf_min_arg(region_min, region_max, coeff)
        # val = evaluate_inf_min(region_min, region_max, coeff) + const
        return val + const, pce

    def backward_with_betas_new(self, c, b, betas_low, betas_up, dom: DOMNEW, slops, intercepts, region_min, region_max):
        coeff = torch.matmul(c, self.weights[-1])
        const = torch.matmul(c, self.biases[-1]) + b
        for i in range(self.layer_num - 2, -1, -1):
            const = torch.matmul(torch.maximum(coeff, torch.zeros_like(coeff)), intercepts[i][0]) \
                    + torch.matmul(torch.minimum(coeff, torch.zeros_like(coeff)), intercepts[i][1]) \
                    + const
            coeff = torch.maximum(coeff, torch.zeros_like(coeff)) * slops[i][0] \
                    + torch.minimum(coeff, torch.zeros_like(coeff)) * slops[i][1]

            const = torch.matmul(betas_low[i], dom.lower_bounds[i]) \
                    - torch.matmul(betas_up[i], dom.upper_bounds[i]) \
                    + const
            coeff = coeff - betas_low[i] + betas_up[i]

            const = torch.matmul(coeff, self.biases[i]) + const
            coeff = torch.matmul(coeff, self.weights[i])
            # print(const)
        val, pce = evaluate_inf_min_arg(region_min, region_max, coeff)
        # val = evaluate_inf_min(region_min, region_max, coeff) + const
        return val + const, pce

    def backward_with_betas_parallel(self, c, b, betas_low, betas_up, dom: DONEW_BAT, batch_size, slops, intercepts, region_min, region_max):
        coeff = torch.matmul(c.repeat(batch_size, 1), self.weights[-1])
        const = torch.matmul(c.repeat(batch_size, 1), self.biases[-1]) + b

        for i in range(self.layer_num - 2, -1, -1):
            const = torch.sum(torch.maximum(coeff, torch.zeros_like(coeff)) * intercepts[i][0], dim=1) + \
                    torch.sum(torch.minimum(coeff, torch.zeros_like(coeff)) * intercepts[i][1], dim=1) + \
                    const
            coeff = torch.maximum(coeff, torch.zeros_like(coeff)) * slops[i][0] + \
                    torch.minimum(coeff, torch.zeros_like(coeff)) * slops[i][1]

            const = torch.sum(betas_low[i] * dom.lower_bounds[i], dim=1) - \
                    torch.sum(betas_up[i] * dom.upper_bounds[i], dim=1) + \
                    const
            coeff = coeff - betas_low[i] + betas_up[i]

            const = torch.matmul(coeff, self.biases[i]) + const
            coeff = torch.matmul(coeff, self.weights[i])

        val, pce = evaluate_inf_min_arg(region_min, region_max, coeff)
        # val = evaluate_inf_min(region_min, region_max, coeff) + const
        return val + const, pce

    def backward_with_betas_parallel_convex(self, c, b, betas_low, betas_up, dom, batch_size, slops, intercepts, region_min, region_max):
        coeff = torch.matmul(c.repeat(batch_size, 1), self.weights[-1])
        const = torch.matmul(c.repeat(batch_size, 1), self.biases[-1]) + b

        for i in range(self.layer_num - 2, -1, -1):
            const = torch.sum(torch.maximum(coeff, torch.zeros_like(coeff)) * intercepts[i][0], dim=1) + \
                    torch.sum(torch.minimum(coeff, torch.zeros_like(coeff)) * intercepts[i][1], dim=1) + \
                    const
            coeff = torch.maximum(coeff, torch.zeros_like(coeff)) * slops[i][0] + \
                    torch.minimum(coeff, torch.zeros_like(coeff)) * slops[i][1]

            const = torch.sum(betas_low[i] * dom.bounds[i][0], dim=1) - \
                    torch.sum(betas_up[i] * dom.bounds[i][1], dim=1) + \
                    const
            coeff = coeff - betas_low[i] + betas_up[i]

            const = torch.matmul(coeff, self.biases[i]) + const
            coeff = torch.matmul(coeff, self.weights[i])

        val, pce = evaluate_inf_min_arg(region_min, region_max, coeff)
        # val = evaluate_inf_min(region_min, region_max, coeff) + const
        return val + const, pce

    def backward_bab_simple(self, x, eps, c, domain: DOMAIN, b=torch.tensor(0)):
        x_min = torch.maximum(x - eps, torch.zeros_like(x))
        x_max = torch.minimum(x + eps, torch.ones_like(x))
        x_min = (x_min - self.mean) / self.std
        x_max = (x_max - self.mean) / self.std

        slops = deepcopy(self.saved_slops)
        intercepts = deepcopy(self.saved_intercepts)

        idxes = domain.idxes
        for i in range(self.layer_num - 1):
            if domain.acting[i]:
                # this_bounds[i][0][idxes[i]] = domain.additional_bounds[i][0]
                # this_bounds[i][1][idxes[i]] = domain.additional_bounds[i][1]
                ls, li, us, ui = sigmoid_parallel(domain.additional_bounds[i][0], domain.additional_bounds[i][1])
                slops[i][0][idxes[i]] = ls
                slops[i][1][idxes[i]] = us
                intercepts[i][0][idxes[i]] = li
                intercepts[i][1][idxes[i]] = ui

        coeff = torch.matmul(c, self.weights[-1])
        const = torch.matmul(c, self.biases[-1]) + b

        for i in range(self.layer_num - 2, -1, -1):
            const = torch.matmul(torch.maximum(coeff, torch.zeros_like(coeff)), intercepts[i][0]) \
                    + torch.matmul(torch.minimum(coeff, torch.zeros_like(coeff)), intercepts[i][1]) \
                    + const
            coeff = torch.maximum(coeff, torch.zeros_like(coeff)) * slops[i][0] \
                    + torch.minimum(coeff, torch.zeros_like(coeff)) * slops[i][1]
            const = torch.matmul(coeff, self.biases[i]) + const
            coeff = torch.matmul(coeff, self.weights[i])

        val = evaluate_inf_min(x_min, x_max, coeff) + const
        return val

    def bab_propagation(self, x, eps, c, domain: DOMAIN, b=torch.tensor(0), lr=0.1, verbose=False):
        x_min = torch.maximum(x - eps, torch.zeros_like(x))
        x_max = torch.minimum(x + eps, torch.ones_like(x))
        x_min = (x_min - self.mean) / self.std
        x_max = (x_max - self.mean) / self.std

        # generating the intermediate bounds in this domain, and generate the corresponding slops
        # this_bounds = deepcopy(self.hidden_bounds)
        this_slops = deepcopy(self.saved_slops)
        this_intercepts = deepcopy(self.saved_intercepts)

        idxes = domain.idxes
        for i in range(self.layer_num - 1):
            if domain.acting[i]:
                # this_bounds[i][0][idxes[i]] = domain.additional_bounds[i][0]
                # this_bounds[i][1][idxes[i]] = domain.additional_bounds[i][1]
                ls, li, us, ui = sigmoid_single(domain.additional_bounds[i][0], domain.additional_bounds[i][1])
                this_slops[i][0][idxes[i]] = ls
                this_slops[i][1][idxes[i]] = us
                this_intercepts[i][0][idxes[i]] = li
                this_intercepts[i][1][idxes[i]] = ui

        betas = betas_init(domain)
        # print(list(filter(lambda x: x is not None, betas)))
        optimizer = opt.Adam(list(filter(lambda x: x is not None, betas)), lr=0.4)
        scheduler = opt.lr_scheduler.ExponentialLR(optimizer, 0.99)

        iter_time = 10
        for i in range(iter_time):
            optimizer.zero_grad()

            betasp = [None] * (self.layer_num - 1)
            for j in range(self.layer_num - 1):
                if domain.acting[j]:
                    betasp[j] = torch.maximum(betas[j], torch.zeros_like(betas[j]))
            ret, pce = self.backward_with_betas(c, b, betasp, domain, this_slops, this_intercepts, x_min, x_max)
            loss = - ret
            loss.backward()
            # print(betas[0].grad)
            optimizer.step()
            scheduler.step()
            # print(ret, betasp[0])
            # quit()
        # todo: save the best betas and load for post bab questions
        return ret.clone().detach(), pce

    def bab_propagation_new(self, x, eps, c, domain: DOMNEW, b=torch.tensor(0), lr=0.1, verbose=False):
        x_min = torch.maximum(x - eps, torch.zeros_like(x))
        x_max = torch.minimum(x + eps, torch.ones_like(x))
        x_min = (x_min - self.mean) / self.std
        x_max = (x_max - self.mean) / self.std

        # generating the intermediate bounds in this domain, and generate the corresponding slops
        # this_bounds = deepcopy(self.hidden_bounds)
        this_slops = deepcopy(self.saved_slops)
        this_intercepts = deepcopy(self.saved_intercepts)

        for i in range(self.layer_num - 1):
            idx_mask = domain.idxes[i].bool()
            if idx_mask.any():
                ls, li, us, ui = sigmoid_single(torch.masked_select(domain.lower_bounds[i], idx_mask),
                                                torch.masked_select(domain.upper_bounds[i], idx_mask))

                this_slops[i][0].masked_scatter_(idx_mask, ls)
                this_slops[i][1].masked_scatter_(idx_mask, us)
                this_intercepts[i][0].masked_scatter_(idx_mask, li)
                this_intercepts[i][1].masked_scatter_(idx_mask, ui)

        betas_low = [torch.rand(self.biases[i].shape[0], requires_grad=True) for i in range(self.layer_num - 1)]
        betas_up = [torch.rand(self.biases[i].shape[0], requires_grad=True) for i in range(self.layer_num - 1)]

        optimizer = opt.Adam(betas_low + betas_up, lr=0.4)
        scheduler = opt.lr_scheduler.ExponentialLR(optimizer, 0.99)

        iter_time = 10
        for i in range(iter_time):
            optimizer.zero_grad()

            betasp_low = [None] * (self.layer_num - 1)
            betasp_up = [None] * (self.layer_num - 1)
            for j in range(self.layer_num - 1):
                betasp_low[j] = torch.maximum(betas_low[j], torch.zeros_like(betas_low[j])) * domain.idxes[j]
                betasp_up[j] = torch.maximum(betas_up[j], torch.zeros_like(betas_up[j])) * domain.idxes[j]
            ret, pce = self.backward_with_betas_new(c, b, betasp_low, betasp_up, domain, this_slops, this_intercepts, x_min, x_max)
            loss = - ret
            loss.backward()
            # print(betas[0].grad)
            optimizer.step()
            scheduler.step()
            # print(ret, betasp[0])
            # quit()
        # todo: save the best betas and load for post bab questions
        return ret.clone().detach(), pce

    def bab_propagation_parallel(self, x, eps, c, domain_batch: DONEW_BAT, batch_size, device, b=torch.tensor(0), lr=0.4, verbose=False):
        x_min = torch.maximum(x - eps, torch.zeros_like(x))
        x_max = torch.minimum(x + eps, torch.ones_like(x))
        x_min = (x_min - self.mean) / self.std
        x_max = (x_max - self.mean) / self.std

        this_slops = deepcopy(self.saved_slops)
        this_intercepts = deepcopy(self.saved_intercepts)

        for i in range(self.layer_num - 1):
            this_slops[i][0] = this_slops[i][0].repeat(batch_size, 1)
            this_slops[i][1] = this_slops[i][1].repeat(batch_size, 1)
            this_intercepts[i][0] = this_intercepts[i][0].repeat(batch_size, 1)
            this_intercepts[i][1] = this_intercepts[i][1].repeat(batch_size, 1)
            idx_mask = domain_batch.idxes[i].bool()
            if idx_mask.any():
                ls, li, us, ui = sigmoid_single(torch.masked_select(domain_batch.lower_bounds[i], idx_mask),
                                                torch.masked_select(domain_batch.upper_bounds[i], idx_mask))
                this_slops[i][0].masked_scatter_(idx_mask, ls)
                this_slops[i][1].masked_scatter_(idx_mask, us)
                this_intercepts[i][0].masked_scatter_(idx_mask, li)
                this_intercepts[i][1].masked_scatter_(idx_mask, li)

        betas_low = [torch.rand((batch_size, self.biases[i].shape[0]), requires_grad=True, device=device) for i in range(self.layer_num - 1)]
        betas_up = [torch.rand((batch_size, self.biases[i].shape[0]), requires_grad=True, device=device) for i in range(self.layer_num - 1)]

        optimizer = opt.Adam(betas_low + betas_up, lr=lr)
        scheduler = opt.lr_scheduler.ExponentialLR(optimizer, 0.99)

        iter_time = 10
        for i in range(iter_time):
            optimizer.zero_grad()

            betasp_low = [None] * (self.layer_num - 1)
            betasp_up = [None] * (self.layer_num - 1)
            for j in range(self.layer_num - 1):
                betasp_low[j] = torch.maximum(betas_low[j], torch.zeros_like(betas_low[j])) * domain_batch.idxes[j]
                betasp_up[j] = torch.maximum(betas_up[j], torch.zeros_like(betas_up[j])) * domain_batch.idxes[j]
            ret, pce = self.backward_with_betas_parallel(c, b, betasp_low, betasp_up, domain_batch, batch_size,
                                                        this_slops, this_intercepts, x_min, x_max)
            loss = -1 * torch.mean(ret)
            loss.backward()
            # print(betas[0].grad)
            optimizer.step()
            scheduler.step()
            # print(ret, betasp[0])
            # quit()
        # todo: save the best betas and load for post bab questions
        return ret.clone().detach(), pce

    def bab_propagation_parallel_convex(self, x, eps, c, domain_batch, batch_size, device, b=torch.tensor(0), lr=0.4, verbose=False):
        x_min = torch.maximum(x - eps, torch.zeros_like(x))
        x_max = torch.minimum(x + eps, torch.ones_like(x))
        x_min = (x_min - self.mean) / self.std
        x_max = (x_max - self.mean) / self.std

        betas_low = [torch.rand((batch_size, self.biases[i].shape[0]), requires_grad=True, device=device) for i in range(self.layer_num - 1)]
        betas_up = [torch.rand((batch_size, self.biases[i].shape[0]), requires_grad=True, device=device) for i in range(self.layer_num - 1)]

        optimizer = opt.Adam(betas_low + betas_up, lr=0.2)
        scheduler = opt.lr_scheduler.ExponentialLR(optimizer, 0.99)

        iter_time = 20
        for i in range(iter_time):
            optimizer.zero_grad()

            betasp_low = [None] * (self.layer_num - 1)
            betasp_up = [None] * (self.layer_num - 1)
            for j in range(self.layer_num - 1):
                betasp_low[j] = torch.maximum(betas_low[j], torch.zeros_like(betas_low[j])) * domain_batch.idxes[j]
                betasp_up[j] = torch.maximum(betas_up[j], torch.zeros_like(betas_up[j])) * domain_batch.idxes[j]
            ret, pce = self.backward_with_betas_parallel_convex(c, b, betasp_low, betasp_up, domain_batch, batch_size,
                                                                domain_batch.slops, domain_batch.intercepts, x_min, x_max)
            loss = -1 * torch.mean(ret)
            loss.backward()

            # loss = -1 * ret
            # loss.backward(torch.ones_like(loss))

            # print(betas[0].grad)
            optimizer.step()
            scheduler.step()
            # print(ret, betasp[0])
            # quit()
        # todo: save the best betas and load for post bab questions
        return ret.clone().detach(), pce


def betas_init(dom: DOMAIN):
    l = len(dom.idxes)
    betas = [None] * l
    for i in range(l):
        if dom.acting[i]:
            betas[i] = torch.rand((2, len(dom.idxes[i])), requires_grad=True)
    return betas


def betas_init_multiple(doms):
    batch_size = len(doms)
    l = len(doms[0].idxes)


if __name__ == '__main__':
    lb = torch.rand((3, 3)) - 1
    ub = torch.rand((3, 3))

    idx = torch.tensor([[1, 0, 0], [0, 1, 0], [1, 0, 1]], dtype=torch.float)
    idx_mask = idx.bool()
    ls, li, us, ui = sigmoid_parallel(torch.masked_select(lb, idx_mask), torch.masked_select(ub, idx_mask))
    os = torch.zeros((3, 3))
    print(ls)
    os.masked_scatter_(idx_mask, ls)
    # new_idx = torch.where(idx)
    #
    # os.index_put_(new_idx, ls)
    print(os)
    quit()

    b1 = torch.rand((2, 2))
    b1.requires_grad = True
    opt = opt.Adam([b1], lr=0.1)
    mask = torch.ones((2, 2))
    mask[1, 1] = 0.0

    print(b1)
    for i in range(15):
        opt.zero_grad()
        paras = torch.tensor([3.0, 2.0])

        b1p = torch.maximum(b1, torch.zeros_like(b1))
        res = torch.matmul(b1p * mask, paras)
        # print(betas.is_leaf)
        res_sum = torch.sum(res)
        res_sum.backward()
        # print(betas.grad)
        opt.step()
        # b1p = torch.relu(b1)
        # b2p = torch.relu(b2)
        print(str(i), end='    ')
        print(res, '    ', b1p)



