import torch
from copy import deepcopy
from itertools import product


class DOMAIN:
    def __init__(self, hidden_layer_num):
        self.idxes = [None] * hidden_layer_num
        self.additional_bounds = [None] * hidden_layer_num
        self.acting = [False] * hidden_layer_num

    def split(self, indexes, intervals):
        pass

    def add_bounds(self, idx, bounds):
        layer_id, neuron_id = idx
        print('--', '\n', bounds, '\n', '--')
        bounds = [torch.unsqueeze(bounds[0], dim=0), torch.unsqueeze(bounds[1], dim=0)]
        if not self.acting[layer_id]:
            self.idxes[layer_id] = [neuron_id]
            self.additional_bounds[layer_id] = bounds
            self.acting[layer_id] = True
        else:
            if neuron_id in self.idxes[layer_id]:
                pos = self.idxes[layer_id].index(neuron_id)
                self.additional_bounds[layer_id][0][pos] = bounds[0]
                self.additional_bounds[layer_id][1][pos] = bounds[1]
            else:
                self.idxes[layer_id].append(neuron_id)
                self.additional_bounds[layer_id][0] = torch.cat([self.additional_bounds[layer_id][0], bounds[0]])
                self.additional_bounds[layer_id][1] = torch.cat([self.additional_bounds[layer_id][1], bounds[1]])

    def merge_bound(self, ori_bounds):
        new_bounds = deepcopy(ori_bounds)
        for i in range(len(new_bounds)):
            if self.acting[i]:
                new_bounds[i][0][self.idxes[i]] = self.additional_bounds[i][0]
                new_bounds[i][1][self.idxes[i]] = self.additional_bounds[i][1]
        return new_bounds


class DOMNEW:
    def __init__(self, layer_num, neuron_shapes, device):
        self.idxes = [torch.zeros(neuron_shapes[i], device=device) for i in range(layer_num - 1)]
        self.lower_bounds = [torch.zeros(neuron_shapes[i], device=device) for i in range(layer_num - 1)]
        self.upper_bounds = [torch.zeros(neuron_shapes[i], device=device) for i in range(layer_num - 1)]

    def add_bounds(self, idx, bounds):
        layer_id, neuron_id = idx
        self.idxes[layer_id][neuron_id] = 1.0
        self.lower_bounds[layer_id][neuron_id] = bounds[0]
        self.upper_bounds[layer_id][neuron_id] = bounds[1]

    def merge_bound(self, ori_bounds):
        new_bounds = deepcopy(ori_bounds)
        for i in range(len(ori_bounds)):
            new_bounds[i][0] = torch.where(self.idxes[i] == 1.0, self.lower_bounds[i], new_bounds[i][0])
            new_bounds[i][1] = torch.where(self.idxes[i] == 1.0, self.upper_bounds[i], new_bounds[i][1])
        return new_bounds


class DOMCOMP:
    def __init__(self, vmodel):
        self.idxes = [torch.zeros_like(vmodel.hidden_bounds[i][0]) for i in range(len(vmodel.hidden_bounds))]
        self.bounds = deepcopy(vmodel.hidden_bounds)
        self.slops = deepcopy(vmodel.saved_slops)
        self.intercepts = deepcopy(vmodel.saved_intercepts)

    def add_info(self, layer_idx, neuron_idx, bounds, relaxation):
        self.idxes[layer_idx][neuron_idx] = 1.0
        self.bounds[layer_idx][0][neuron_idx] = bounds[0]
        self.bounds[layer_idx][1][neuron_idx] = bounds[1]
        self.slops[layer_idx][0][neuron_idx] = relaxation[0]
        self.slops[layer_idx][1][neuron_idx] = relaxation[2]
        self.intercepts[layer_idx][0][neuron_idx] = relaxation[1]
        self.intercepts[layer_idx][1][neuron_idx] = relaxation[3]


class DONEW_BAT:
    # def __init__(self, layer_num, neuron_shapes, batch_size):
    #     self.idxes = [torch.zeros((batch_size, neuron_shapes[i])) for i in range(layer_num - 1)]
    #     self.lower_bounds = [torch.zeros((batch_size, neuron_shapes[i])) for i in range(layer_num - 1)]
    #     self.upper_bounds = [torch.zeros((batch_size, neuron_shapes[i])) for i in range(layer_num - 1)]
    def __init__(self, hidden_num):
        self.idxes = [None] * hidden_num
        self.lower_bounds = [None] * hidden_num
        self.upper_bounds = [None] * hidden_num

    def from_domain_list(self, domains_new):
        # unfinished
        l_num = len(domains_new[0].idxes)


def make_domnew_batch(domains_new):
    h_num = len(domains_new[0].idxes)
    batch_size = len(domains_new)
    dom_batch = DONEW_BAT(h_num)
    for i in range(h_num):
        dom_batch.idxes[i] = torch.stack([domains_new[j].idxes[i] for j in range(batch_size)])
        dom_batch.lower_bounds[i] = torch.stack([domains_new[j].lower_bounds[i] for j in range(batch_size)])
        dom_batch.upper_bounds[i] = torch.stack([domains_new[j].upper_bounds[i] for j in range(batch_size)])
    return dom_batch


def make_domcomp_batch(domcomp_list):
    dom_batch = deepcopy(domcomp_list[0])
    batch_size = len(domcomp_list)
    for i in range(len(dom_batch.bounds)):
        dom_batch.idxes[i] = torch.stack([domcomp_list[j].idxes[i] for j in range(batch_size)])
        dom_batch.bounds[i][0] = torch.stack([domcomp_list[j].bounds[i][0] for j in range(batch_size)])
        dom_batch.bounds[i][1] = torch.stack([domcomp_list[j].bounds[i][1] for j in range(batch_size)])
        dom_batch.slops[i][0] = torch.stack([domcomp_list[j].slops[i][0] for j in range(batch_size)])
        dom_batch.slops[i][1] = torch.stack([domcomp_list[j].slops[i][1] for j in range(batch_size)])
        dom_batch.intercepts[i][0] = torch.stack([domcomp_list[j].intercepts[i][0] for j in range(batch_size)])
        dom_batch.intercepts[i][1] = torch.stack([domcomp_list[j].intercepts[i][1] for j in range(batch_size)])
    return dom_batch


def split_domain(d: DOMAIN, indexes, intervals: list):
    num = len(intervals)
    new_domains = [deepcopy(d) for i in range(num)]

    for i in range(num):
        new_domains[i].add_bounds(indexes, intervals[i])
        # print(new_domains[i].additional_bounds)

    return new_domains


def split_domain_complete(d: DOMCOMP, layer_idx, neuron_idx, interval_list, relaxation_list):
    num = len(interval_list)
    new_domains = [deepcopy(d) for i in range(num)]

    for i in range(num):
        new_domains[i].add_info(layer_idx, neuron_idx, interval_list[i], relaxation_list[i])
    return new_domains


def split_domain_multi(d: DOMNEW, idx_list, interval_list):
    combination = list(product(*interval_list))
    new_domains = [deepcopy(d) for i in range(len(combination))]
    for nid in range(len(new_domains)):
        for i in range(len(idx_list)):
            new_domains[nid].add_bounds(idx_list[i], combination[nid][i])
    return new_domains


def split_domain_new(d: DOMNEW, indexes, intervals: list):
    num = len(intervals)
    new_domains = [deepcopy(d) for i in range(num)]

    for i in range(num):
        new_domains[i].add_bounds(indexes, intervals[i])
        # print(new_domains[i].additional_bounds)
    return new_domains


if __name__ == '__main__':
    idx = torch.tensor([0, 1, 0, 1, 0, 0], dtype=torch.float)
    a = torch.rand(6)
    b = torch.tensor([0, 0.1, 0, 0.1, 0, 0], dtype=torch.float)
    c = torch.where(idx == 1, b, a)
    print(c)
