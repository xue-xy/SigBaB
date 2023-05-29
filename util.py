import torch
from copy import deepcopy


def evaluate_inf_max(region_min, region_max, c: torch.Tensor):
    if c.dim() == 1:
        c = torch.unsqueeze(c, 0)

    upper = torch.matmul(torch.maximum(c, torch.zeros_like(c)), region_max) + \
            torch.matmul(torch.minimum(c, torch.zeros_like(c)), region_min)

    return upper


def evaluate_inf_min(region_min, region_max, c: torch.Tensor):
    if c.dim() == 1:
        c = torch.unsqueeze(c, 0)

    lower = torch.matmul(torch.maximum(c, torch.zeros_like(c)), region_min) + \
            torch.matmul(torch.minimum(c, torch.zeros_like(c)), region_max)

    return lower


def evaluate_inf_min_arg(region_min, region_max, c: torch.Tensor):
    if c.dim() == 1:
        c = torch.unsqueeze(c, dim=0)
    x = torch.where(c > 0, region_min, region_max)

    return torch.sum(x * c, dim=1), x


if __name__ == '__main__':
    a = torch.tensor([2, -2], dtype=torch.float)
    x_max = torch.tensor([1, 0.7])
    x_min = torch.tensor([0, 0.2])
    print(evaluate_inf_min_arg(x_min, x_max, a))
