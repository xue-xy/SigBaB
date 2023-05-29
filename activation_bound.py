import torch
import torch.nn as nn


def sigmoid(x):
    f = nn.Sigmoid()
    return f(x)


def sigmoid_p(x):
    s = nn.Sigmoid()(x)
    return s * (1 - s)


def find_point_sigmoid(k, left, right, convex):
    iteration = 10
    if convex:
        for i in range(iteration):
            mid = (left + right) / 2
            if k < sigmoid_p(mid):
                right = mid
            else:
                left = mid
    else:
        for i in range(iteration):
            mid = (left + right) / 2
            if k < sigmoid_p(mid):
                left = mid
            else:
                right = mid
    return mid, sigmoid(mid)


def find_point_sigmoid_parallel(k, left, right, convex):
    iteration = 10
    if convex:
        for i in range(iteration):
            mid = (left + right) / 2
            lower_id = torch.where(k < sigmoid_p(mid))
            upper_id = torch.where(k >= sigmoid_p(mid))
            right[lower_id] = mid[lower_id]
            left[upper_id] = mid[upper_id]
    else:
        for i in range(iteration):
            mid = (left + right) / 2
            upper_id = torch.where(k < sigmoid_p(mid))
            lower_id = torch.where(k >= sigmoid_p(mid))
            left[upper_id] = mid[upper_id]
            right[lower_id] = mid[lower_id]
    return mid, sigmoid(mid)


def find_t_sigmoid(t, left, right, t_sig=None):
    '''
    t for tangent
    the bisec is same for the upper and lower bound
    '''
    if t_sig is None:
        t_sig = sigmoid(t)
    iteration = 10
    for i in range(iteration):
        mid = (left + right) / 2
        sig_mid = sigmoid(mid)
        if sig_mid * (1 - sig_mid) * (t - mid) + sig_mid < t_sig:
            left = mid
        else:
            right = mid
    return mid, sig_mid


def find_t_sigmoid_parallel(t, left, right, t_sig=None):
    '''
    t for tangent
    the bisec is same for the upper and lower bound
    '''
    if t_sig is None:
        t_sig = sigmoid(t)
    iteration = 10
    for i in range(iteration):
        mid = (left + right) / 2
        sig_mid = sigmoid(mid)
        lower_id = torch.where(sig_mid * (1 - sig_mid) * (t - mid) + sig_mid < t_sig)[0]
        upper_id = torch.where(sig_mid * (1 - sig_mid) * (t - mid) + sig_mid >= t_sig)[0]
        left[lower_id] = mid[lower_id]
        right[upper_id] = mid[upper_id]
    return mid, sig_mid


def sigmoid_single(lb, ub):
    assert lb.shape == ub.shape
    if ub.dim() == 0:
        ub = torch.unsqueeze(ub, dim=0)
        lb = torch.unsqueeze(lb, dim=0)

    sig_lb = sigmoid(lb)
    sig_ub = sigmoid(ub)
    k = (sig_ub - sig_lb) / (ub - lb)
    sig_p_lb = sig_lb * (1 - sig_lb)
    sig_p_ub = sig_ub * (1 - sig_ub)

    upper_slops = torch.zeros_like(lb)
    upper_intercepts = torch.zeros_like(lb)
    lower_slops = torch.zeros_like(lb)
    lower_intercepts = torch.zeros_like(lb)

    # print(sig_lb)
    # print(sig_ub)
    # print(k)
    for i in range(lb.shape[0]):
        if sig_p_lb[i] < k[i] and sig_p_ub[i] > k[i]:
            # print('1')
            upper_slops[i] = k[i]
            upper_intercepts[i] = sig_lb[i] - k[i] * lb[i]
            d, sig_d = find_point_sigmoid(k[i], lb[i], ub[i], convex=True)
            lower_slops[i] = k[i]
            lower_intercepts[i] = sig_d - k[i] * d
        elif sig_p_lb[i] > k[i] and sig_p_ub[i] < k[i]:
            # print('2')
            lower_slops[i] = k[i]
            lower_intercepts[i] = sig_lb[i] - k[i] * lb[i]
            d, sig_d = find_point_sigmoid(k[i], lb[i], ub[i], convex=False)
            upper_slops[i] = k[i]
            upper_intercepts[i] = sig_d - k[i] * d
        elif sig_p_lb[i] < k[i] and sig_p_ub[i] < k[i]:
            # print('3')
            upper_d, upper_sig_d = find_t_sigmoid(lb[i], torch.tensor(0), ub[i], t_sig=sig_lb[i])
            upper_slops[i] = upper_sig_d * (1 - upper_sig_d)
            upper_intercepts[i] = sig_lb[i] - upper_slops[i] * lb[i]
            lower_d, lower_sig_d = find_t_sigmoid(ub[i], lb[i], torch.tensor(0), t_sig=sig_ub[i])
            lower_slops[i] = lower_sig_d * (1 - lower_sig_d)
            lower_intercepts[i] = sig_ub[i] - lower_slops[i] * ub[i]
        else:
            lower_intercepts[i] = sig_lb[i]
            upper_intercepts[i] = sig_ub[i]

    return lower_slops, lower_intercepts, upper_slops, upper_intercepts


def sigmoid_parallel(lb, ub):
    # deepcert bound
    assert lb.shape == ub.shape
    if ub.dim() == 0:
        ub = torch.unsqueeze(ub, dim=0)
        lb = torch.unsqueeze(lb, dim=0)

    sig_lb = sigmoid(lb)
    sig_ub = sigmoid(ub)
    k = (sig_ub - sig_lb) / (ub - lb)
    sig_p_lb = sig_lb * (1 - sig_lb)
    sig_p_ub = sig_ub * (1 - sig_ub)

    upper_slops = torch.zeros_like(lb)
    upper_intercepts = torch.zeros_like(lb)
    lower_slops = torch.zeros_like(lb)
    lower_intercepts = torch.zeros_like(lb)

    # print(sig_lb)
    # print(sig_ub)
    # print(k)

    id1 = torch.where(torch.logical_and(sig_p_lb < k, sig_p_ub > k))[0]
    id2 = torch.where(torch.logical_and(sig_p_lb > k, sig_p_ub < k))[0]
    id3 = torch.where(torch.logical_and(sig_p_lb < k, sig_p_ub < k))[0]
    id4 = torch.where(k == 0)[0]

    if id1.shape[0] != 0:
        upper_slops[id1] = k[id1]
        upper_intercepts[id1] = sig_lb[id1] - k[id1] * lb[id1]
        d, sig_d = find_point_sigmoid_parallel(k[id1], lb[id1], ub[id1], convex=True)
        lower_slops[id1] = k[id1]
        lower_intercepts[id1] = sig_d - k[id1] * d
    if id2.shape[0] != 0:
        lower_slops[id2] = k[id2]
        lower_intercepts[id2] = sig_lb[id2] - k[id2] * lb[id2]
        d, sig_d = find_point_sigmoid_parallel(k[id2], lb[id2], ub[id2], convex=False)
        upper_slops[id2] = k[id2]
        upper_intercepts[id2] = sig_d - k[id2] * d
    if id3.shape[0] != 0:
        upper_d, upper_sig_d = find_t_sigmoid_parallel(lb[id3], torch.zeros_like(ub[id3]), ub[id3], t_sig=sig_lb[id3])
        upper_slops[id3] = upper_sig_d * (1 - upper_sig_d)
        upper_intercepts[id3] = sig_lb[id3] - upper_slops[id3] * lb[id3]
        lower_d, lower_sig_d = find_t_sigmoid_parallel(ub[id3], lb[id3], torch.zeros_like(lb[id3]), t_sig=sig_ub[id3])
        lower_slops[id3] = lower_sig_d * (1 - lower_sig_d)
        lower_intercepts[id3] = sig_ub[id3] - lower_slops[id3] * ub[id3]
    if id4.shape[0] != 0:
        upper_intercepts[id4] = sig_ub[id4]
        lower_intercepts[id4] = sig_lb[id4]

    return lower_slops, lower_intercepts, upper_slops, upper_intercepts


def sigmoid_parallel_crown(lb, ub):
    assert lb.shape == ub.shape
    if ub.dim() == 0:
        ub = torch.unsqueeze(ub, dim=0)
        lb = torch.unsqueeze(lb, dim=0)

    sig_lb = sigmoid(lb)
    sig_ub = sigmoid(ub)
    sig_p_lb = sig_lb * (1 - sig_lb)
    sig_p_ub = sig_ub * (1 - sig_ub)

    id1 = torch.where(ub <= 0)[0]
    id2 = torch.where(lb >= 0)[0]
    id3 = torch.where(torch.logical_and(lb < 0, ub > 0))[0]
    # id4 = torch.where(sig_ub - sig_lb <= 0.1)[0]

    upper_slops = torch.zeros_like(lb)
    upper_intercepts = torch.zeros_like(lb)
    lower_slops = torch.zeros_like(lb)
    lower_intercepts = torch.zeros_like(lb)

    if id1.shape[0] != 0:
        lower_slops[id1] = sig_p_lb[id1]
        lower_intercepts[id1] = sig_lb[id1] - sig_p_lb[id1] * lb[id1]
        upper_slops[id1] = (sig_ub[id1] - sig_lb[id1]) / (ub[id1] - lb[id1])
        upper_intercepts[id1] = sig_lb[id1] - upper_slops[id1] * lb[id1]
    if id2.shape[0] != 0:
        lower_slops[id2] = (sig_ub[id2] - sig_lb[id2]) / (ub[id2] - lb[id2])
        lower_intercepts[id2] = sig_lb[id2] - lower_slops[id2] * lb[id2]
        upper_slops[id2] = sig_p_ub[id2]
        upper_intercepts[id2] = sig_ub[id2] - sig_p_ub[id2] * ub[id2]
    if id3.shape[0] != 0:
        upper_d, upper_sig_d = find_t_sigmoid_parallel(lb[id3], torch.zeros_like(lb[id3]), -1 * lb[id3], t_sig=sig_lb[id3])
        upper_slops[id3] = upper_sig_d * (1 - upper_sig_d)
        upper_intercepts[id3] = sig_lb[id3] - upper_slops[id3] * lb[id3]
        lower_d, lower_sig_d = find_t_sigmoid_parallel(ub[id3], -1 * ub[id3], torch.zeros_like(ub[id3]), t_sig=sig_ub[id3])
        lower_slops[id3] = lower_sig_d * (1 - lower_sig_d)
        lower_intercepts[id3] = sig_ub[id3] - lower_slops[id3] * ub[id3]
    # if id4.shape[0] != 0:
    #     lower_slops[id4] = torch.zeros_like(lower_slops[id4])
    #     lower_intercepts[id4] = sig_lb[id4]
    #     upper_slops[id4] = torch.zeros_like(upper_slops[id4])
    #     upper_intercepts[id4] = sig_ub[id4]

    return lower_slops, lower_intercepts, upper_slops, upper_intercepts


def sigmoid_parallel_verinet(lb, ub):
    # verinet bound
    assert lb.shape == ub.shape
    if ub.dim() == 0:
        ub = torch.unsqueeze(ub, dim=0)
        lb = torch.unsqueeze(lb, dim=0)

    sig_lb = sigmoid(lb)
    sig_ub = sigmoid(ub)
    k = (sig_ub - sig_lb) / (ub - lb)
    sig_p_lb = sig_lb * (1 - sig_lb)
    sig_p_ub = sig_ub * (1 - sig_ub)

    upper_slops = torch.zeros_like(lb)
    upper_intercepts = torch.zeros_like(lb)
    lower_slops = torch.zeros_like(lb)
    lower_intercepts = torch.zeros_like(lb)

    mid = (ub + lb) / 2
    sig_mid = sigmoid(mid)
    sig_p_mid = sig_mid * (1 - sig_mid)

    id4 = torch.where(k == 0)[0]

    uid1 = torch.where(sig_p_ub >= k)[0]
    uid2 = torch.where(torch.logical_and(sig_p_ub < k, sig_p_mid * (lb - mid) + sig_mid >= sig_lb))[0]
    uid3 = torch.where(torch.logical_and(sig_p_ub < k, sig_p_mid * (lb - mid) + sig_mid < sig_lb))[0]
    lid1 = torch.where(sig_p_lb >= k)[0]
    lid2 = torch.where(torch.logical_and(sig_p_lb < k, sig_p_mid * (ub - mid) + sig_mid <= sig_ub))[0]
    lid3 = torch.where(torch.logical_and(sig_p_lb < k, sig_p_mid * (ub - mid) + sig_mid > sig_ub))[0]

    if uid1.shape[0] != 0:
        upper_slops[uid1] = k[uid1]
        upper_intercepts[uid1] = sig_lb[uid1] - k[uid1] * lb[uid1]
    if uid2.shape[0] != 0:
        upper_slops[uid2] = sig_p_mid[uid2]
        upper_intercepts[uid2] = sig_mid[uid2] - sig_p_mid[uid2] * mid[uid2]
    if uid3.shape[0] != 0:
        upper_d, upper_sig_d = find_t_sigmoid_parallel(lb[uid3], torch.zeros_like(lb[uid3]), ub[uid3], t_sig=sig_lb[uid3])
        upper_slops[uid3] = upper_sig_d * (1 - upper_sig_d)
        upper_intercepts[uid3] = sig_lb[uid3] - upper_slops[uid3] * lb[uid3]
    if lid1.shape[0] != 0:
        lower_slops[lid1] = k[lid1]
        lower_intercepts[lid1] = sig_ub[lid1] - k[lid1] * ub[lid1]
    if lid2.shape[0] != 0:
        lower_slops[lid2] = sig_p_mid[lid2]
        lower_intercepts[lid2] = sig_mid[lid2] - sig_p_mid[lid2] * mid[lid2]
    if lid3.shape[0] != 0:
        lower_d, lower_sig_d = find_t_sigmoid_parallel(ub[lid3], lb[lid3], torch.zeros_like(ub[lid3]), t_sig=sig_ub[lid3])
        lower_slops[lid3] = lower_sig_d * (1 - lower_sig_d)
        lower_intercepts[lid3] = sig_ub[lid3] - lower_slops[lid3] * ub[lid3]

    if id4.shape[0] != 0:
        upper_intercepts[id4] = sig_ub[id4]
        lower_intercepts[id4] = sig_lb[id4]

    return lower_slops, lower_intercepts, upper_slops, upper_intercepts


if __name__ == '__main__':
    lowerb = torch.tensor([-4.1811, -6, 0, 1, -3, -5], dtype=torch.float)
    upperb = torch.tensor([29.0880, 10, 2, 2, 0, 0.3], dtype=torch.float)
    ls, li, us, ui = sigmoid_parallel(lowerb, upperb)
    ls1, li1, us1, ui1 = sigmoid_single(lowerb, upperb)
    print(ls == ls1)
    print(li == li1)
    print(us == us1)
    print(ui == ui1)
    # t = sigmoid(torch.tensor(0))
