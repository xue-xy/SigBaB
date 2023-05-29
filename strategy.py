import torch
from domain import split_domain
from activation_bound import sigmoid_p, sigmoid, sigmoid_parallel
from domain import DOMCOMP
from model_verification import VModel


def neuron_split_zero(lb, ub):
    if lb < 0 and ub > 0:
        return [(lb, torch.tensor(0.0)), (torch.tensor(0.0), ub)]
    else:
        return [(lb, (lb + ub)/2), ((lb + ub)/2, ub)]


def neuron_split_zero_multi(lb, ub):
    pass


def neuron_split_zero_complete(lb, ub):
    if lb < 0 < ub:
        mid = torch.zeros_like(lb)
    else:
        mid = (lb + ub) / 2
    bound_list = [sigmoid_parallel(lb, mid), sigmoid_parallel(mid, ub)]
    return [(lb, mid), (mid, ub)], bound_list


def neuron_split_half(lb, ub):
    return [(lb, (lb + ub)/2), ((lb + ub)/2, ub)]


def neuron_split_half_complete(lb, ub):
    bound_list = [sigmoid_parallel(lb, (lb+ub)/2), sigmoid_parallel((lb+ub)/2, ub)]
    return [(lb, (lb + ub) / 2), ((lb + ub) / 2, ub)], bound_list


def neuron_split_inflection_sigmoid(lb, ub):
    # ln(2+sqrt(3)) and ln(2-sqrt(3))
    lp = torch.log(torch.tensor(2) + torch.sqrt(torch.tensor(3)))
    up = torch.log(torch.tensor(2) - torch.sqrt(torch.tensor(3)))

    if lb < lp:
        if ub < lp:
            return [(lb, (lb + ub)/2), ((lb + ub)/2, ub)]
        elif ub < up:
            return [(lb, lp), (lp, ub)]
        else:
            return [(lb, lp), (lp, up), (up, ub)]
    elif lb < up:
        if ub < up:
            return [(lb, (lb + ub)/2), ((lb + ub)/2, ub)]
        else:
            return [(lb, up), (up, ub)]
    else:
        return [(lb, (lb + ub)/2), ((lb + ub)/2, ub)]


def neuron_split_convex_sigmoid(lb, ub, l_slop, l_intercept, u_slop, u_intercept):
    '''
    split to two part for region takes both positive and negative part
    :param lb:
    :param ub:
    :param l_slop:
    :param l_intercept:
    :param u_slop:
    :param u_intercept:
    :return:
    '''
    sig_lb = sigmoid(lb)
    sig_ub = sigmoid(ub)
    # k = (sig_ub - sig_lb) / (ub - lb)
    sig_p_lb = sig_lb * (1 - sig_lb)
    sig_p_ub = sig_ub * (1 - sig_ub)
    if lb < 0 and ub > 0:
        bound_list = [(l_slop, l_intercept, (sig_lb - 0.5) / lb, torch.tensor(0.5, device=lb.device)),
                      ((sig_ub - 0.5) / ub, torch.tensor(0.5, device=lb.device), u_slop, u_intercept)]
        return [(lb, torch.tensor(0.0, device=lb.device)), (torch.tensor(0.0, device=lb.device), ub)], bound_list
        # if sig_p_lb < k and sig_p_ub > k:
        #     bound_list = [(l_slop, l_intercept, (sig_lb - 0.5)/lb, torch.tensor(0.5, device=lb.device)),
        #                   ((sig_ub - 0.5)/ub, torch.tensor(0.5, device=lb.device), u_slop, u_intercept)]
        #     return [(lb, torch.tensor(0.0, device=lb.device)), (torch.tensor(0.0, device=lb.device), ub)], bound_list
        # elif sig_p_lb > k and sig_p_ub < k:
        #     bound_list = [(l_slop, l_intercept, (sig_lb - 0.5)/lb, torch.tensor(0.5, device=lb.device)),
        #                   ((sig_ub - 0.5)/ub, torch.tensor(0.5, device=lb.device), u_slop, u_intercept)]
        #     return [(lb, torch.tensor(0.0, device=lb.device)), (torch.tensor(0.0, device=lb.device), ub)], bound_list
        # else:
        #     bound_list = [(l_slop, l_intercept, (sig_lb - 0.5)/lb, torch.tensor(0.5, device=lb.device)),
        #                   ((sig_ub - 0.5)/ub, torch.tensor(0.5, device=lb.device), u_slop, u_intercept)]
        #     return [(lb, torch.tensor(0.0, device=lb.device)), (torch.tensor(0.0, device=lb.device), ub)], bound_list
    elif ub < 0:
        if l_slop == sig_p_lb or l_slop == sig_p_ub:
            d = (sig_p_ub * ub - sig_p_lb * lb + sig_lb - sig_ub) / (sig_p_ub - sig_p_lb)
            sig_d = sigmoid(d)
            k1 = (sig_d - sig_lb) / (d - lb)
            k2 = (sig_ub - sig_d) / (ub - d)
            bound_list = [(sig_p_lb, sig_lb - sig_p_lb * lb, k1, sig_lb - k1 * lb),
                          (sig_p_ub, sig_ub - sig_p_ub * ub, k2, sig_ub - k2 * ub)]
            return [(lb, d), (d, ub)], bound_list
        else:
            d1 = (l_intercept - sig_lb + sig_p_lb * lb) / (sig_p_lb - l_slop)
            d2 = (l_intercept - sig_ub + sig_p_ub * ub) / (sig_p_ub - l_slop)
            k1 = (sigmoid(d1) - sig_lb) / (d1 - lb)
            k2 = (sigmoid(d2) - sigmoid(d1)) / (d2 - d1)
            k3 = (sig_ub - sigmoid(d2)) / (ub - d2)
            bound_list = [(sig_p_lb, sig_lb - sig_p_lb * lb, k1, sig_lb - k1 * lb),
                          (l_slop, l_intercept, k2, sigmoid(d1) - k2 * d1),
                          (sig_p_ub, sig_ub - sig_p_ub * ub, k3, sig_ub - k3 * ub)]
            return [(lb, d1), (d1, d2), (d2, ub)], bound_list
    else:
        if u_slop == sig_p_lb or u_slop == sig_p_ub:
            d = (sig_p_ub * ub - sig_p_lb * lb + sig_lb - sig_ub) / (sig_p_ub - sig_p_lb)
            sig_d = sigmoid(d)
            k1 = (sig_d - sig_lb) / (d - lb)
            k2 = (sig_ub - sig_d) / (ub - d)
            bound_list = [(k1, sig_lb - k1 * lb, sig_p_lb, sig_lb - sig_p_lb * lb),
                          (k2, sig_ub - k2 * ub, sig_p_ub, sig_ub - sig_p_ub * ub)]
            return [(lb, d), (d, ub)], bound_list
        else:
            d1 = (u_intercept - sig_lb + sig_p_lb * lb) / (sig_p_lb - u_slop)
            d2 = (u_intercept - sig_ub + sig_p_ub * ub) / (sig_p_ub - u_slop)
            k1 = (sigmoid(d1) - sig_lb) / (d1 - lb)
            k2 = (sigmoid(d2) - sigmoid(d1)) / (d2 - d1)
            k3 = (sig_ub - sigmoid(d2)) / (ub - d2)
            bound_list = [(k1, sig_lb - k1 * lb, sig_p_lb, sig_lb - sig_p_lb * lb),
                          (k2, sigmoid(d1) - k2 * d1, u_slop, u_intercept),
                          (k3, sig_ub - k3 * ub, sig_p_ub, sig_ub - sig_p_ub * ub)]
            return [(lb, d1), (d1, d2), (d2, ub)], bound_list


def branch_max_interval(bounds, layer_id=None):
    if layer_id is None:
        intervals_max = [torch.max(bounds[i][1] - bounds[i][0]) for i in range(len(bounds))]
        layer_id = torch.argmax(torch.tensor(intervals_max)).item()
        neuron_id = torch.argmax(bounds[layer_id][1] - bounds[layer_id][0]).item()
        return [layer_id, neuron_id]
    else:
        neuron_id = torch.argmax(bounds[layer_id][1] - bounds[layer_id][0]).item()
        return [layer_id, neuron_id]


def branch_max_interval_multi(bounds, num, layer_id=None):
    '''
    only works with neurons number in all layers are same
    '''
    if layer_id is None:
        layer_neuron_num = bounds[0][0].shape[0]
        intervals_max = torch.cat([bounds[i][1] - bounds[i][0] for i in range(len(bounds))])
        _, neuron_ids = torch.topk(intervals_max, num)
        layer_ids = neuron_ids // layer_neuron_num
        neuron_ids = neuron_ids % layer_neuron_num
        return list(zip(layer_ids.tolist(), neuron_ids.tolist()))
    else:
        _, neuron_ids = torch.topk(bounds[layer_id][1] - bounds[layer_id][0], num)
        neuron_ids = neuron_ids.tolist()
        return [[layer_id, neuron_ids[i]] for i in range(len(neuron_ids))]


def branch_integral_single(bounds, layer_id=None):
    if layer_id is None:
        values = []
        ids = []
        for i in range(len(bounds)):
            val, idx = torch.max(integral_second_sigmoid(bounds[i][0], bounds[i][1]), dim=0)
            values.append(val.item())
            ids.append(idx.item())
        layer_id = torch.argmax(torch.tensor(values)).item()
        neuron_id = ids[layer_id]
    else:
        scores = integral_second_sigmoid(bounds[layer_id][0], bounds[layer_id][1])
        neuron_id = torch.argmax(scores).item()
    return [layer_id, neuron_id]


def integral_second_sigmoid(lb, ub):
    neg_id = torch.where(ub <= 0)[0]
    pos_id = torch.where(lb >= 0)[0]
    cross_id = torch.where(torch.logical_and(lb < 0, ub > 0))[0]

    score = torch.zeros_like(lb)

    score[neg_id] = sigmoid_p(ub[neg_id]) - sigmoid_p(lb[neg_id])
    score[pos_id] = sigmoid_p(lb[pos_id]) - sigmoid_p(ub[pos_id])
    score[cross_id] = 0.5*torch.ones_like(lb[cross_id]) - sigmoid_p(lb[cross_id]) - sigmoid_p(ub[cross_id])

    return score


def branch_improvement(vmodel: VModel, domc: DOMCOMP, c, split_method, layer_id=None):
    coeff = torch.matmul(c, vmodel.weights[-1])
    const = torch.matmul(c, vmodel.biases[-1])
    val = 0

    for i in range(len(domc.bounds) - 1, -1, -1):
        if split_method == 'half':
            new_val, idx = improvement_score_half(coeff, vmodel.biases[i], domc.bounds[i], domc.slops[i], domc.intercepts[i])
        elif split_method == 'zero':
            new_val, idx = improvement_score_zero(coeff, vmodel.biases[i], domc.bounds[i], domc.slops[i], domc.intercepts[i])
        elif split_method == 'convex':
            new_val, idx = improvement_score_convex(coeff, vmodel.biases[i], domc.bounds[i], domc.slops[i], domc.intercepts[i])

        # print(new_val, idx)
        if new_val > val:
            val = new_val
            neuron_idx = idx
            layer_idx = i

        const = torch.sum(torch.maximum(coeff, torch.zeros_like(coeff)) * domc.intercepts[i][0]) + \
                torch.sum(torch.minimum(coeff, torch.zeros_like(coeff)) * domc.intercepts[i][1]) + \
                const
        coeff = torch.maximum(coeff, torch.zeros_like(coeff)) * domc.slops[i][0] + \
                torch.minimum(coeff, torch.zeros_like(coeff)) * domc.slops[i][1]
        const = torch.matmul(coeff, vmodel.biases[i]) + const
        coeff = torch.matmul(coeff, vmodel.weights[i])

    return layer_idx, neuron_idx


def improvement_score_half(coeff, bias, bounds, slops, intercepts):
    mid = (bounds[0] + bounds[1]) / 2

    # lower half
    ls, li, us, ui = sigmoid_parallel(bounds[0], mid)
    score_1 = torch.maximum(coeff, torch.zeros_like(coeff)) * (bias * (ls - slops[0]) + li - intercepts[0]) + \
              torch.minimum(coeff, torch.zeros_like(coeff)) * (bias * (us - slops[1]) + ui - intercepts[1])
    # upper half
    ls, li, us, ui = sigmoid_parallel(mid, bounds[1])
    score_2 = torch.maximum(coeff, torch.zeros_like(coeff)) * (bias * (ls - slops[0]) + li - intercepts[0]) + \
              torch.minimum(coeff, torch.zeros_like(coeff)) * (bias * (us - slops[1]) + ui - intercepts[1])

    val, idx = torch.max(torch.maximum(score_1, score_2), dim=0)
    # print(score_2 > 0)
    # quit()
    return val, idx


def improvement_score_zero(coeff, bias, bounds, slops, intercepts):
    mid = torch.where(torch.logical_and(bounds[0]<0, bounds[1]>0), torch.zeros_like(bounds[0]), (bounds[0] + bounds[1])/2)

    # lower half
    ls, li, us, ui = sigmoid_parallel(bounds[0], mid)
    score_1 = torch.maximum(coeff, torch.zeros_like(coeff)) * (bias * (ls - slops[0]) + li - intercepts[0]) + \
              torch.minimum(coeff, torch.zeros_like(coeff)) * (bias * (us - slops[1]) + ui - intercepts[1])
    # upper half
    ls, li, us, ui = sigmoid_parallel(mid, bounds[1])
    score_2 = torch.maximum(coeff, torch.zeros_like(coeff)) * (bias * (ls - slops[0]) + li - intercepts[0]) + \
              torch.minimum(coeff, torch.zeros_like(coeff)) * (bias * (us - slops[1]) + ui - intercepts[1])
    val, idx = torch.max(torch.maximum(score_1, score_2), dim=0)
    return val, idx


def improvement_score_convex(coeff, bias, bounds, slops, intercepts):
    # babsr
    score = torch.zeros_like(coeff)
    sig_lb = sigmoid(bounds[0])
    sig_ub = sigmoid(bounds[1])
    k = (sig_ub - sig_lb) / (bounds[1] - bounds[0])
    sig_p_lb = sig_lb * (1 - sig_lb)
    sig_p_ub = sig_ub * (1 - sig_ub)

    id1 = torch.where(torch.logical_and(bounds[0] < 0, bounds[1] > 0))[0]
    id2 = torch.where(bounds[1] < 0)[0]
    id3 = torch.where(bounds[0] > 0)[0]
    id4 = torch.where(k == 0)[0]

    if id1.shape[0] != 0:
        score_1 = torch.minimum(coeff[id1], torch.zeros_like(coeff[id1])) * \
                  (bias[id1] * ((sig_lb[id1] - 0.5)/bounds[0][id1] - slops[1][id1]) + 0.5 - intercepts[1][id1])
        score_2 = torch.maximum(coeff[id1], torch.zeros_like(coeff[id1])) * \
                  (bias[id1] * ((sig_ub[id1] - 0.5)/bounds[1][id1] - slops[0][id1]) + 0.5 - intercepts[0][id1])
        score[id1] = torch.maximum(score_1, score_2)
        # score[id1] = (score_1 + score_2) / 2
    if id2.shape[0] != 0:
        cond = torch.logical_or(slops[0] == sig_p_lb, slops[0] == sig_p_ub)
        id2_1 = torch.where(torch.logical_and(cond, bounds[1] < 0))[0]
        if id2_1.shape[0] != 0:
            d = (sig_p_ub[id2_1] * bounds[1][id2_1] - sig_p_lb[id2_1] * bounds[0][id2_1] + sig_lb[id2_1] - sig_ub[id2_1]) / (sig_p_ub[id2_1] - sig_p_lb[id2_1])
            sig_d = sigmoid(d)
            k1 = (sig_d - sig_lb[id2_1]) / (d - bounds[0][id2_1])
            k2 = (sig_ub[id2_1] - sig_d) / (bounds[1][id2_1] - d)
            score_1 = torch.maximum(coeff[id2_1], torch.zeros_like(coeff[id2_1])) * (bias[id2_1] * (sig_p_lb[id2_1] - slops[0][id2_1]) + sig_lb[id2_1] - sig_p_lb[id2_1] * bounds[0][id2_1] - intercepts[0][id2_1]) + \
                      torch.minimum(coeff[id2_1], torch.zeros_like(coeff[id2_1])) * (bias[id2_1] * (k1 - slops[1][id2_1]) + sig_lb[id2_1] - k1 * bounds[0][id2_1] - intercepts[1][id2_1])
            score_2 = torch.maximum(coeff[id2_1], torch.zeros_like(coeff[id2_1])) * (bias[id2_1] * (sig_p_ub[id2_1] - slops[0][id2_1]) + sig_ub[id2_1] - sig_p_ub[id2_1] * bounds[1][id2_1] - intercepts[0][id2_1]) + \
                      torch.minimum(coeff[id2_1], torch.zeros_like(coeff[id2_1])) * (bias[id2_1] * (k2 - slops[1][id2_1]) + sig_ub[id2_1] - k2 * bounds[1][id2_1] - intercepts[1][id2_1])
            score[id2_1] = torch.maximum(score_1, score_2)
            # score[id2_1] = (score_1 + score_2) / 2
        id2_2 = torch.where(torch.logical_and(torch.logical_not(cond), bounds[1] < 0))[0]
        if id2_2.shape[0] != 0:
            d1 = (intercepts[0][id2_2] - sig_lb[id2_2] + sig_p_lb[id2_2] * bounds[0][id2_2]) / (sig_p_lb[id2_2] - slops[0][id2_2])
            d2 = (intercepts[0][id2_2] - sig_ub[id2_2] + sig_p_ub[id2_2] * bounds[1][id2_2]) / (sig_p_ub[id2_2] - slops[0][id2_2])
            k1 = (sigmoid(d1) - sig_lb[id2_2]) / (d1 - bounds[0][id2_2])
            k2 = (sigmoid(d2) - sigmoid(d1)) / (d2 - d1)
            k3 = (sig_ub[id2_2] - sigmoid(d2)) / (bounds[1][id2_2] - d2)
            score_1 = torch.maximum(coeff[id2_2], torch.zeros_like(coeff[id2_2])) * (bias[id2_2] * (sig_p_lb[id2_2] - slops[0][id2_2]) + sig_lb[id2_2] - sig_p_lb[id2_2] * bounds[0][id2_2] - intercepts[0][id2_2]) + \
                      torch.minimum(coeff[id2_2], torch.zeros_like(coeff[id2_2])) * (bias[id2_2] * (k1 - slops[1][id2_2]) + sig_lb[id2_2] - k1 * bounds[0][id2_2] - intercepts[1][id2_2])
            score_2 = torch.minimum(coeff[id2_2], torch.zeros_like(coeff[id2_2])) * (bias[id2_2] * (k2 - slops[1][id2_2]) + sigmoid(d1) - k2 * d1 - intercepts[1][id2_2])
            score_3 = torch.maximum(coeff[id2_2], torch.zeros_like(coeff[id2_2])) * (bias[id2_2] * (sig_p_ub[id2_2] - slops[0][id2_2]) + sig_ub[id2_2] - sig_p_ub[id2_2] * bounds[1][id2_2] - intercepts[0][id2_2]) + \
                      torch.minimum(coeff[id2_2], torch.zeros_like(coeff[id2_2])) * (bias[id2_2] * (k3 - slops[1][id2_2]) + sig_ub[id2_2] - k3 * bounds[1][id2_2] - intercepts[1][id2_2])
            score[id2_2] = torch.maximum(score_1, torch.maximum(score_2, score_3))
            # score[id2_2] = (score_1 + score_2 + score_3) / 3
    if id3.shape[0] != 0:
        cond = torch.logical_or(slops[0] == sig_p_lb, slops[0] == sig_p_ub)
        id3_1 = torch.where(torch.logical_and(cond, bounds[0] > 0))[0]
        if id3_1.shape[0] != 0:
            d = (sig_p_ub[id3_1] * bounds[1][id3_1] - sig_p_lb[id3_1] * bounds[0][id3_1] + sig_lb[id3_1] - sig_ub[id3_1]) / (sig_p_ub[id3_1] - sig_p_lb[id3_1])
            sig_d = sigmoid(d)
            k1 = (sig_d - sig_lb[id3_1]) / (d - bounds[0][id3_1])
            k2 = (sig_ub[id3_1] - sig_d) / (bounds[1][id3_1] - d)
            score_1 = torch.maximum(coeff[id3_1], torch.zeros_like(coeff[id3_1])) * (bias[id3_1] * (k1 - slops[0][id3_1]) + sig_lb[id3_1] - k1 * bounds[0][id3_1] - intercepts[0][id3_1]) + \
                      torch.minimum(coeff[id3_1], torch.zeros_like(coeff[id3_1])) * (bias[id3_1] * (sig_p_lb[id3_1] - slops[1][id3_1]) + sig_lb[id3_1] - sig_p_lb[id3_1] * bounds[0][id3_1] - intercepts[1][id3_1])
            score_2 = torch.maximum(coeff[id3_1], torch.zeros_like(coeff[id3_1])) * (bias[id3_1] * (k2 - slops[0][id3_1]) + sig_ub[id3_1] - k2 * bounds[1][id3_1] - intercepts[0][id3_1]) + \
                      torch.minimum(coeff[id3_1], torch.zeros_like(coeff[id3_1])) * (bias[id3_1] * (sig_p_ub[id3_1] - slops[1][id3_1]) + sig_ub[id3_1] - sig_p_ub[id3_1] * bounds[1][id3_1] - intercepts[1][id3_1])
            score[id3_1] = torch.maximum(score_1, score_2)
            # score[id3_1] = (score_1 + score_2) / 2
        id3_2 = torch.where(torch.logical_and(torch.logical_not(cond), bounds[0] > 0))[0]
        if id3_2.shape[0] != 0:
            d1 = (intercepts[1][id3_2] - sig_lb[id3_2] + sig_p_lb[id3_2] * bounds[0][id3_2]) / (sig_p_lb[id3_2] - slops[1][id3_2])
            d2 = (intercepts[1][id3_2] - sig_ub[id3_2] + sig_p_ub[id3_2] * bounds[1][id3_2]) / (sig_p_ub[id3_2] - slops[1][id3_2])
            k1 = (sigmoid(d1) - sig_lb[id3_2]) / (d1 - bounds[0][id3_2])
            k2 = (sigmoid(d2) - sigmoid(d1)) / (d2 - d1)
            k3 = (sig_ub[id3_2] - sigmoid(d2)) / (bounds[1][id3_2] - d2)
            score_1 = torch.maximum(coeff[id3_2], torch.zeros_like(coeff[id3_2])) * (bias[id3_2] * (k1 - slops[0][id3_2]) + sig_lb[id3_2] - k1 * bounds[0][id3_2] - intercepts[0][id3_2]) + \
                      torch.minimum(coeff[id3_2], torch.zeros_like(coeff[id3_2])) * (bias[id3_2] * (sig_p_lb[id3_2] - slops[1][id3_2]) + sig_lb[id3_2] - sig_p_lb[id3_2] * bounds[0][id3_2] - intercepts[1][id3_2])
            score_2 = torch.maximum(coeff[id3_2], torch.zeros_like(coeff[id3_2])) * (bias[id3_2] * (k2 - slops[0][id3_2]) + sigmoid(d1) - k2 * d1 - intercepts[0][id3_2])
            score_3 = torch.maximum(coeff[id3_2], torch.zeros_like(coeff[id3_2])) * (bias[id3_2] * (k3 - slops[0][id3_2]) + sig_ub[id3_2] - k3 * bounds[1][id3_2] - intercepts[0][id3_2]) + \
                      torch.minimum(coeff[id3_2], torch.zeros_like(coeff[id3_2])) * (bias[id3_2] * (sig_p_ub[id3_2] - slops[1][id3_2]) + sig_ub[id3_2] - sig_p_ub[id3_2] * bounds[1][id3_2] - intercepts[1][id3_2])
            score[id3_2] = torch.maximum(score_1, torch.maximum(score_2, score_3))
            # score[id3_2] = (score_1 + score_2 + score_3) / 3
    # if id4.shape[0] != 0:
    #     score[id4] = 0.0
    score = torch.where(torch.isnan(score), torch.zeros_like(score), score)
    val, idx = torch.max(score, dim=0)
    return val, idx


def improvement_score_convex_full(coeff, bias, bounds, slops, intercepts):
    score = torch.zeros_like(coeff)
    sig_lb = sigmoid(bounds[0])
    sig_ub = sigmoid(bounds[1])
    k = (sig_ub - sig_lb) / (bounds[1] - bounds[0])
    sig_p_lb = sig_lb * (1 - sig_lb)
    sig_p_ub = sig_ub * (1 - sig_ub)

    id1 = torch.where(torch.logical_and(bounds[0] < 0, bounds[1] > 0))[0]
    id2 = torch.where(bounds[1] < 0)[0]
    id3 = torch.where(bounds[0] > 0)[0]
    id4 = torch.where(k == 0)[0]

    if id1.shape[0] != 0:
        score_1 = torch.minimum(coeff[id1], torch.zeros_like(coeff[id1])) * \
                  (bounds[1][id1] * ((sig_lb[id1] - 0.5)/bounds[0][id1] - slops[1][id1]) + 0.5 - intercepts[1][id1])
        score_2 = torch.maximum(coeff[id1], torch.zeros_like(coeff[id1])) * \
                  (bounds[0][id1] * ((sig_ub[id1] - 0.5)/bounds[1][id1] - slops[0][id1]) + 0.5 - intercepts[0][id1])
        # score[id1] = torch.maximum(score_1, score_2)
        score[id1] = (score_1 + score_2) / 2
    if id2.shape[0] != 0:
        cond = torch.logical_or(slops[0] == sig_p_lb, slops[0] == sig_p_ub)
        id2_1 = torch.where(torch.logical_and(cond, bounds[1] < 0))[0]
        if id2_1.shape[0] != 0:
            d = (sig_p_ub[id2_1] * bounds[1][id2_1] - sig_p_lb[id2_1] * bounds[0][id2_1] + sig_lb[id2_1] - sig_ub[id2_1]) / (sig_p_ub[id2_1] - sig_p_lb[id2_1])
            sig_d = sigmoid(d)
            k1 = (sig_d - sig_lb[id2_1]) / (d - bounds[0][id2_1])
            k2 = (sig_ub[id2_1] - sig_d) / (bounds[1][id2_1] - d)
            score_1 = torch.maximum(coeff[id2_1], torch.zeros_like(coeff[id2_1])) * (bounds[0][id2_1] * (sig_p_lb[id2_1] - slops[0][id2_1]) + sig_lb[id2_1] - sig_p_lb[id2_1] * bounds[0][id2_1] - intercepts[0][id2_1]) + \
                      torch.minimum(coeff[id2_1], torch.zeros_like(coeff[id2_1])) * (bounds[1][id2_1] * (k1 - slops[1][id2_1]) + sig_lb[id2_1] - k1 * bounds[0][id2_1] - intercepts[1][id2_1])
            score_2 = torch.maximum(coeff[id2_1], torch.zeros_like(coeff[id2_1])) * (bounds[0][id2_1] * (sig_p_ub[id2_1] - slops[0][id2_1]) + sig_ub[id2_1] - sig_p_ub[id2_1] * bounds[1][id2_1] - intercepts[0][id2_1]) + \
                      torch.minimum(coeff[id2_1], torch.zeros_like(coeff[id2_1])) * (bounds[1][id2_1] * (k2 - slops[1][id2_1]) + sig_ub[id2_1] - k2 * bounds[1][id2_1] - intercepts[1][id2_1])
            # score[id2_1] = torch.maximum(score_1, score_2)
            score[id2_1] = (score_1 + score_2) / 2
        id2_2 = torch.where(torch.logical_and(torch.logical_not(cond), bounds[1] < 0))[0]
        if id2_2.shape[0] != 0:
            d1 = (intercepts[0][id2_2] - sig_lb[id2_2] + sig_p_lb[id2_2] * bounds[0][id2_2]) / (sig_p_lb[id2_2] - slops[0][id2_2])
            d2 = (intercepts[0][id2_2] - sig_ub[id2_2] + sig_p_ub[id2_2] * bounds[1][id2_2]) / (sig_p_ub[id2_2] - slops[0][id2_2])
            k1 = (sigmoid(d1) - sig_lb[id2_2]) / (d1 - bounds[0][id2_2])
            k2 = (sigmoid(d2) - sigmoid(d1)) / (d2 - d1)
            k3 = (sig_ub[id2_2] - sigmoid(d2)) / (bounds[1][id2_2] - d2)
            score_1 = torch.maximum(coeff[id2_2], torch.zeros_like(coeff[id2_2])) * (bounds[0][id2_2] * (sig_p_lb[id2_2] - slops[0][id2_2]) + sig_lb[id2_2] - sig_p_lb[id2_2] * bounds[0][id2_2] - intercepts[0][id2_2]) + \
                      torch.minimum(coeff[id2_2], torch.zeros_like(coeff[id2_2])) * (bounds[1][id2_2] * (k1 - slops[1][id2_2]) + sig_lb[id2_2] - k1 * bounds[0][id2_2] - intercepts[1][id2_2])
            score_2 = torch.minimum(coeff[id2_2], torch.zeros_like(coeff[id2_2])) * (bounds[1][id2_2] * (k2 - slops[1][id2_2]) + sigmoid(d1) - k2 * d1 - intercepts[1][id2_2])
            score_3 = torch.maximum(coeff[id2_2], torch.zeros_like(coeff[id2_2])) * (bounds[0][id2_2] * (sig_p_ub[id2_2] - slops[0][id2_2]) + sig_ub[id2_2] - sig_p_ub[id2_2] * bounds[1][id2_2] - intercepts[0][id2_2]) + \
                      torch.minimum(coeff[id2_2], torch.zeros_like(coeff[id2_2])) * (bounds[1][id2_2] * (k3 - slops[1][id2_2]) + sig_ub[id2_2] - k3 * bounds[1][id2_2] - intercepts[1][id2_2])
            # score[id2_2] = torch.maximum(score_1, torch.maximum(score_2, score_3))
            score[id2_2] = (score_1 + score_2 + score_3) / 3
    if id3.shape[0] != 0:
        cond = torch.logical_or(slops[0] == sig_p_lb, slops[0] == sig_p_ub)
        id3_1 = torch.where(torch.logical_and(cond, bounds[0] > 0))[0]
        if id3_1.shape[0] != 0:
            d = (sig_p_ub[id3_1] * bounds[1][id3_1] - sig_p_lb[id3_1] * bounds[0][id3_1] + sig_lb[id3_1] - sig_ub[id3_1]) / (sig_p_ub[id3_1] - sig_p_lb[id3_1])
            sig_d = sigmoid(d)
            k1 = (sig_d - sig_lb[id3_1]) / (d - bounds[0][id3_1])
            k2 = (sig_ub[id3_1] - sig_d) / (bounds[1][id3_1] - d)
            score_1 = torch.maximum(coeff[id3_1], torch.zeros_like(coeff[id3_1])) * (bounds[0][id3_1] * (k1 - slops[0][id3_1]) + sig_lb[id3_1] - k1 * bounds[0][id3_1] - intercepts[0][id3_1]) + \
                      torch.minimum(coeff[id3_1], torch.zeros_like(coeff[id3_1])) * (bounds[1][id3_1] * (sig_p_lb[id3_1] - slops[1][id3_1]) + sig_lb[id3_1] - sig_p_lb[id3_1] * bounds[0][id3_1] - intercepts[1][id3_1])
            score_2 = torch.maximum(coeff[id3_1], torch.zeros_like(coeff[id3_1])) * (bounds[0][id3_1] * (k2 - slops[0][id3_1]) + sig_ub[id3_1] - k2 * bounds[1][id3_1] - intercepts[0][id3_1]) + \
                      torch.minimum(coeff[id3_1], torch.zeros_like(coeff[id3_1])) * (bounds[1][id3_1] * (sig_p_ub[id3_1] - slops[1][id3_1]) + sig_ub[id3_1] - sig_p_ub[id3_1] * bounds[1][id3_1] - intercepts[1][id3_1])
            # score[id3_1] = torch.maximum(score_1, score_2)
            score[id3_1] = (score_1 + score_2) / 2
        id3_2 = torch.where(torch.logical_and(torch.logical_not(cond), bounds[0] > 0))[0]
        if id3_2.shape[0] != 0:
            d1 = (intercepts[1][id3_2] - sig_lb[id3_2] + sig_p_lb[id3_2] * bounds[0][id3_2]) / (sig_p_lb[id3_2] - slops[1][id3_2])
            d2 = (intercepts[1][id3_2] - sig_ub[id3_2] + sig_p_ub[id3_2] * bounds[1][id3_2]) / (sig_p_ub[id3_2] - slops[1][id3_2])
            k1 = (sigmoid(d1) - sig_lb[id3_2]) / (d1 - bounds[0][id3_2])
            k2 = (sigmoid(d2) - sigmoid(d1)) / (d2 - d1)
            k3 = (sig_ub[id3_2] - sigmoid(d2)) / (bounds[1][id3_2] - d2)
            score_1 = torch.maximum(coeff[id3_2], torch.zeros_like(coeff[id3_2])) * (bounds[0][id3_2] * (k1 - slops[0][id3_2]) + sig_lb[id3_2] - k1 * bounds[0][id3_2] - intercepts[0][id3_2]) + \
                      torch.minimum(coeff[id3_2], torch.zeros_like(coeff[id3_2])) * (bounds[1][id3_2] * (sig_p_lb[id3_2] - slops[1][id3_2]) + sig_lb[id3_2] - sig_p_lb[id3_2] * bounds[0][id3_2] - intercepts[1][id3_2])
            score_2 = torch.maximum(coeff[id3_2], torch.zeros_like(coeff[id3_2])) * (bounds[0][id3_2] * (k2 - slops[0][id3_2]) + sigmoid(d1) - k2 * d1 - intercepts[0][id3_2])
            score_3 = torch.maximum(coeff[id3_2], torch.zeros_like(coeff[id3_2])) * (bounds[0][id3_2] * (k3 - slops[0][id3_2]) + sig_ub[id3_2] - k3 * bounds[1][id3_2] - intercepts[0][id3_2]) + \
                      torch.minimum(coeff[id3_2], torch.zeros_like(coeff[id3_2])) * (bounds[1][id3_2] * (sig_p_ub[id3_2] - slops[1][id3_2]) + sig_ub[id3_2] - sig_p_ub[id3_2] * bounds[1][id3_2] - intercepts[1][id3_2])
            # score[id3_2] = torch.maximum(score_1, torch.maximum(score_2, score_3))
            score[id3_2] = (score_1 + score_2 + score_3) / 3
    # if id4.shape[0] != 0:
    #     score[id4] = 0.0
    score = torch.where(torch.isnan(score), torch.zeros_like(score), score)
    val, idx = torch.max(score, dim=0)
    return val, idx


def split_dom_to_doms_single(dom, origin_bounds, branch='integral', split='zero', layer_id=None):
    dom_bounds = dom.merge_bound(origin_bounds)
    if branch == 'max':
        layer_idx, neuron_idx = branch_max_interval(dom_bounds, layer_id=layer_id)
    elif branch == 'integral':
        layer_idx, neuron_idx = branch_integral_single(dom_bounds, layer_id=layer_id)

    lb = dom_bounds[layer_idx][0][neuron_idx]
    ub = dom_bounds[layer_idx][1][neuron_idx]
    # print(ub - lb)
    if split == 'half':
        interval_split = neuron_split_half(lb, ub)
    elif split == 'inflection':
        interval_split = neuron_split_inflection_sigmoid(lb, ub)
    elif split == 'zero':
        interval_split = neuron_split_zero(lb, ub)
    new_doms = split_domain(dom, [layer_idx, neuron_idx], interval_split)

    return new_doms


if __name__ == '__main__':
    bds = []
    for i in range(5):
        bds.append([torch.rand(20)-1, torch.rand(20)])
    li, ni = branch_integral_single(bds)
    print(li, ni)

