import torch
from model.models import *
from model_verification import VModel
from domain import *
from strategy import *
from pending import *
import heapq
import time


def robustness(vmodel: VModel, x, label, eps, args, norm='inf', verbose=False):
    '''
    :param vmodel:
    :param x:
    :param label:
    :param eps:
    :param norm:
    :return: 1 for verified, -1 for counterexample found, 0 for unknown
    '''
    # Currently only works for 10 classes
    x = x.to(args.device)

    ans = [0] * 10
    ans[label] = 1
    res = torch.ones(10)

    for i in range(10):
        if i == label:
            continue

        c = torch.zeros(10, dtype=torch.float, device=args.device)
        c[label] = 1
        c[i] = -1
        val, pce = vmodel.backward_propagation(x, eps, c, torch.tensor(0))
        res[i] = val

        # print(val)
        if val > 0:
            ans[i] = 1
        else:
            if vmodel.check(pce, c) <= 0:
                ans[i] = -1
            elif args.bab:
                bab_res = bab_verification_parallel_bfs_complete(vmodel, x, eps, c, args)
                # if args.split == 'convex':
                #     bab_res = bab_verification_parallel_bfs_complete(vmodel, x, eps, c, args)
                # else:
                #     bab_res = bab_verification_parallel_bfs(vmodel, x, eps, c, args)
                # bab_res = bab_verification_parallel(vmodel, x, eps, c, args)
                # bab_res = bab_verification(vmodel, x, eps, c)
                ans[i] = bab_res
        # print('-'*40)
        # quit()

    undecidable = ans.count(0)
    prove_false = ans.count(-1)
    prove_true = 9 - undecidable - prove_false

    if verbose:
        print(res)
    # print(ans)
    if -1 in ans:
        return -1, prove_true, prove_false, undecidable
    elif sum(ans) == 10:
        return 1, prove_true, prove_false, undecidable
    else:
        return 0, prove_true, prove_false, undecidable


def bab_verification(vmodel: VModel, x, eps, c, time_limit=30):
    init_domain = DOMNEW(vmodel.layer_num, [vmodel.biases[i].shape[0] for i in range(vmodel.layer_num)])
    unsat_domains = [(0, init_domain)]
    heapq.heapify(unsat_domains)
    start_time = time.time()
    count = 0

    while len(unsat_domains) != 0:
        dom = heapq.heappop(unsat_domains)[1]

        layer_idx, neuron_idx = branch_max_interval(dom.merge_bound(vmodel.hidden_bounds))

        lb = vmodel.hidden_bounds[layer_idx][0][neuron_idx]
        ub = vmodel.hidden_bounds[layer_idx][1][neuron_idx]

        interval_split = neuron_split_half(lb, ub)
        new_doms = split_domain(dom, [layer_idx, neuron_idx], interval_split)

        for ndom in new_doms:
            tval, pce = vmodel.bab_propagation_new(x, eps, c, ndom)

            count += 1
            # print(tval)

            if tval < 0:
                if vmodel.check(pce, c) <= 0:
                    return -1
                heapq.heappush(unsat_domains, [-1*tval, ndom])
            # if tval < -2:
                # tval, pce = vmodel.bab_propagation(x, eps, c, ndom)
            #     print('-'*20)
            #     print(vmodel.backward_bab_simple(x, eps, c, ndom))
            #     print(ndom.idxes)
            #     print(ndom.additional_bounds)
            #     print(tval)
            #     quit()
        ite_time = time.time()
        if ite_time - start_time > time_limit:
            break
    print(count)
    print(ite_time - start_time)
    maxi = heapq.nlargest(1, unsat_domains)
    print(maxi)
    print(len(unsat_domains))
    if len(unsat_domains) == 0:
        return 1
    else:
        return 0


def bab_verification_parallel(vmodel: VModel, x, eps, c, args, neuron_num=1):
    init_domain = DOMNEW(vmodel.layer_num, [vmodel.biases[i].shape[0] for i in range(vmodel.layer_num)], args.device)
    unsat_domains = LIST([[0, init_domain]])
    start_time = time.time()
    count = 0
    batch_size = 2 ** neuron_num
    depth = 0

    bad_idx = 0

    while len(unsat_domains) != 0:
        loop_start = time.time()
        # print(len(unsat_domains))
        origin_val, dom = unsat_domains.get()
        origin_val *= -1

        dom_bounds = dom.merge_bound(vmodel.hidden_bounds)
        choice_list = branch_max_interval_multi(dom_bounds, neuron_num)

        # lb = [vmodel.hidden_bounds[layer_idx][0][neuron_idx] for layer_idx, neuron_idx in choice_list]
        # ub = [vmodel.hidden_bounds[layer_idx][1][neuron_idx] for layer_idx, neuron_idx in choice_list]
        lb = [dom_bounds[layer_idx][0][neuron_idx] for layer_idx, neuron_idx in choice_list]
        ub = [dom_bounds[layer_idx][1][neuron_idx] for layer_idx, neuron_idx in choice_list]
        intervals_split = [neuron_split_half(lb[i], ub[i]) for i in range(len(lb))]
        new_doms = split_domain_multi(dom, choice_list, intervals_split)

        selected_domains = new_doms
        dom_batch = make_domnew_batch(selected_domains)

        process_time = time.time()

        tval, pce = vmodel.bab_propagation_parallel(x, eps, c, dom_batch, batch_size, args.device)

        # print(tval)
        # print(choice_list)
        # print('-'*30)

        # print(tval)
        # print(process_time - loop_start, time.time() - loop_start)

        count += batch_size
        depth += neuron_num

        # if (tval < origin_val).any():
        #     bad_idx += 1
        #     print(origin_val, bad_idx)
        #     print(tval)
        #     print('-'*40)
        #     if bad_idx == 2:
        #         from activation_bound import sigmoid_single, sigmoid
        #         print(choice_list)
        #         print(lb, ub)
        #
        #         instance_bounds = dom.merge_bound(vmodel.hidden_bounds)
        #         # 1, 266; 0, 365
        #         print('original:')
        #         olb = instance_bounds[1][0][266]
        #         oub = instance_bounds[1][1][266]
        #         print(olb, oub)
        #         olb1 = instance_bounds[0][0][365]
        #         oub1 = instance_bounds[0][1][365]
        #         print(olb1, oub1)
        #         print(sigmoid_single(olb1, oub1))
        #         # print(sigmoid(olb), sigmoid(oub))
        #         print('*-'*20)
        #         print('children:')
        #         # for ten in dom_batch.idxes:
        #         #     print(torch.nonzero(ten))
        #         # print(torch.nonzero(dom_batch.idxes[1]))
        #         nlb = dom_batch.lower_bounds[1][0, 266]
        #         nub = dom_batch.upper_bounds[1][0, 266]
        #         print(nlb, nub)
        #         nlb1 = dom_batch.lower_bounds[0][0, 365]
        #         nub1 = dom_batch.upper_bounds[0][0, 365]
        #         print(nlb1, nub1)
        #         print(sigmoid_single(nlb1, nub1))
        #         # print(sigmoid(nlb), sigmoid(nub))
        #
        #         print('-*'*20)
        #         print('single_dom:')
        #         choice_list = choice_list[1:2]
        #         print(choice_list)
        #         lb = [dom_bounds[layer_idx][0][neuron_idx] for layer_idx, neuron_idx in choice_list]
        #         ub = [dom_bounds[layer_idx][1][neuron_idx] for layer_idx, neuron_idx in choice_list]
        #         intervals_split = [neuron_split_half(lb[i], ub[i]) for i in range(len(lb))]
        #         new_doms = split_domain_multi(dom, choice_list, intervals_split)
        #         dom_batch = make_domnew_batch(new_doms)
        #         tval, pce = vmodel.bab_propagation_parallel(x, eps, c, dom_batch, 2, verbose=True)
        #         print(tval)
        #         quit()

        unknown_idx = torch.where(tval < 0)[0]
        if unknown_idx.shape[0] > 0:
            if (vmodel.check(pce, c) <= 0).any():
                return -1
            unknown_idx_list = unknown_idx.tolist()
            for un_idx in unknown_idx_list:
                unsat_domains.push([-1*tval[un_idx], selected_domains[un_idx]])
        ite_time = time.time()

        if ite_time - start_time > args.tlimit:
            break
    print(count)
    # maxi = heapq.nlargest(1, unsat_domains)
    # print(maxi)
    # print(len(unsat_domains))
    quit()
    if len(unsat_domains) == 0:
        return 1
    else:
        return 0


def bab_verification_parallel_bfs(vmodel: VModel, x, eps, c, args):
    init_domain = DOMNEW(vmodel.layer_num, [vmodel.biases[i].shape[0] for i in range(vmodel.layer_num)], args.device)
    unsat_domains = LIST([[0, init_domain]])
    start_time = time.time()
    count = 0
    depth = 0

    bad_idx = 0

    while len(unsat_domains) != 0:
        loop_start = time.time()
        # print(len(unsat_domains))
        # origin_val, dom = unsat_domains.get()
        # origin_val *= -1

        candidate_doms = []
        while len(candidate_doms) < args.batch_size and len(unsat_domains) > 0:
            _, dom = unsat_domains.get()
            candidate_doms.extend(split_dom_to_doms_single(dom, vmodel.hidden_bounds, branch=args.branch, split=args.split))
        this_batch_size = len(candidate_doms)

        dom_batch = make_domnew_batch(candidate_doms)

        process_time = time.time()

        tval, pce = vmodel.bab_propagation_parallel(x, eps, c, dom_batch, this_batch_size, args.device)
        # print(tval)
        # print('-'*30)

        # print(tval)
        # print(process_time - loop_start, time.time() - loop_start)

        count += this_batch_size

        # if (tval < origin_val).any():
        #     bad_idx += 1
        #     print(origin_val, bad_idx)
        #     print(tval)
        #     print('-'*40)
        #     if bad_idx == 2:
        #         from activation_bound import sigmoid_single, sigmoid
        #         print(choice_list)
        #         print(lb, ub)
        #
        #         instance_bounds = dom.merge_bound(vmodel.hidden_bounds)
        #         # 1, 266; 0, 365
        #         print('original:')
        #         olb = instance_bounds[1][0][266]
        #         oub = instance_bounds[1][1][266]
        #         print(olb, oub)
        #         olb1 = instance_bounds[0][0][365]
        #         oub1 = instance_bounds[0][1][365]
        #         print(olb1, oub1)
        #         print(sigmoid_single(olb1, oub1))
        #         # print(sigmoid(olb), sigmoid(oub))
        #         print('*-'*20)
        #         print('children:')
        #         # for ten in dom_batch.idxes:
        #         #     print(torch.nonzero(ten))
        #         # print(torch.nonzero(dom_batch.idxes[1]))
        #         nlb = dom_batch.lower_bounds[1][0, 266]
        #         nub = dom_batch.upper_bounds[1][0, 266]
        #         print(nlb, nub)
        #         nlb1 = dom_batch.lower_bounds[0][0, 365]
        #         nub1 = dom_batch.upper_bounds[0][0, 365]
        #         print(nlb1, nub1)
        #         print(sigmoid_single(nlb1, nub1))
        #         # print(sigmoid(nlb), sigmoid(nub))
        #
        #         print('-*'*20)
        #         print('single_dom:')
        #         choice_list = choice_list[1:2]
        #         print(choice_list)
        #         lb = [dom_bounds[layer_idx][0][neuron_idx] for layer_idx, neuron_idx in choice_list]
        #         ub = [dom_bounds[layer_idx][1][neuron_idx] for layer_idx, neuron_idx in choice_list]
        #         intervals_split = [neuron_split_half(lb[i], ub[i]) for i in range(len(lb))]
        #         new_doms = split_domain_multi(dom, choice_list, intervals_split)
        #         dom_batch = make_domnew_batch(new_doms)
        #         tval, pce = vmodel.bab_propagation_parallel(x, eps, c, dom_batch, 2, verbose=True)
        #         print(tval)
        #         quit()

        unknown_idx = torch.where(tval < 0)[0]
        if unknown_idx.shape[0] > 0:
            if (vmodel.check(pce, c) <= 0).any():
                return -1
            unknown_idx_list = unknown_idx.tolist()
            for un_idx in unknown_idx_list:
                unsat_domains.push([-1*tval[un_idx], candidate_doms[un_idx]])
        ite_time = time.time()

        if ite_time - start_time > args.tlimit:
            break
    # print(count)
    # maxi = heapq.nlargest(1, unsat_domains)
    # print(maxi)
    # print(len(unsat_domains))
    if len(unsat_domains) == 0:
        return 1
    else:
        return 0


def bab_verification_parallel_bfs_complete(vmodel: VModel, x, eps, c, args):
    init_domain = DOMCOMP(vmodel)
    unsat_domains = LIST([[0, init_domain]])
    start_time = time.time()
    count = 0
    depth = 0

    while len(unsat_domains) != 0:
        loop_start = time.time()
        # print(len(unsat_domains))
        # origin_val, dom = unsat_domains.get()
        # origin_val *= -1


        candidate_doms = []
        while len(candidate_doms) < args.batch_size and len(unsat_domains) > 0:
            _, dom = unsat_domains.get()
            if args.branch == 'max':
                layer_idx, neuron_idx = branch_max_interval(dom.bounds)
            elif args.branch == 'integral':
                layer_idx, neuron_idx = branch_integral_single(dom.bounds)
            elif args.branch == 'improvement':
                layer_idx, neuron_idx = branch_improvement(vmodel, dom, c, split_method=args.split)

            if args.split == 'half':
                interval_list, bound_list = neuron_split_half_complete(dom.bounds[layer_idx][0][neuron_idx], dom.bounds[layer_idx][1][neuron_idx])
            elif args.split == 'zero':
                interval_list, bound_list = neuron_split_zero_complete(dom.bounds[layer_idx][0][neuron_idx], dom.bounds[layer_idx][1][neuron_idx])
            elif args.split == 'convex':
                interval_list, bound_list = neuron_split_convex_sigmoid(dom.bounds[layer_idx][0][neuron_idx], dom.bounds[layer_idx][1][neuron_idx],
                                                                    dom.slops[layer_idx][0][neuron_idx], dom.intercepts[layer_idx][0][neuron_idx],
                                                                    dom.slops[layer_idx][1][neuron_idx], dom.intercepts[layer_idx][1][neuron_idx])
            candidate_doms.extend(split_domain_complete(dom, layer_idx, neuron_idx, interval_list, bound_list))

        this_batch_size = len(candidate_doms)

        dom_batch = make_domcomp_batch(candidate_doms)

        process_time = time.time()

        tval, pce = vmodel.bab_propagation_parallel_convex(x, eps, c, dom_batch, this_batch_size, args.device)
        # print('-'*30)

        # print(tval)
        # print(process_time - loop_start, time.time() - loop_start)

        count += this_batch_size

        unknown_idx = torch.where(tval < 0)[0]
        # print(unknown_idx)
        if unknown_idx.shape[0] > 0:
            if (vmodel.check(pce, c) <= 0).any():
                return -1
            unknown_idx_list = unknown_idx.tolist()
            for un_idx in unknown_idx_list:
                unsat_domains.push([-1*tval[un_idx], candidate_doms[un_idx]])
        ite_time = time.time()

        # if unknown_idx.shape[0] < this_batch_size:
        #     print('decrease')
        #     print('-'*40)

        if ite_time - start_time > args.tlimit:
            break
    print(count)
    # maxi = unsat_domains.get_max()
    # print(maxi)
    print(len(unsat_domains))
    print('-'*40)
    if len(unsat_domains) == 0:
        return 1
    else:
        return 0


if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    net = mnist_fnn_3_100()
    net.load_state_dict(torch.load('./model/saved_weights/mnist_fnn_3_100_weights.pth'))

    model = VModel(net, mean=0, std=1, device=device)

    test_data = MNIST('./data/', train=False, download=False, transform=ToTensor())
    data = torch.flatten(test_data.data / 255, start_dim=1)
    labels = test_data.targets

    torch.set_printoptions(linewidth=400)

    i = 96

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='choose a saved model')
    parser.add_argument('--dataset', default='mnist', help='dataset')
    parser.add_argument('--device', default='cuda:0', help='cpu or gpu')
    parser.add_argument('--eps', default=0.01, help='radius')
    parser.add_argument('--branch', default='improvement', choices=['max', 'integral', 'improvement'],
                        help='branching strategy')
    parser.add_argument('--split', default='convex', choices=['zero', 'half', 'inflection', 'convex'],
                        help='neuron split method')
    parser.add_argument('--bab', default=False, type=bool, help='whether or not to use bab')
    parser.add_argument('--batch_size', default=400, type=int, help='batch size')
    parser.add_argument('--tlimit', default=300, help='time limit for each property')
    args = parser.parse_args()

    l_ans, p_true, p_false, und = robustness(model, data[i], labels[i], args.eps, args)
    print(l_ans, p_true, p_false, und)

    # c = torch.zeros(10, dtype=torch.float32)
    # c[7] = 1
    # c[1] = -1
    # model.backward_propagation(data[0], 0.01, c, torch.tensor(0))
