import se_graph as sg
import graph_generator as gg
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import time
import copy
import math

# mpl.rc('font',family='Times New Roman') # Global Setting

def incre_update(incre_dict, key, value):
    if key in incre_dict:
        incre_dict[key] += value
    else:
        incre_dict[key] = value

def evaluation_pipeline_1d(seq, time_step_num, seg: sg.SEGraph):
    eval_seg = copy.deepcopy(seg)

    # start
    pipeline_start_time = time.time()
    GI_list = []
    LD_list = []
    SE_list = []

    # Initialization: calculate 1d structural expressions
    init_start_time = time.time()
    expression_1d = 0
    for node in eval_seg.graph.nodes:
        d = eval_seg.graph.degree[node]
        expression_1d += d * math.log2(d)
    m = eval_seg.m
    init_end_time = time.time()
    init_time = init_end_time - init_start_time
    print('Initialization time cost: ' + str(init_time))

    t_time_list = []  # time cost at time t
    length = len(seq)
    num = int(length / time_step_num)
    for t in range(time_step_num):
        if (t + 1) * num > length:
            size = length
        else:
            size = (t + 1) * num
        subseq = seq[:size]
        n = len(subseq)
        # count incremental data and calculate 1dSE
        t_time_start = time.time()
        degree_incre = {}
        for edge in subseq:
            n0 = edge[0]
            n1 = edge[1]
            incre_update(degree_incre, n0, 1)
            incre_update(degree_incre, n1, 1)
        GI1d = - 1 / (2 * m + 2 * n) * expression_1d + (2 * m) / (2 * m + 2 * n) * math.log2(2 * m + 2 * n)
        LD1d = 0
        for node in degree_incre.keys():
            delta_d = degree_incre[node]
            if node in eval_seg.graph.nodes:
                d = eval_seg.graph.degree[node]
                LD1d += d * math.log2(d) - (d + delta_d) * math.log2(d + delta_d)
            else:
                LD1d += - delta_d * math.log2(delta_d)

        LD1d += (2 * n) * math.log2(2 * m + 2 * n)
        LD1d *= 1 / (2 * m + 2 * n)
        SE1d = GI1d + LD1d
        GI_list.append(GI1d)
        LD_list.append(LD1d)
        SE_list.append(SE1d)

        t_time_end = time.time()
        t_time = t_time_end - t_time_start
        t_time_list.append(t_time)

    pipeline_end_time = time.time()
    pipeline_time_cost = pipeline_end_time - pipeline_start_time
    print('Total time cost: ' + str(pipeline_time_cost))
    return SE_list, t_time_list

def traditional_pipeline_1d(seq, time_step_num, seg: sg.SEGraph):
    pipeline_start_time = time.time()
    SE_list = []
    length = len(seq)
    n = int(length / time_step_num)
    t_time_list = []  # time cost at time t
    for t in range(time_step_num):
        if (t + 1) * n > length:
            size = length
        else:
            size = (t + 1) * n
        subseq = seq[:size]
        eval_seg = copy.deepcopy(seg)
        # 增量序列逐段到达，修改图，Louvain进行社区划分，并按定义计算结构熵
        t_time_start = time.time()
        eval_seg.incre_seq_arrival_only_graph(subseq)
        SE_list.append(eval_seg.calc_1dSE())
        t_time_end = time.time()
        t_time = t_time_end - t_time_start
        t_time_list.append(t_time)

    pipeline_end_time = time.time()
    pipeline_time_cost = pipeline_end_time - pipeline_start_time
    print('Traditional total time cost: ' + str(pipeline_time_cost))

    return SE_list, t_time_list

def main():
    # read initial state
    with open('./save_random_process/init_state', 'rb') as file:
        saved_graph = pickle.load(file)
    seg = sg.SEGraph(saved_graph)

    # read cumulative incremental sequence at T
    hawkes_seq = np.load('./save_random_process/hawkes_seq.npy').tolist()
    triad_seq = np.load('./save_random_process/triad_seq.npy').tolist()
    random_seq = np.load('./save_random_process/random_seq.npy').tolist()

    # settings
    time_step_num = 20

    # 1d evaluation:
    # save SE & time list
    save_SE_time_list(hawkes_seq, time_step_num, seg, 'hawkes')
    save_SE_time_list(triad_seq, time_step_num, seg, 'triad')
    save_SE_time_list(random_seq, time_step_num, seg, 'random')

    # get initalization time cost
    mean, std = get_init_mean_time_cost(seg)
    print(mean,std)

def get_init_mean_time_cost(seg):
    time_list = []
    for i in range(5):
        eval_seg = copy.deepcopy(seg)
        # Initialization: calculate 1d structural expressions
        init_start_time = time.time()
        expression_1d = 0
        for node in eval_seg.graph.nodes:
            d = eval_seg.graph.degree[node]
            expression_1d += d * math.log2(d)
        m = eval_seg.m
        init_end_time = time.time()
        init_time = init_end_time - init_start_time
        time_list.append(init_time)
        print(init_time)
    time_array = np.array(time_list)
    mean = time_array.mean()
    std = time_array.std()
    return mean, std

def save_SE_time_list(seq, T, seg, name):
    # 预热
    print('heating...')
    evaluation_pipeline_1d(seq, T, seg)
    evaluation_pipeline_1d(seq, T, seg)
    evaluation_pipeline_1d(seq, T, seg)

    print('save begin...')
    t_array = None
    t_trad_array = None

    repeat_times = 5
    for i in range(repeat_times):
        SE, t = evaluation_pipeline_1d(seq, T, seg)
        SE_trad, t_trad = traditional_pipeline_1d(seq, T, seg)
        if i == 0:
            t_array = np.array([t])
            t_trad_array = np.array([t_trad])
        else:
            t_array = np.append(t_array, [t], axis=0)
            t_trad_array = np.append(t_trad_array, [t_trad], axis=0)

    save_list(f'./results/{name}_1d_logs/{name}_time.txt', t_array)
    save_list(f'./results/{name}_1d_logs/{name}_time_trad.txt', t_trad_array)
    print(t_array)
    print(t_trad_array)
    print(f'saved {name}, times {i}')
    print('save over')

def save_list(path, list):
    np.savetxt(path, list)

def draw_time_figure():
    # read files & calculate means and std
    hawkes_time = np.loadtxt(f'./results/hawkes_1d_logs/hawkes_time.txt')
    hawkes_time_trad = np.loadtxt(f'./results/hawkes_1d_logs/hawkes_time_trad.txt')

    triad_time = np.loadtxt(f'./results/triad_1d_logs/triad_time.txt')
    triad_time_trad = np.loadtxt(f'./results/triad_1d_logs/triad_time_trad.txt')

    random_time = np.loadtxt(f'./results/random_1d_logs/random_time.txt')
    random_time_trad = np.loadtxt(f'./results/random_1d_logs/random_time_trad.txt')

    T = hawkes_time.shape[1]

    hm = hawkes_time.mean(axis=0)
    hv = hawkes_time.std(axis=0)
    htm = hawkes_time_trad.mean(axis=0)
    htv = hawkes_time_trad.std(axis=0)
    har = htm/hm

    tm = triad_time.mean(axis=0)
    tv = triad_time.std(axis=0)
    ttm = triad_time_trad.mean(axis=0)
    ttv = triad_time_trad.std(axis=0)
    tar = ttm/tm

    rm = random_time.mean(axis=0)
    rv = random_time.std(axis=0)
    rtm = random_time_trad.mean(axis=0)
    rtv =random_time_trad.std(axis=0)
    rar = rtm/rm

    draw_mean_time_cost(hm, htm, T, 'hawkes')
    draw_mean_time_cost(tm, ttm, T, 'triad')
    draw_mean_time_cost(rm, rtm, T, 'random')


def draw_mean_time_cost(time_mean, time_trad_mean, T, name):
    # draw mean time figure
    axis_x = np.array(range(T)) / T * 100 + 5
    width = 6
    height = 3
    fig, ax = plt.subplots(figsize=(width, height), dpi=300)
    ax.set_xlabel('Incremental Percentage (%)', fontsize=13)
    ax.set_ylabel('Time Cost (s)', fontsize=13)
    ax.set_xticks(axis_x)
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 1.2 * np.max(time_trad_mean))
    #ax.plot(axis_x, time_mean, color='orange', marker='v')
    #ax.plot(axis_x, time_mean, color='red', marker='v')
    ax.bar(axis_x-1, time_mean, width=1.7, color='blue', edgecolor='black')

    # init_time = np.ones(20) * 1.70
    # ax.plot(axis_x, init_time, color='navajowhite', linestyle='--')
    # ax.plot(axis_x, init_time, color='red', linestyle='--')

    # ax.plot(axis_x, time_trad_mean, color='burlywood', marker='^')
    # ax.plot(axis_x, time_trad_mean, color='blue', marker='^')
    ax.bar(axis_x+1, time_trad_mean,width=1.7, color='lightgreen', edgecolor='black')

    ax.grid()
    ax.legend(['Incre-1dSE', '1d-RFS'], fontsize='medium', loc=(0.1/width, 2.29/height))

    ax2 = ax.twinx()
    acc_ratio = time_trad_mean / time_mean
    ax2.set_ylim(0, np.max(acc_ratio))
    ax2.set_ylabel('Speedup', fontsize = 13, color = 'red')
    # ax2.plot(axis_x, acc_ratio, color='darkkhaki', marker='o')
    ax2.plot(axis_x, acc_ratio, color='red', marker='o')
    ax2.set_ylim(np.min(acc_ratio)-0.5, 1.25 * np.max(acc_ratio))
    ax2.legend(['Speedup'], fontsize='medium', loc='upper right')
    plt.savefig(f'./results/figures_random_process_1d/{name}_time_1d.pdf', bbox_inches='tight')
    plt.show()

# def draw_mean_time_cost(time_mean, time_trad_mean, T, name):
#     # draw main figure
#     axis_x = np.array(range(T)) / T * 100 + 5
#     fig, ax = plt.subplots(figsize=(12, 4), dpi=300)
#     ax.set_xlabel('Incremental Percentage (%)', fontsize=13)
#     ax.set_ylabel('Time Cost (s)', fontsize=13)
#     ax.set_xticks(axis_x)
#     ax.set_ylim(0,1.5*np.max(time_trad_mean))
#     ax.plot(axis_x, time_mean, color='orange', marker='v')
#     ax.plot(axis_x, time_trad_mean, color='burlywood', marker='^')
#     # init_time = np.ones(20)*0.00323
#     # ax.plot(axis_x, init_time, color = 'navajowhite', linestyle = ':')
#     ax.grid()
#     ax.legend(['Incre-1dSE', 'One-dimensional RFS'], fontsize='medium', loc = 'upper left')
#
#     ax2 = ax.twinx()
#     acc_ratio = time_trad_mean / time_mean
#     ax2.set_ylim(0, np.max(acc_ratio))
#     ax2.plot(axis_x, acc_ratio, color='darkkhaki', marker='o')
#     ax2.legend(['Accelerate Ratio'], fontsize='medium', loc='upper right')
#     # plt.savefig(f'./results/figures_random_process_1d/{name}_1d', bbox_inches = 'tight')
#     plt.show()

def test():
    g = nx.Graph()
    g.add_edges_from([('a1', 'a2'), ('a1', 'a3'), ('a2', 'a3'),
                      ('a3', 'b1'), ('b1', 'b2'), ('b1', 'b4'),
                      ('b2', 'b3'), ('b3', 'b4')])

    seg = sg.SEGraph(g)
    seg.print_graph()
    print("1dSE: ", seg.calc_1dSE())
    seq = [('a2', 'a5'), ('b2', 'b4'), ('a5', 'a3'), ('a5', 'b1')]
    SE, time = evaluation_pipeline_1d(seq, 4, seg)
    SE_trad, time_trad = traditional_pipeline_1d(seq, 4, seg)
    print(SE)
    print(SE_trad)

    seg.incre_seq_arrival_only_graph(seq)
    print("1dSE: ", seg.calc_1dSE())
    pass

if __name__ == '__main__':
    # main()
    draw_time_figure()
