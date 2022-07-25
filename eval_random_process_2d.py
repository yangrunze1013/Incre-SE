import se_graph as sg
import graph_generator as gg
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pickle
import time
import math
import copy


def evaluation_pipeline_2d(seq, time_step_num, seg: sg.SEGraph):
    eval_seg = copy.deepcopy(seg)

    # start
    pipeline_start_time = time.time()
    GI_list = []
    LD_list = []
    SE_list = []

    init_start_time = time.time()
    eval_seg.update_division_Louvain()
    eval_seg.update_struc_data()
    eval_seg.calc_expression_SD()
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

        # count incremental data and calculate 2dSE
        t_time_start = time.time()
        eval_seg.get_incre_data(subseq)
        GI2d = eval_seg.fast_GI2d(size)
        LD2d = eval_seg.calc_LD2d()
        SE2d = GI2d + LD2d
        t_time_end = time.time()
        t_time = t_time_end - t_time_start
        t_time_list.append(t_time)

        GI_list.append(GI2d)
        LD_list.append(LD2d)
        SE_list.append(SE2d)

    pipeline_end_time = time.time()
    pipeline_time_cost = pipeline_end_time - pipeline_start_time
    print('Total time cost: ' + str(pipeline_time_cost))

    return SE_list, t_time_list

def traditional_pipeline_2d(seq, time_step_num, seg: sg.SEGraph):
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
        eval_seg.update_division_Louvain()
        SE_list.append(eval_seg.calc_2dSE())
        t_time_end = time.time()
        t_time = t_time_end - t_time_start
        t_time_list.append(t_time)

    pipeline_end_time = time.time()
    pipeline_time_cost = pipeline_end_time - pipeline_start_time
    print('Traditional total time cost: ' + str(pipeline_time_cost))

    return SE_list, t_time_list

def save_SE_time_list(seq, T, seg, name):
    # 预热
    print('heating...')
    evaluation_pipeline_2d(seq, T, seg)

    print('save begin...')
    t_array = None
    t_trad_array = None
    se_array = None
    se_trad_array = None

    repeat_times = 5
    for i in range(repeat_times):
        SE, t = evaluation_pipeline_2d(seq, T, seg)
        SE_trad, t_trad = traditional_pipeline_2d(seq, T, seg)
        if i == 0:
            t_array = np.array([t])
            t_trad_array = np.array([t_trad])
            se_array = np.array([SE], dtype=np.float64)
            se_trad_array = np.array([SE_trad], dtype=np.float64)
        else:
            t_array = np.append(t_array, [t], axis=0)
            t_trad_array = np.append(t_trad_array, [t_trad], axis=0)
            se_array = np.append(se_array, [SE], axis=0)
            se_trad_array = np.append(se_trad_array, [SE_trad], axis=0)

    save_list(f'./results/{name}_2d_logs/{name}_time.txt', t_array)
    save_list(f'./results/{name}_2d_logs/{name}_time_trad.txt', t_trad_array)
    save_list(f'./results/{name}_2d_logs/{name}_SE.txt', se_array)
    save_list(f'./results/{name}_2d_logs/{name}_SE_trad.txt', se_trad_array)
    print(f'saved {name}, times {i}')
    print('save over')


def save_list(path, list):
    np.savetxt(path, list)


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
    # ax.plot(axis_x, time_mean, color='orange', marker='v')
    # ax.plot(axis_x, time_mean, color='red', marker='v')
    ax.bar(axis_x - 1, time_mean, width=1.7, color='blue', edgecolor='black')

    # init_time = np.ones(20) * 0.0698
    # ax.plot(axis_x, init_time, color='blue', linestyle='--')

    # ax.plot(axis_x, time_trad_mean, color='burlywood', marker='^')
    # ax.plot(axis_x, time_trad_mean, color='blue', marker='^')
    ax.bar(axis_x + 1, time_trad_mean, width=1.7, color='lightgreen', edgecolor='black')

    ax.grid()
    ax.legend(['Incre-2dSE', '2d-RFS'], fontsize='medium',
              loc=(0.7 / width, 2.29 / height))

    ax2 = ax.twinx()
    acc_ratio = time_trad_mean / time_mean
    ax2.set_ylim(0, 1.05 * np.max(acc_ratio))
    ax2.set_ylabel('Speedup', fontsize=13, color='red')
    # ax2.plot(axis_x, acc_ratio, color='darkkhaki', marker='o')
    ax2.plot(axis_x, acc_ratio, color='red', marker='o')
    ax2.legend(['Speedup'], fontsize='medium', loc='upper right')
    plt.savefig(f'./results/figures_random_process_2d/{name}_time_2d.pdf', bbox_inches='tight')
    plt.show()


def draw_mean_error(SE, SE_trad, T, name, save=False):
    # draw mean error figure (relative error)
    SE_relative_error = (SE - SE_trad) / SE_trad * 100
    # SE_relative_error = (SE/SE_trad - 1)/axis_x * 100
    mean_error = SE_relative_error.mean(axis=0)
    std_error = SE_relative_error.std(axis=0)
    print(f'process name: {name}')
    for i in range(len(mean_error)):
        print(f'mean {i + 1}: {mean_error[i]}, std {i + 1}: {std_error[i]}')
    axis_x = np.array(range(T)) / T * 100 + 5
    fig, ax = plt.subplots(figsize=(12, 4), dpi=300)
    ax.set_xlabel('Incremental Percentage (%)', fontsize=13)
    ax.set_ylabel('Mean Relative Error (%)', fontsize=13)
    ax.set_xticks(axis_x)
    # ax.set_ylim(-1.2*mean_error[-1], 1.2*mean_error[-1])
    ax.plot(axis_x, mean_error, color='orange', marker='v')
    ax.plot(axis_x, std_error, color='burlywood', marker='^')
    ax.grid()
    ax.legend(['Mean Value', 'Standard Deviation'], fontsize='medium', loc='upper left')
    if save:
        plt.savefig(f'./results/figures_random_process_2d/{name}_error_2d', bbox_inches='tight')
    plt.show()

def save_init_mean_time():
    # read initial state
    with open('./save_random_process/init_state', 'rb') as file:
        saved_graph = pickle.load(file)
    seg = sg.SEGraph(saved_graph)

    # get initalization time cost
    mean, std = get_init_mean_time_cost(seg, 5)
    print(mean, std)

def save_time_SE():
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

    # 2d evaluation:
    # save SE & time list
    save_SE_time_list(hawkes_seq, time_step_num, seg, 'hawkes')
    save_SE_time_list(triad_seq, time_step_num, seg, 'triad')
    save_SE_time_list(random_seq, time_step_num, seg, 'random')

    # get initalization time cost
    mean, std = get_init_mean_time_cost(seg, 5)
    print(mean, std)


def draw_time_figure():
    # read files & calculate means and std
    hawkes_time = np.loadtxt(f'./results/hawkes_2d_logs/hawkes_time.txt')
    hawkes_time_trad = np.loadtxt(f'./results/hawkes_2d_logs/hawkes_time_trad.txt')

    triad_time = np.loadtxt(f'./results/triad_2d_logs/triad_time.txt')
    triad_time_trad = np.loadtxt(f'./results/triad_2d_logs/triad_time_trad.txt')

    random_time = np.loadtxt(f'./results/random_2d_logs/random_time.txt')
    random_time_trad = np.loadtxt(f'./results/random_2d_logs/random_time_trad.txt')

    T = hawkes_time.shape[1]

    hm = hawkes_time.mean(axis=0)
    hv = hawkes_time.var(axis=0)
    htm = hawkes_time_trad.mean(axis=0)
    htv = hawkes_time_trad.var(axis=0)
    har = htm / hm

    tm = triad_time.mean(axis=0)
    tv = triad_time.var(axis=0)
    ttm = triad_time_trad.mean(axis=0)
    ttv = triad_time_trad.var(axis=0)
    tar = ttm / tm

    rm = random_time.mean(axis=0)
    rv = random_time.var(axis=0)
    rtm = random_time_trad.mean(axis=0)
    rtv = random_time_trad.var(axis=0)
    rar = rtm / rm

    draw_mean_time_cost(hm, htm, T, 'hawkes')
    draw_mean_time_cost(tm, ttm, T, 'triad')
    draw_mean_time_cost(rm, rtm, T, 'random')

    # draw_mean_var_time_cost(hm, htm, hv, htv, T, 'hawkes')
    # draw_mean_var_time_cost(tm, ttm, tv, ttv, T, 'triad')
    # draw_mean_var_time_cost(rm, rtm, rv, rtv, T, 'random')


def draw_error_figure():
    # read files & calculate means and std
    hawkes_se = np.loadtxt(f'./results/hawkes_2d_logs/hawkes_SE.txt')
    hawkes_se_trad = np.loadtxt(f'./results/hawkes_2d_logs/hawkes_SE_trad.txt')

    triad_se = np.loadtxt(f'./results/triad_2d_logs/triad_SE.txt')
    triad_se_trad = np.loadtxt(f'./results/triad_2d_logs/triad_SE_trad.txt')

    random_se = np.loadtxt(f'./results/random_2d_logs/random_SE.txt')
    random_se_trad = np.loadtxt(f'./results/random_2d_logs/random_SE_trad.txt')

    T = hawkes_se.shape[1]

    draw_mean_error(hawkes_se, hawkes_se_trad, T, 'hawkes')
    draw_mean_error(triad_se, triad_se_trad, T, 'triad')
    draw_mean_error(random_se, random_se_trad, T, 'random')

def get_init_mean_time_cost(seg, repeat):
    time_list = []
    for i in range(repeat):
        eval_seg = copy.deepcopy(seg)
        eval_seg.update_division_Louvain()
        # Initialization: calculate 2d structural expressions
        init_start_time = time.time()
        eval_seg.update_struc_data()
        eval_seg.calc_expression_SD()
        init_end_time = time.time()
        init_time = init_end_time - init_start_time
        time_list.append(init_time)
        print(init_time)
    time_array = np.array(time_list)
    mean = time_array.mean()
    std = time_array.std()
    return mean, std

if __name__ == '__main__':
    # save_time_SE()
    draw_time_figure()
    # draw_error_figure()
    # save_init_mean_time()

    # # example graph
    # g = nx.Graph()
    # g.add_edges_from([('a1', 'a2'), ('a1', 'a3'), ('a2', 'a3'),
    #                   ('a3', 'b1'), ('b1', 'b2'), ('b1', 'b4'),
    #                   ('b2', 'b3'), ('b3', 'b4')])
    #
    # seg = sg.SEGraph(g)
    # seg.show_division()
    # print("1dSE: ", seg.calc_1dSE())
    #
    # #############DEMO#############
    # seg.division = {'v1': ['a1', 'a2', 'a3'], 'v2': ['b1', 'b2', 'b3', 'b4']}
    # for node in seg.division['v1']:
    #     seg.graph.nodes[node]['comm'] = 'v1'
    # for node in seg.division['v2']:
    #     seg.graph.nodes[node]['comm'] = 'v2'
    # ##############################
    #
    # print("2dSE: ", seg.calc_2dSE())
    #
    # seq = [('b2', 'b4'), ('a3', 'b4'), ('a2', 'a4'), ('a4', 'a3')]
    # moment_statistic_pipeline_2d(seq, 4, seg)
