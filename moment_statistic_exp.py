import se_graph as sg
import graph_generator as gg
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pickle
import math
import copy
import random

XSTART = 480
XEND = 24000
XNUM = 100

SAMPLE_NUM = 30
SAMPLE_MEAN = 100
SAMPLE_STD = 10

def draw_plot_moment_statistic_1d():
    LD_list = np.loadtxt('./results/moment_statistic_logs/LD_list_1d_exp.txt')
    ubLD_list = np.loadtxt('./results/moment_statistic_logs/ubLD_list_1d_exp.txt')
    m_list = np.loadtxt('./results/moment_statistic_logs/m_list_1d_exp.txt')
    print(m_list)
    LD_mean = LD_list.mean(axis=1)
    LD_max = LD_list.max(axis=1)
    LD_min = LD_list.min(axis=1)
    # LD_var = LD_list.var(axis=1)

    ubLD_mean = ubLD_list.mean(axis=1)
    ubLD_max = ubLD_list.max(axis=1)
    ubLD_min = ubLD_list.min(axis=1)
    # ubLD_var = ubLD_list.var(axis=1)

    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    # axis_x = np.linspace(XSTART, XEND, XNUM)
    axis_x = np.array(m_list)
    ax.plot(axis_x, LD_mean, color='blue') # original color C0/C1
    ax.fill_between(axis_x, LD_min, LD_max, alpha=0.2, color='blue')
    ax.plot(axis_x, ubLD_mean, color='red')
    ax.fill_between(axis_x, ubLD_min, ubLD_max, alpha=0.2, color='red')

    fontsize = 13
    ax.set_ylabel('One-dimensional Local Difference', fontsize=fontsize)
    ax.set_xlabel('Total edge number of the original graph', fontsize=fontsize)
    # ax.grid()
    ax.legend(['Mean Local Difference', 'Max/min Local Difference',
               'Mean upper bound','Max/min  upper bound'], fontsize=fontsize, loc='upper right')
    ax.set_ylim(bottom = 0)
    plt.xscale('log')
    plt.savefig('./results/moment_statistic_logs/moment_plot_figure_1d_exp.pdf', bbox_inches='tight')
    plt.show()
    print(LD_mean[0], LD_mean[-1], (LD_mean[0]-LD_mean[-1])/LD_mean[0])

def save_moment_statistic_1d():
    # load original graph
    with open('./save_random_process/moment_statistic/init_state_moment', 'rb') as file:
        saved_graph = pickle.load(file)
    seg = sg.SEGraph(saved_graph)
    print(len(seg.graph.nodes))
    print(len(seg.graph.edges))
    # load the cumulative incremental sequence
    random_seq = np.load('./save_random_process/moment_statistic/random_seq_moment.npy').tolist()
    print(len(random_seq))

    # count & save
    moment_statistic_pipeline_1d(random_seq, XNUM, seg)

def moment_statistic_pipeline_1d(seq, time_step_num, seg: sg.SEGraph):
    LD_list = []  # 2d-array of shape sample_num * time_step_num
    ubLD_list = []  # 2d-array of shape sample_num * time_step_num
    m_list = []  # (total edge number list) 1d-array of shape time_step_num
    length = len(seq)

    start_log_m = math.log10(XSTART)
    end_log_m = math.log10(XEND)
    log_m_list = np.linspace(start_log_m, end_log_m, XNUM)
    size_list = (np.power(10, log_m_list)-XSTART).astype(np.int)
    print(size_list)

    t = 0
    for size in size_list:
        print('step:', t)
        t+=1
        cumseq = seq[:size]
        print('cumseq length', len(cumseq))

        eval_seg = copy.deepcopy(seg)
        eval_seg.incre_seq_arrival_only_graph(cumseq)

        eval_seg.update_division_Louvain()
        eval_seg.update_struc_data()
        eval_seg.calc_expression_SD()

        mean_n = SAMPLE_MEAN
        std_n = SAMPLE_STD
        sample_num = SAMPLE_NUM
        incre_size_list = np.random.normal(loc=mean_n, scale=std_n, size=sample_num)
        LD_array = []
        ubLD_array = []
        for n in incre_size_list:
            calc_seg = copy.deepcopy(eval_seg)
            cur_size = int(n)
            ubLD_array.append(ubLD_1d(eval_seg.m, cur_size))
            generator = gg.graph_generator()
            generator.graph = copy.deepcopy(calc_seg.graph)
            p_in = 0.8 * random.random()
            p_out = 0.1 * random.random()
            incre_seq = generator.generate_seq_random(n=cur_size, inner_prob=p_in, outer_prob=p_out, epoch=1)
            calc_seg.get_incre_data(incre_seq)
            LD = calc_seg.calc_LD1d()
            LD_array.append(LD)
        LD_list.append(LD_array)
        ubLD_list.append(ubLD_array)
        m_list.append(eval_seg.m)

    np.savetxt('./results/moment_statistic_logs/LD_list_1d_exp.txt', np.array(LD_list))
    np.savetxt('./results/moment_statistic_logs/ubLD_list_1d_exp.txt', np.array(ubLD_list))
    np.savetxt('./results/moment_statistic_logs/m_list_1d_exp.txt', np.array(m_list))

def draw_plot_moment_statistic_2d():
    LD_list = np.loadtxt('./results/moment_statistic_logs/LD_list_2d_exp.txt')
    ubLD_list = np.loadtxt('./results/moment_statistic_logs/ubLD_list_2d_exp.txt')
    m_list = np.loadtxt('./results/moment_statistic_logs/m_list_2d_exp.txt')
    print(m_list)
    LD_mean = LD_list.mean(axis=1)
    LD_max = LD_list.max(axis=1)
    LD_min = LD_list.min(axis=1)
    # LD_var = LD_list.var(axis=1)

    ubLD_mean = ubLD_list.mean(axis=1)
    ubLD_max = ubLD_list.max(axis=1)
    ubLD_min = ubLD_list.min(axis=1)
    # ubLD_var = ubLD_list.var(axis=1)

    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    # axis_x = np.linspace(XSTART, XEND, XNUM)
    axis_x = m_list
    ax.plot(axis_x, LD_mean, color = 'blue')
    ax.fill_between(axis_x, LD_min, LD_max, alpha = 0.2, color = 'blue')
    ax.plot(axis_x, ubLD_mean, color = 'red')
    ax.fill_between(axis_x, ubLD_min, ubLD_max, alpha=0.2, color='red')

    fontsize = 13
    ax.set_ylabel('Two-dimensional Local Difference', fontsize=fontsize)
    ax.set_xlabel('Total edge number of the original graph', fontsize=fontsize)
    # ax.grid()
    ax.legend(['Mean Local Difference', 'Max/min Local Difference',
               'Mean upper bound','Max/min  upper bound'], fontsize=fontsize, loc='upper right')
    ax.set_ylim(bottom = 0)
    ax.set_xscale('log')
    # ax.set_xscale('linear')
    # ticks = np.log10(axis_x)
    # ax.set_xticks(ticks, labels=axis_x.astype(int))

    plt.savefig('./results/moment_statistic_logs/moment_plot_figure_2d_exp.pdf', bbox_inches='tight')
    plt.show()
    print(LD_mean[0], LD_mean[-1], (LD_mean[0]-LD_mean[-1])/LD_mean[0])

def save_moment_statistic_2d():
    # load original graph
    with open('./save_random_process/moment_statistic/init_state_moment', 'rb') as file:
        saved_graph = pickle.load(file)
    seg = sg.SEGraph(saved_graph)

    # load the cumulative incremental sequence
    random_seq = np.load('./save_random_process/moment_statistic/random_seq_moment.npy').tolist()

    # count & save
    moment_statistic_pipeline_2d(random_seq, XNUM, seg)

def moment_statistic_pipeline_2d(seq, time_step_num, seg: sg.SEGraph):
    LD_list = []  # 2d-array of shape sample_num * time_step_num
    ubLD_list = []  # 2d-array of shape sample_num * time_step_num
    m_list = []  # (total edge number list) 1d-array of shape time_step_num

    start_log_m = math.log10(XSTART)
    end_log_m = math.log10(XEND)
    log_m_list = np.linspace(start_log_m, end_log_m, XNUM)
    size_list = (np.power(10, log_m_list) - XSTART).astype(np.int)
    print(size_list)

    t = 0
    for size in size_list:
        print('step:', t)
        t += 1
        cumseq = seq[:size]
        print('cumseq length', len(cumseq))

        eval_seg = copy.deepcopy(seg)
        eval_seg.incre_seq_arrival_only_graph(cumseq)

        eval_seg.update_division_Louvain()
        eval_seg.update_struc_data()
        eval_seg.calc_expression_SD()

        mean_n = SAMPLE_MEAN
        std_n = SAMPLE_STD
        sample_num = SAMPLE_NUM
        incre_size_list = np.random.normal(loc=mean_n, scale=std_n, size=sample_num)
        LD_array = []
        ubLD_array = []
        for n in incre_size_list:
            calc_seg = copy.deepcopy(eval_seg)
            cur_size = int(n)
            ubLD_array.append(ubLD(eval_seg.m, cur_size))
            generator = gg.graph_generator()
            generator.graph = copy.deepcopy(calc_seg.graph)
            p_in = 0.8 * random.random()
            p_out = 0.1 * random.random()
            incre_seq = generator.generate_seq_random(n=cur_size, inner_prob=p_in, outer_prob=p_out, epoch=1)
            calc_seg.get_incre_data(incre_seq)
            LD = calc_seg.calc_LD2d()
            LD_array.append(LD)
        LD_list.append(LD_array)
        ubLD_list.append(ubLD_array)
        m_list.append(eval_seg.m)
    np.savetxt('./results/moment_statistic_logs/LD_list_2d_exp.txt', np.array(LD_list))
    np.savetxt('./results/moment_statistic_logs/ubLD_list_2d_exp.txt', np.array(ubLD_list))
    np.savetxt('./results/moment_statistic_logs/m_list_2d_exp.txt', np.array(m_list))

def ubLD(m, n):
    return (n * math.log2(m + n) + 2.5 * n) / (m + n)

def ubLD_1d(m, n):
    return (n * math.log2(m + n) + 1.5 * n) / (m + n)

if __name__ == '__main__':
    # save_moment_statistic_1d()
    draw_plot_moment_statistic_1d()
    # save_moment_statistic_2d()
    draw_plot_moment_statistic_2d()

# def draw_box_moment_statistic():
#     LD_list = np.loadtxt('./results/moment_statistic_logs/LD_list.txt')
#     ubLD_list = np.loadtxt('./results/moment_statistic_logs/ubLD_list.txt')
#     m_list = np.loadtxt('./results/moment_statistic_logs/m_list.txt')
#     width = 0.6
#     print(LD_list.mean(axis=1))
#
#     fig, ax = plt.subplots(figsize=(24, 6), dpi=300)
#     ax.boxplot(x=LD_list.T, widths=width, showmeans=True, meanline=True)
#     ax.boxplot(x=ubLD_list.T, widths=width, showmeans=True, meanline=True)
#     plt.savefig('./results/moment_statistic_logs/moment_box_figure', bbox_inches='tight')
#     plt.show()

# def save_single_statistic():
#     # load original graph
#     with open('./save_random_process/moment_statistic/init_state_moment', 'rb') as file:
#         saved_graph = pickle.load(file)
#     seg = sg.SEGraph(saved_graph)
#
#     # read cumulative incremental sequence
#     # hawkes_seq = np.load('./save_random_process/hawkes_seq_moment.npy').tolist()
#     triad_seq = np.load('./save_random_process/moment_statistic/triad_seq_moment.npy').tolist()
#     # random_seq = np.load('./save_random_process/random_seq_moment.npy').tolist()
#
#     # settings
#     time_step_num = 100
#
#     # save single statistic
#     # hGI, hLD, hub = single_statistic_pipeline(hawkes_seq, time_step_num, seg)
#     tGI, tLD, tub = single_statistic_pipeline(triad_seq, time_step_num, seg)
#     # rGI, rLD, rub = single_statistic_pipeline(random_seq, time_step_num, seg)
#
#     # plt.plot(hLD)
#     # plt.plot(hub)
#
#     plt.plot(tLD)
#     plt.plot(tub)
#
#     # plt.plot(rLD)
#     # plt.plot(rub)
#
#     plt.show()

# def single_statistic_pipeline(seq, time_step_num, seg: sg.SEGraph):
#     GI_list = []
#     LD_list = []
#     ubLD_list = []
#     length = len(seq)
#     step_num = int(length / time_step_num)
#     for t in range(time_step_num - 1):
#         print('step:', t)
#         if (t + 1) * step_num > length:
#             size = length
#         else:
#             size = (t + 1) * step_num
#         cumseq = seq[:size]
#
#         partnum = 150
#         ubLD_list.append(ubLD(size, partnum))
#         partseq = seq[size: size + partnum]
#
#         # 修改得到t时刻的图G
#         eval_seg = copy.deepcopy(seg)
#         eval_seg.incre_seq_arrival_only_graph(cumseq)
#         # 初始化
#         eval_seg.update_division_Louvain()
#         eval_seg.update_struc_data()
#         eval_seg.calc_expression_SD()
#         # 基于G和t时刻的增量序列求出t+1时的GI和LD
#         eval_seg.get_incre_data(partseq)
#         GI2d = eval_seg.fast_GI2d(partnum)
#         LD2d = eval_seg.calc_LD2d()
#         print(LD2d)
#
#         GI_list.append(GI2d)
#         LD_list.append(LD2d)
#
#     return GI_list, LD_list, ubLD_list