import se_graph as sg
import graph_generator as gg
import numpy as np
import copy
import networkx as nx
import matplotlib.pyplot as plt
import pickle

# Settings
# init state
init_num_list = [400, 600, 300, 200, 500] # node_num == 2000
init_pin = 0.3
init_pout = 0.01
init_seed = 12
init_comments = f'init_num_list = {init_num_list}, init_pin = {init_pin}, init_pout = {init_pout}, init_seed = {init_seed}\n'

# processes
incre_size = 150187

# hawkes
hawkes_size = incre_size
hawkes_p = 0.95 # edge prob
hawkes_sample_num = 10

# triad
triad_size = incre_size
triad_p = 0.95 # edge prob

# random
random_size = incre_size
random_p = 0.9 # inner edge prob
random_q = 0.05 # outer edge prob

# ---------------Time & Error Evaluation-------------------#

# # save_random_process init state
# generator = gg.graph_generator()
# generator.make_random_partition_graph(init_num_list, init_pin, init_pout, seed=init_seed)
# with open('./save_random_process/init_state', 'wb') as file:
#     pickle.dump(generator.graph, file)
# print('Initial State Generation Finished')
#
# # save_random_process hawkes sequence (cumulative sequence at T)
# hawkes_generator = copy.deepcopy(generator)
# hawkes_seq = hawkes_generator.generate_seq_hawkes(hawkes_size, hawkes_p, hawkes_sample_num)
# np.save('./save_random_process/hawkes_seq.npy', np.array(hawkes_seq))
# # load: np.load('demo.npy').tolist()
# print('Hawkes Sequence Generation Finished')
#
# # save_random_process triad clusure sequence
# with open('./save_random_process/init_state', 'rb') as file:
#     init_graph = pickle.load(file)
# generator = gg.graph_generator(init_graph)
# triad_generator = copy.deepcopy(generator)
# triad_seq = triad_generator.generate_seq_triad(triad_size, triad_p)
# print(len(triad_seq))
# np.save('./save_random_process/triad_seq.npy', np.array(triad_seq))
# print('Triad Closure Sequence Generation Finished')
#
# # save_random_process PRG sequence
# random_generator = copy.deepcopy(generator)
# random_seq = random_generator.generate_seq_random(random_size, random_p, random_q)
# np.save('./save_random_process/random_seq_new.npy', np.array(random_seq))
# print('Random (PRG) Sequence Generation Finished')

# --------------- Moment Statistic -------------------#

# save_random_process init state
# generator = gg.graph_generator()
# generator.make_random_partition_graph([30,20,20,30,20], init_pin, init_pout, seed=init_seed)
# print('origin node:',len(generator.graph.nodes))
# print('origin edge:',len(generator.graph.edges))
# with open('./save_random_process/moment_statistic/init_state_moment', 'wb') as file:
#     pickle.dump(generator.graph, file)
# print('Initial State Generation Finished')
# print(len(generator.graph.nodes))
# print(len(generator.graph.edges))

# save_random_process PRG sequence
# random_generator = copy.deepcopy(generator)
# random_seq = random_generator.generate_seq_random(23520, 0.90, 0.05) # edges*(step-1) 480*49
# np.save('./save_random_process/moment_statistic/random_seq_moment.npy', np.array(random_seq))
# print(len(random_seq))
# print('Random (PRG) Sequence Generation Finished')
