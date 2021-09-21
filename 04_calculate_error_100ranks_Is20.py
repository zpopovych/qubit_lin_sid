import os
import time
from datetime import datetime
from rho_lib import *
from boson_data_lib import *
import pandas as pd

is_data = []

prec=.1

os.chdir('/home/zah/PycharmProjects/Kurt2021/2021JUN6')
print('Current directory', os.getcwd())

g_list = ['0.25133', '0.79477', '2.5133', '7.9477', '25.133', '79.477', '251.33']
# f_mod_name = 'Identified_models_all_ranks_20210727_2001_34.hdf5'
f_mod_name = 'Identified_models_all_100_ranks_20210823_1446_18.hdf5'

max_error_df = pd.DataFrame(columns=['State', 'Gamma', 'Rank', 'MaxError'])

# Checking g
for i in range(20):
     f_name = 'Is_' + str(i + 1) + '_data.h5'
     print(f_name)
     for g in g_list:
          print(g)
          rho_kurt, dt = extract_rho(f_name, g)
          times = np.arange(0., dt * rho_kurt.shape[0], dt)
          for rank in range(3, 100):
               print(rank)
               model = load_model_g_rank(f_mod_name, g, rank)
               rho_sid = evolve_rho(rho_kurt[0], model, times, exact_init_states=True)
               n = min([rho_kurt.shape[0], rho_sid.shape[0]])
               err = np.abs(rho_kurt[:n] - rho_sid[:n])
               max_err = np.max(err)
               data_row = [{'Gamma': g, 'State': i, 'Rank': rank, 'MaxError': max_err}]
               max_error_df = max_error_df.append(data_row, ignore_index=True)

print(max_error_df)

print(max_error_df.groupby(['Gamma']).max(numeric_only=True))

max_error_df.to_pickle("models_100_rank_prec.pkl")
