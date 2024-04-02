#!/usr/bin/python
import yaml
with open("config.yaml", "r") as stream:
    try:
        CONFIG=yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

import os.path
from cynet import cynet as cn
import cynet_utils.spatial as sp
import pandas as pd
import numpy as np
import os
from glob import glob


FUTURE = CONFIG['FUTURE']
LOGFILE = CONFIG['EVENTLOG']
STOREFILE = os.path.join('./', os.path.basename(LOGFILE).split('.')[0] + '.p')
coord1, coord2 =CONFIG['coord1'], CONFIG['coord2']

# Tiles
lat_min, lat_max = CONFIG['LATITUDE_LIM']
lon_min, lon_max = CONFIG['LONGITUDE_LIM']
lat_eps, lon_eps = CONFIG['LATITUDE_EPS'], CONFIG['LONGITUDE_EPS']

lat = np.around(np.linspace(lat_min, lat_max, lat_eps + 1), decimals=5)
lon = np.around(np.linspace(lon_min, lon_max, lon_eps + 1), decimals=5)
tiles = [[lat[i], lat[i + 1], lon[j], lon[j + 1]]
         for i in np.arange(lat_eps) for j in np.arange(lon_eps)]
DATE_STR=CONFIG['DATE']

init_date, end_date, oos_end, freq = CONFIG['train_init_date'],\
    CONFIG['train_end_date'],\
    CONFIG['oos_end'], CONFIG['freq']

event_dict=CONFIG['VAR_LIST']
CSVfiles =[x[list(x.keys())[0]]['csvname_prefix']+'.csv'
           for x in CONFIG['VAR_LIST']]

split_dir = CONFIG['split_dir']
split_dirname = split_dir+'/'
split_prefix = init_date + '_' + oos_end + '_'

# File parameters
FILEPATH = CONFIG['MODELDIR'] # Make sure to create a folder with name `FILEPATH` below
LOG_PATH = 'log.txt'

# XgenESSeS parameters
BEG = int(CONFIG['BEG'])  # minimum delay considered
END = int(CONFIG['END']) # maximum delay considered
NUM_RERUNS = int(CONFIG['NUM']) # number of reruns
PARTITION = CONFIG['PARTITION'] # partitioning points. 
XgenESeSS = CONFIG['XgenESeSS']
RUN_LOCAL = CONFIG['RUN_LOCAL']

# make sure a folder named `models` is created
cores = int(CONFIG['NUMPROC'])

# File parameters
MODEL_GLOB = FILEPATH+'/*model.json'
RESPATH = FILEPATH+'/*model*res'
DATA_PATH = os.path.join(split_dirname, split_prefix)
# the split files path prefix

RUNLEN = len(pd.date_range(start=init_date, end=oos_end, freq=freq))+FUTURE
# Now we get the start of the test period.
# Since the temporal resolution is 1 day,
# we just need to find the tomorrow of the training end.
from datetime import datetime, timedelta
start_of_test = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)
start_of_test = start_of_test.date()

#start_of_test='2022-06-01'
FLEX_TAIL_LEN = len(pd.date_range(start=start_of_test, end=oos_end, freq=freq))


model_nums = CONFIG['model_nums']
horizon = CONFIG['horizons'][0]
VARNAME = list(set([fname.split('#')[-1] for fname in glob(DATA_PATH + "*")])) + ['ALL']

# Running parameters
# Make sure you have multi-core access when using cores greater than 1. 
print(f'run length (train + test) = {RUNLEN}\ntest length = {FLEX_TAIL_LEN}')

cn.run_pipeline(
    MODEL_GLOB,
    model_nums, 
    horizon, 
    DATA_PATH, 
    RUNLEN, 
    VARNAME, 
    RESPATH, 
    FLEX_TAIL_LEN=FLEX_TAIL_LEN,
    cores=1,
    gamma=True)
res = pd.read_csv(CONFIG['RESPATH'])
res[ (res['varsrc'] == 'ALL') & (res['auc'] < .999)]['auc'].describe()

log_path = FILEPATH
cn.flexroc_only_parallel(
    '{}/*.log'.format(log_path),
    tpr_threshold=CONFIG['TPR_TH'],
    fpr_threshold=None,
    FLEX_TAIL_LEN=FLEX_TAIL_LEN, 
    cores=cores)

mapper = cn.mapped_events('{}/*{}models#*#*.csv'.format(log_path, model_nums[0]))
mapper.concat_dataframes('{}/{}'.format(log_path,CONFIG['SIMPATH']))


end_of_sim = datetime.strptime(oos_end, '%Y-%m-%d') + timedelta(days=FUTURE+1)
end_of_sim = end_of_sim.date()
datetime_range = pd.date_range(start=init_date, end=end_of_sim, freq=freq) # this is the oos period

simf=pd.read_csv('{}/{}'.format(log_path,CONFIG['SIMPATH']))
simf['date']=[datetime_range[x] for x in simf["day"].values]
simf.to_csv('{}/{}'.format(log_path,CONFIG['SIMPATH']),index=None)


 
