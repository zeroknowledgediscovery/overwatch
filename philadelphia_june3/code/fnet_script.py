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
    
LOGFILE = CONFIG['EVENTLOG']
STOREFILE = os.path.join('./', os.path.basename(LOGFILE).split('.')[0] + '.p')

# Column names in the event log file for coordinate 1 and 2
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
for i in np.arange(len(event_dict)):
    event_name = list(event_dict[i].keys())[0]
    if 'value_limits' in event_dict[i][event_name].keys():
        value_limits = event_dict[i][event_name]['value_limits']
        S0 = cn.spatioTemporal(
            # File
            log_file=LOGFILE,
            log_store=STOREFILE,
            # Spatial
            coord1=coord1,
            coord2=coord2,
            grid=tiles,
            # Temporal
            #year=year,
            #month=month,
            #day=day,
            DATE=DATE_STR,
            init_date=init_date,
            end_date=oos_end,
            freq=freq,
            # Event
            EVENT=event_dict[i][event_name]['col_name'],
            value_limits=event_dict[i][event_name]['value_limits'],
            threshold=event_dict[i][event_name]['threshold']
        )             
    else:
        types=[event_dict[i][event_name]['types']]
        S0 = cn.spatioTemporal(
            # File
            log_file=LOGFILE,
            log_store=STOREFILE,
            # Spatial
            coord1=coord1,
            coord2=coord2,
            grid=tiles,
            # Temporal
            #year=year,
            #month=month,
            #day=day,
            DATE=DATE_STR,
            init_date=init_date,
            end_date=oos_end,
            freq=freq,
            # Event
            EVENT=event_dict[i][event_name]['col_name'],
            types=types,
            threshold=event_dict[i][event_name]['threshold']
        )             
    S0.fit(csvPREF=event_dict[i][event_name]['csvname_prefix'])
    if i == 0:
        tiles = S0.getGrid()


CSVfiles = [x[list(x.keys())[0]]['csvname_prefix']+'.csv'
            for x in CONFIG['VAR_LIST']]


# Make sure the triplet folder and split folder exist
triplet_dir, split_dir = CONFIG['triplet_dir'], CONFIG['split_dir']
if not os.path.exists(triplet_dir):
    os.makedirs(triplet_dir)
if not os.path.exists(split_dir):
    os.makedirs(split_dir)


# Triplet
triplet_fnames_prefix = triplet_dir+'/TRP_' + init_date + '_' + end_date
cn.readTS(
    CSVfiles, 
    csvNAME=triplet_fnames_prefix, 
    BEG=init_date, 
    END=end_date)

# Split
split_dirname = split_dir+'/'
split_prefix = init_date + '_' + oos_end + '_'
cn.splitTS(
    CSVfiles, 
    BEG=init_date, 
    END=oos_end, 
    dirname=split_dirname, 
    prefix=split_prefix)

# File parameters
TS_PATH = triplet_fnames_prefix + '.csv' # The time series (data only)
NAME_PATH = triplet_fnames_prefix + '.coords' # The names for each time series
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
if not os.path.exists(FILEPATH):
    os.makedirs(FILEPATH)
cores = int(CONFIG['NUMPROC'])

print(cores,BEG,END,NUM_RERUNS,PARTITION)

XG = cn.xgModels( 
    TS_PATH,
    NAME_PATH, 
    LOG_PATH,
    FILEPATH, 
    BEG, 
    END, 
    NUM_RERUNS, 
    PARTITION,
    XgenESeSS,
    RUN_LOCAL)
XG.run(workers=cores)

