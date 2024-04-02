#!/usr/bin/env python
import yaml
with open("config.yaml", "r") as stream:
    try:
        CONFIG=yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

import sys
import os.path
import numpy as np
from cynet import cynet as cn
import cynet_utils.spatial as sp
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import geopandas as gpd
import contextily as ctx
from tqdm import tqdm
import geopy.distance
import glob
import hashlib

SAVED_PARAM='./generate_sim.py -Z 0.17 -r 0.009 -m 0.005 -d 0.001 -f True -N 1'

import argparse
from argparse import RawTextHelpFormatter
banner='GENERATE fractalnet performance'
zed='copyright 2020 zed.uchicago.edu'
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='########\n'
                                 +banner+'\n'
                                 +zed,formatter_class=RawTextHelpFormatter)
parser._optionals.title="Program Options\n"+SAVED_PARAM

parser.add_argument('-Z', metavar="",dest='zx',
                    action="store", type=float,
                    default=.06,
                    help="z threshold")
parser.add_argument('-r', metavar="",dest='rx',
                    action="store", type=float,
                    default=8.0,
                    help="radius")
parser.add_argument('-m', metavar="",dest='mx',
                    action="store", type=float,
                    default=15,
                    help="miles")
parser.add_argument('-d', metavar="",dest='dx',
                    action="store", type=float,
                    default=1.2,
                    help="detail")
parser.add_argument('-f', metavar="",dest='FIGSHOW',
                    action="store", type=str2bool,
                    default=False,
                    help="draw figures")
parser.add_argument('-N', metavar="",dest='DAYNUM',
                    action="store", type=int,
                    default=None,
                    help="generate only N days")

parser.add_argument('-R', metavar="",dest='PRECISION',
                    action="store", type=int,
                    default=5,
                    help="coordinate precision")



VERBOSE=False
args=parser.parse_args()
rx=args.rx
dx=args.dx
mx=args.mx
zx=args.zx
FIGSHOW=args.FIGSHOW
DAYNUM=args.DAYNUM
PRECISION=args.PRECISION

DESC='P#z'+str(zx)+'#d'+str(dx)+'#r'+str(rx)+'#m'+str(mx)
FUTURE = CONFIG['FUTURE']

FILEPATH = CONFIG['MODELDIR'] 
PREDPATH = CONFIG['PREDICTIONDIR']
OUTPUTDIR = CONFIG['OUTPUTDIR']
MOVIEDIR= CONFIG['MOVIESDIR']
os.makedirs(PREDPATH, exist_ok=True)
os.makedirs(MOVIEDIR, exist_ok=True)
os.makedirs(OUTPUTDIR, exist_ok=True)

init_date, end_date, oos_end, freq = CONFIG['train_init_date'],\
    CONFIG['train_end_date'],\
    CONFIG['oos_end'], CONFIG['freq']

from datetime import datetime, timedelta
start_of_test = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)
start_of_test = start_of_test.date()
end_of_sim = datetime.strptime(oos_end, '%Y-%m-%d') + timedelta(days=FUTURE+1)
end_of_sim = end_of_sim.date()



datetime_range = pd.date_range(start=init_date, end=end_of_sim, freq=freq) # this is the oos period
log_path = FILEPATH
model_nums = CONFIG['model_nums']
horizon = CONFIG['horizons'][0]

day_min = np.where(datetime_range==str(start_of_test))[0][0]  # num of rows in triplet/XX.columns
day_max= len(datetime_range)   # no. of columns in any file in split/XX


events=[]
var_title_map = {}
event_dict=CONFIG['VAR_LIST']
for i in np.arange(len(event_dict)):
    event_name = list(event_dict[i].keys())[0]
    types=event_dict[i][event_name]['types']
    hashed_types = hashlib.sha1(bytearray(cn.stringify(types).encode('utf-8'))).hexdigest()
    events.append(hashed_types)
    var_title_map[events[i]]= event_name

    
# =========================== DF column names ==========================START
day_col = 'day'
actual_event_col = 'actual_event'
variable_col = 'target'
source_col = 'source'
predictin_col = 'predictions'
lon_col = 'lon2'
lat_col = 'lat2'
source = None
grace = 1

print('reading simfile: {}/{}'.format(log_path,CONFIG['SIMPATH']))

# =========================== DF column names ==========================END
df = pd.read_csv('{}/{}'.format(log_path,CONFIG['SIMPATH']))

lat_min, lat_max = df.lat1.min(), df.lat2.max()
lon_min, lon_max = df.lon1.min(), df.lon2.max()

df = df[ (df['day'] >= day_min) & (df['target'] != 'VAR') ]


parameters = {e: {} for e in events}

# =========================== FUNCTIONS ==========================


def df_intersect(df1, df2, columns=[]):
    df1__ = df1[columns]
    df2__ = df2[columns]

    df1__m = df1__.apply(lambda x: hash(tuple(x)), axis=1)
    df2__m = df2__.apply(lambda x: hash(tuple(x)), axis=1)

    df_ = df1[df1__m.isin(df2__m)]

    return df_
 

def df_setdiff(df1, df2, columns=[]):
    df1__ = df1[columns]
    df2__ = df2[columns]

    df1__m = df1__.apply(lambda x: hash(tuple(x)), axis=1)
    df2__m = df2__.apply(lambda x: hash(tuple(x)), axis=1)

    df_ = df1[~df1__m.isin(df2__m)]

    return df_


def df_union(df_1, df_2, columns=[], count_only=False):

    dfh_1 = df_1[columns].apply(lambda x: hash(tuple(x)), axis=1)
    dfh_2 = df_2[columns].apply(lambda x: hash(tuple(x)), axis=1)
    
    diff = df_1[~dfh_1.isin(dfh_2)]
    union = pd.concat([diff, df_2], axis=0, sort=False)
    if count_only:
        return len(union)
    else:
        return union
    
def transCMAP(cmap=plt.cm.RdBu,linear=True):
    cmap1 = cmap(np.arange(cmap.N))
    if linear:
        cmap1[:,-1] = np.linspace(0, 1, cmap.N)
    else:
        cmap1[:,-1] = np.logspace(0, 1, cmap.N)
    return ListedColormap(cmap1)


def getHausdorf(coord,pt):
    return np.min([geopy.distance.distance(pt,i).miles for i in coord])

def getHausdorfarg(coord,pt,MAX_DISTANCE=1):
    C=coord[np.argmin([geopy.distance.distance(pt,i).miles for i in coord])]
    if geopy.distance.distance(pt,C).miles < MAX_DISTANCE:
        return C
    return np.array([np.nan,np.nan])

def getHausdorf_df(df, pt, EPS=0.0001):
    if len(df) == 0:
        return  np.array([np.nan,np.nan]), []
    
    while True:
        T = [tuple(i) for i in df[(np.abs(df.lat-pt[0])<EPS) 
              & (np.abs(df.lon-pt[1])<EPS)].values]
        if len(T)>0:
            break
        else:
            EPS=2*EPS
    return getHausdorfarg(T,tuple(pt)),T
 

def get_intensity(intensity,lon_mesh,lat_mesh,pt_,sigma=3,radius=2):
    '''
    single point spread calculation with Gaussian diffusion
    '''
    lon_del=lon_mesh[0,:]
    lat_del=lat_mesh[:,0]
    lon_index=np.arange(len(lon_del))[(pt_[1]-lon_del<radius)*(pt_[1]-lon_del>-radius)]
    lat_index=np.arange(len(lat_del))[(pt_[0]-lat_del<radius)*(pt_[0]-lat_del>-radius)]
    mu=np.mean(lon_index)
    bins=lon_index
    intensity_lon=1/(sigma*np.sqrt(2*np.pi))*np.exp(-(bins - mu)**2/(2 * sigma**2))

    mu=np.mean(lat_index)
    bins=lat_index
    intensity_lat=1/(sigma*np.sqrt(2*np.pi))*np.exp(-(bins - mu)**2/(2 * sigma**2))
    for i in np.arange(len(lon_index)):
        for j in np.arange(len(lat_index)):
            intensity[lat_index[j],lon_index[i]]=intensity[lat_index[j],
                                lon_index[i]]+intensity_lon[i]*intensity_lat[j]
    return intensity


def get_mesh(df0,lat_min,lat_max,lon_min,lon_max,radius=2,detail=0.25):
    coord_=df0[[lat_col,lon_col]].values
    lon_grid=np.arange(lon_min-radius,lon_max+radius,detail)
    lat_grid=np.arange(lat_min-radius,lat_max+radius,detail)
    lon_mesh,lat_mesh=np.meshgrid(lon_grid,lat_grid)
    return lon_mesh,lat_mesh,coord_

def get_prediction(
        df,
        days,
        types,
        lat_min,
        lat_max,
        lon_min,
        lon_max,
        sigma=3.5, #=======YI made sigma a parameter
        radius=0.01,
        detail=0.2,
        Z=1.0,
        miles=50, #=======YI made miles in spatial relaxation a paramter
        RETURN_PRED=False
    ):

    # ============= DF column names ==========================START
    day_col = 'day'
    actual_event_col = 'actual_event'
    variable_col = 'target'
    source_col = 'source'
    predictin_col = 'predictions'
    lon_col = 'lon2'
    lat_col = 'lat2'
    source = None
    grace = 1
    # ================ DF column names ==========================END

    df = df[df[day_col].between(days - grace,days + grace)]
    df = df[df[variable_col].isin(types)]
    df_prd0 = df[(df[day_col]==days) & (df[predictin_col]==1)]

    # =============== Find all tiles ============================ #
    triplet=CONFIG['triplet_dir']
    def getCoord(x):
        return np.array(x.split('#')[:4]).astype(float)

    File=glob.glob(triplet+'/*coords')
    dtiles=pd.DataFrame([getCoord(x[0]) for x in pd.read_csv(File[0],header=None).values],
                        columns=['lat1','lat2','lon1','lon2'])    
    # =============== Find all tiles ============================ #
    # SPATIAL ADJUSTMENT
    lon_grid = np.arange(lon_min - radius, lon_max + radius, detail)
    lat_grid = np.arange(lat_min - radius, lat_max + radius, detail)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid,lat_grid)
    
    lon_mesh0, lat_mesh0, coord_= get_mesh(
        df_prd0,
        lat_min,
        lat_max,
        lon_min,
        lon_max,
        radius=radius,
        detail=detail)
    
    intensity = np.zeros(lat_mesh0.shape)
    for i in coord_:
        intensity = get_intensity(
            intensity,
            lon_mesh0,
            lat_mesh0,
            i,
            sigma=sigma,
            radius=radius)
        
    intensity0 = np.multiply(intensity, (intensity > Z))
    intensity0=(1. / intensity0.max()) * intensity0
    
    lon_del=lon_mesh0[0,:]
    lat_del=lat_mesh0[:,0]
    A=(intensity0>Z).nonzero()
    coordNZ=[(lat_del[A[0][i]],lon_del[A[1][i]]) for i in np.arange(len(A[0]))]
    df_cnz=pd.DataFrame(coordNZ,columns=['lat','lon'])

    xgpp = np.array([getHausdorf_df(df_cnz,
                                    tuple(i),EPS=0.001)[0] 
                     for i in (dtiles[[lat_col,
                                       lon_col]].drop_duplicates().values)])
    
    if VERBOSE:
        print(xgpp)
    xgpp = pd.DataFrame(xgpp, columns=['lat','lon']).round(PRECISION).dropna().drop_duplicates()

    if RETURN_PRED:
        return lon_mesh0, lat_mesh0, intensity, intensity0,df_cnz, xgpp, df_prd0
    
    return lon_mesh0, lat_mesh0, intensity, intensity0,df_cnz,xgpp


def getFig(parameters,xlim = [-.838e7, -.835e7],
           ylim =[.485e7, .4878e7],
           AXCOL='#778833',
           var_title_map=var_title_map):

    base_cmaps = [plt.cm.Reds, plt.cm.Blues, plt.cm.Greens]
    
    # Initialize parameters dictionary if it's not populated
    for i, (var, title) in enumerate(var_title_map.items()):
        cmap = base_cmaps[i % len(base_cmaps)]  # Cycle through base_cmaps if there are more events than cmaps
        truncated_cmap = sp.truncate_colormap(cmap, 0.01, .9)
        parameters[var].update({
            'cmap': truncated_cmap,
            'bcolor': truncated_cmap(0.9),  # Use a color from the colormap for border color
            'mcolor': truncated_cmap(0.9),  # Same color for marker color
            'mtype': 'o'  # Default marker type, you can customize this if needed
        })

    for var in var_title_map:
        if var not in parameters or 'cmap' not in parameters[var]:
            raise KeyError(f"'cmap' for {var} not found in parameters")

    width_height_ratio = (xlim[1] - xlim[0]) / (ylim[1] - ylim[0])
    height = 10
    width = width_height_ratio * height
    fig = plt.figure(figsize=(width, height))
    ax = fig.gca()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    for var, props in parameters.items():
        lon_, lat_ = props['lon'], props['lat']
        int_ = props['int']

        Longitude, Latitude, density = [], [], []
        for row in range(len(int_)):
            for col in range(len(int_[row])):
                if int_[row, col] > 0:
                    Longitude.append(lon_[row, col])
                    Latitude.append(lat_[row, col])
                    density.append(int_[row, col])

        df_density = pd.DataFrame({
            'Longitude': Longitude,
            'Latitude': Latitude,
            'density': density
        })

        wdf = gpd.GeoDataFrame(
            df_density, 
            geometry=gpd.points_from_xy(df_density.Longitude, df_density.Latitude)
        )
        wdf.crs = {'init': 'epsg:4326'}
        wdf = wdf.to_crs(epsg=3857)

        ax = wdf.plot(
            ax=ax,
            column='density',
            edgecolor='k',
            linewidth=0,
            cmap=props['cmap'],
            alpha=.8, 
            zorder=5, 
            marker='s',
            markersize=85)

        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Voyager, alpha=1)

    ax.text(
        0.825, 0.965,
        str(time).split()[0], 
        transform=ax.transAxes,
        fontweight='light',
        fontsize=18,
        color=AXCOL,
        alpha=1,
        verticalalignment='bottom', 
        horizontalalignment='right'
    )

    sp.saveFIG(f'{MOVIEDIR}/snapshot_{str(time).split()[0]}.png') 


def getPred(day=1000,Z=0.02,radius=.007,sigma=2.75,detail=0.0007,miles=.01):
    '''
    returns time stamp, sum_ and parameters. 
    In the future-pred version of generate_sym, simm_ is set to None
    The parameters record <lon>, <lat> grid, and intensity0.
    Note <intensity0> is normalized intensity after thresholding by Z
    '''

    time = datetime_range[day].date()
    sum_={}
    for var in events:
        types = [var]
        lon_, lat_, int_plot, int_,df_cnz,xgpp = get_prediction(
            df,
            day,
            types,
            lat_min,
            lat_max,
            lon_min,
            lon_max,
            radius=radius,
            sigma=sigma,
            detail=detail,
            miles=miles,
            Z=Z)

        #print('day={}, {}: fp={}, tp={}, fn={}; ppv={:.3f}, sens={:.3f}'
        #      .format(day, var, fp, tp, fn, ppv, sens))
        
        sum_[var]=None
        parameters[var].update({
            'lon': lon_, 'lat': lat_,
            'int': int_, 'df_cnz': df_cnz, 'predictions': xgpp
        })
    return time,sum_,parameters#,time,events

#======================================= END FUNCTIONS ======================================
if DAYNUM is not None:
    day_max=day_min+DAYNUM

for day in tqdm(np.arange(day_min,day_max)):
    time,sum_,parameters=getPred(day=day,Z=zx,radius=rx,miles=mx,detail=dx)
    if FIGSHOW:
        getFig(parameters)

    PRED=pd.concat([parameters[event]['predictions'][['lat','lon']]
               .drop_duplicates().assign(typ=var_title_map[event]) 
               for event in events]).drop_duplicates()

    PRED.to_csv(PREDPATH+'prediction_'+str(time)+'.csv',index=None)




