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
import hashlib

VERBOSE=False 

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

parser = argparse.ArgumentParser(description='########\n'+banner+'\n'+zed,formatter_class=RawTextHelpFormatter)
parser._optionals.title="Program Options"

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


 
args=parser.parse_args()
rx=args.rx
dx=args.dx
mx=args.mx
zx=args.zx
FIGSHOW=args.FIGSHOW
DAYNUM=args.DAYNUM

DESC='P#z'+str(zx)+'#d'+str(dx)+'#r'+str(rx)+'#m'+str(mx)

FILEPATH = CONFIG['MODELDIR'] # Make sure to create a folder with name `FILEPATH` below
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

#start_of_test = '2023-01-01'

end_of_sim = oos_end
datetime_range = pd.date_range(start=init_date, end=end_of_sim, freq=freq) # this is the oos period

log_path = FILEPATH
model_nums = CONFIG['model_nums']
horizon = CONFIG['horizons'][0]
#print(start_of_test)

day_min = np.where(datetime_range==str(start_of_test))[0][0] 
day_max= len(datetime_range)

events=[] 
event_dict=CONFIG['VAR_LIST']
for i in np.arange(len(event_dict)):
    event_name = list(event_dict[i].keys())[0]
    types=event_dict[i][event_name]['types']
    #hashed_types = hashlib.sha1(bytearray('-'.join(types).encode('utf-8'))).hexdigest()
    hashed_types = hashlib.sha1(bytearray(cn.stringify(types).encode('utf-8'))).hexdigest()
    events.append(hashed_types)

    #events=events+[('-'.join(types)).replace(' ','_')]

print(events)
#events = ['Kidnapping-ArmedAssault-Assassination-Hostage-Hijacking',
#          'Bombing-InfrastructureAttack'
#]
var_title_map = {
    events[0]: 'PROP',
    events[1]: 'VIOL',
    events[2]: 'NARCO'
}

######---> get the above from config.yaml ### UPDATE TODO

#print(events)
#quit()

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


def getHausdorf_df(df, pt, EPS=0.0001):
    if len(df) == 0:
        return np.inf, []
    
    while True:
        T = [tuple(i) for i in df[(np.abs(df.lat-pt[0])<EPS) 
              & (np.abs(df.lon-pt[1])<EPS)].values]
        if len(T)>0:
            break
        else:
            EPS=2*EPS
    return getHausdorf(T,tuple(pt)),T


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


    print(df.head())

    df = df[df[day_col].between(days - grace,days + grace)]

    if VERBOSE:
        print(df)
        df.to_csv('tmp.csv')
    
    print('xxxxx\n',df[variable_col].value_counts())
    df = df[df[variable_col].isin(types)]

    print('ddddd\n',df.head(),variable_col,types)

    
    df_gnd = df[(df[day_col]==days) & (df[actual_event_col]==1)]
    df_prd0 = df[(df[day_col]==days) & (df[predictin_col]==1)]
    df_prd1 = df[(df[day_col]==days - grace) & (df[predictin_col]==1)]
    df_prd2 = df[(df[day_col]==days + grace) & (df[predictin_col]==1)]
    
    df_prd0_tp = df_prd0[df_prd0[actual_event_col]==1]

    if VERBOSE:
        print(df_prd0)

    # UPDXX calculate tp
    df_gndB = df[(df[day_col]==days-grace) & (df[actual_event_col]==1)]
    df_gndF = df[(df[day_col]==days+grace) & (df[actual_event_col]==1)]    
    df_tpB = df_intersect(df_prd0,df_gndB, columns=[lat_col, lon_col])
    df_tpF = df_intersect(df_prd0,df_gndF, columns=[lat_col, lon_col])
    df_tp = df_union(
        df_union(df_prd0_tp, df_tpB, columns=[lat_col, lon_col]),
        df_tpF,
        columns=[lat_col, lon_col])
    tp = df_tp.index.size
    
    df_fp = df_setdiff(df_prd0,df_tp,columns=[lat_col, lon_col])
    fp = df_fp.index.size
    
    df_fn0 = df[(df[day_col]==days) & (df[actual_event_col]==1) & (df[predictin_col]==0)]
    df_fn1 = df[(df[day_col]==days - grace)  & (df[predictin_col]==0)]
    df_fn2 = df[(df[day_col]==days + grace)  & (df[predictin_col]==0)]
    df_fn = df_intersect(df_intersect(df_fn0,df_fn1,columns=[lat_col, lon_col]),
                      df_fn2,columns=[lat_col, lon_col])
    fn= df_fn.index.size

    if VERBOSE:
        print('tmporal comp: --> ', 'tp ',tp, ' fp ', fp, ' fn ',fn)
        
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

    xgfp = np.array([getHausdorf_df(df_cnz,tuple(i),EPS=0.01)[0] for i in (df_fp[[lat_col,lon_col]].drop_duplicates().values)])
    fp = np.sum(xgfp < miles)
    
    xgfn = np.array([getHausdorf_df(df_cnz, tuple(i), EPS=0.01)[0] for i in (df_fn[[lat_col,lon_col]].drop_duplicates().values)])
    fn = np.sum(xgfn > 1.5 * miles)
    
    df_tp_0 = df_intersect(df_tp, df_prd0,columns=[lat_col, lon_col])

    if RETURN_PRED:
        return fn, tp, fp, tp/(tp+fp), tp/(tp+fn), lon_mesh0, lat_mesh0, intensity, intensity0, df_gnd, df_fn,df_tp,df_fp,df_tp_0,df_prd0
    
    return fn, tp, fp, tp/(tp+fp), tp/(tp+fn), lon_mesh0, lat_mesh0, intensity, intensity0, df_gnd, df_fn,df_tp,df_fp,df_tp_0


#     xlim = [-.8376e7, -.835e7]
#     ylim =[.485e7, .488e7]

def getFig(time_,xlim = [-.8372e7, -.836e7],
           ylim =[.486e7, .4875e7],AXCOL='#778833',var_title_map=var_title_map
          ):
    cmap0 = sp.truncate_colormap(plt.cm.get_cmap('Reds'), 0.01, .9)
    cmap1 = sp.truncate_colormap(plt.cm.get_cmap('Blues'), 0.01, .9)
    cmap2 = sp.truncate_colormap(plt.cm.get_cmap('Greens'), 0.01, .9)
    
    bcolor0 = 'brown'
    bcolor1 = 'blue'
    bcolor2 = 'green'

    parameters[events[0]]['cmap'] = cmap0
    parameters[events[0]]['bcolor'] = bcolor0
    parameters[events[0]]['mcolor'] = bcolor0
    parameters[events[0]]['mtype'] = 'v'

    parameters[events[1]]['cmap'] = cmap1
    parameters[events[1]]['bcolor'] = bcolor1
    parameters[events[1]]['mcolor'] = bcolor1
    parameters[events[1]]['mtype'] = '^'

    if len(events)>2:
        parameters[events[2]]['cmap'] = cmap2
        parameters[events[2]]['bcolor'] = bcolor2
        parameters[events[2]]['mcolor'] = bcolor2
        parameters[events[2]]['mtype'] = '>'
     
    barTitleSize, barTickerSize = 20, 18
    barWidth, barHeight = 1.4, 1.76
    markerSize = 60
    lighten_ratio = .2

    width_height_ratio = (xlim[1] - xlim[0]) / (ylim[1] - ylim[0])
    height = 10
    width = width_height_ratio * height
    fig = plt.figure(figsize=(width, height))
    # A MUST-HAVE OR OTHERWISE THE DIMENSION OF THE PLOT ARE NOT UNIFORM
    ax = fig.gca()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    for var in events:

        fp, tp, fn = parameters[var]['fp'], parameters[var]['tp'], parameters[var]['fn']
        lon_, lat_ = parameters[var]['lon'], parameters[var]['lat']
        int_, dfG = parameters[var]['int'], parameters[var]['dftp']

        dfG['Latitude'] = (dfG.lat1 + dfG.lat2) / 2
        dfG['Longitude'] = (dfG.lon1 + dfG.lon2) / 2
        dfG_ = dfG[['Latitude','Longitude']]
        GND = gpd.GeoDataFrame(
            dfG_, 
            geometry=gpd.points_from_xy(dfG_.Longitude, dfG_.Latitude))
        GND.crs= {'init' :'epsg:4326'}
        GND = GND.to_crs(epsg=3857)

        Longitude = []
        Latitude = []
        density = []
        for row in np.arange(0,len(int_)):
            for col in np.arange(0,len(int_[row])):
                if int_[row, col] > 0:
                    Longitude = np.append(Longitude,lon_[row,col])
                    Latitude = np.append(Latitude,lat_[row,col])
                    density = np.append(density,int_[row,col])

        df_density=pd.DataFrame(data={
            'Longitude': Longitude, 
            'Latitude': Latitude, 
            'density':density})

        wdf = gpd.GeoDataFrame(
            df_density, 
            geometry=gpd.points_from_xy(df_density.Longitude, df_density.Latitude))
        wdf.crs= {'init' :'epsg:4326'}
        wdf = wdf.to_crs(epsg=3857)

        ax = fig.gca()
        ax = wdf.plot(
            ax=ax,
            column='density',
            edgecolor='none',
            linewidth=0,
            cmap=parameters[var]['cmap'],
            alpha=.8, 
            zorder=5, 
            marker='s',
            markersize=40)

        ax=GND.plot(
            ax=ax, 
            marker=parameters[var]['mtype'],
            lw=1,
            edgecolor='w',
            facecolors=parameters[var]['mcolor'],
            #edgecolor=parameters[var]['mcolor'],
            # color=lighten_color(parameters[var]['mcolor'], lighten_ratio), 
            markersize=80, 
            alpha=.3,
            zorder=10)

        #ctx.add_basemap(ax, source=ctx.providers.Stamen.Toner,alpha=.5)  ###IXC
        #ctx.add_basemap(ax, source=ctx.providers.Stamen.TonerLite,alpha=1)  ###IXC
        #ctx.add_basemap(ax, source=ctx.providers.Stamen.TonerLite,alpha=.5)  ###IXC
        #ctx.add_basemap(ax)  ###IXC
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron,alpha=.98)  ###IXC
        #ctx.add_basemap(ax)  ###IXC

    # ================== Add bars of FN, TP, FP ==============START
    width_ratio = barWidth / width
    height_ratio = barHeight / height
    parameters[events[0]]['bar_location'] = [0.7, 0.75, width_ratio, height_ratio]
    parameters[events[1]]['bar_location'] = [0.4, 0.75, width_ratio, height_ratio]
    if len(events) > 2:
        parameters[events[2]]['bar_location'] = [0.1, 0.75, width_ratio, height_ratio]

    plt.gcf().patches.extend([
        plt.Rectangle(
            (0.75, 0.4), 
            0.5, 0.6, 
            fill=True, 
            color='k', 
            alpha=1, 
            zorder=-101,
            transform=fig.transFigure, 
            figure=plt.gcf())
    ])
    # ======================== Time stamp ===================== START
    ax.text(
        0.825, 0.965,
        str(time_).split()[0], 
        transform=ax.transAxes,
        fontweight='light',
        fontsize=18,
        color=AXCOL,
        alpha=1,
        verticalalignment='bottom', 
        horizontalalignment='right', 
       # bbox=dict(
       #     boxstyle='round', 
      #      facecolor='k', 
       #     alpha=0.1)
    )
    # ======================== Time stamp ===================== END


    for var in events:
        cmap = parameters[var]['cmap']
        bar_location = parameters[var]['bar_location']
        
        bar_data = np.array([
            parameters[var]['fn'], 
            parameters[var]['tp'], 
            parameters[var]['fp']
        ])
        if bar_data.sum() > 0:
            bar_data = bar_data / bar_data.sum()
        else:
            bar_data = [0.,0.,0.]
        
        ax2 = plt.gcf().add_axes(bar_location, zorder=20)
        ax2.patch.set_alpha(1)
        ax2.set_facecolor("white")

        ax2.bar(
            ['FN','TP','FP'],
            bar_data,
            color=parameters[var]['bcolor'],
            lw=0,
            zorder=20,
            alpha=.9)

        ax2.spines['bottom'].set_color(AXCOL)
        ax2.spines['top'].set_color(AXCOL) 
        ax2.spines['right'].set_visible(False) 
        ax2.spines['left'].set_visible(False) 
        ax2.tick_params(axis='x', colors=AXCOL, pad=8)
        ax2.tick_params(axis='y', colors=AXCOL)
        ax2.set_ylim(0,1)

        ax2.set_title(
            var_title_map[var], 
            color=AXCOL, 
            fontdict={'fontsize': barTitleSize, 'fontweight': 'normal'})
        ttl = ax2.title
        ttl.set_position([.5, 1.05])
        ax2.grid(True)

        for label in ax2.get_xticklabels():
            label.set_color(AXCOL)
            label.set_fontsize(barTickerSize)
            label.set_fontweight('light')

        for label in ax2.get_yticklabels():
            label.set_color(AXCOL)
            label.set_fontsize(barTickerSize)
            label.set_fontweight('light')

        ax2.tick_params(axis=u'both', which=u'both',length=0)
    # ================== Add bars of FN, TP, FP ==============END
    sp.saveFIG(f'{MOVIEDIR}/snapshot_{str(time_).split()[0]}.png') 

def getPred(day=1000,Z=0.02,radius=.007,sigma=2.75,detail=0.0007,miles=.01):
    time = datetime_range[day].date()
    sum_={}
    for var in events:
        types = [var]
        fn, tp, fp, ppv, sens, lon_, lat_, int_plot, int_, dfG, dfFN, dfTP, dfFP, dfTP0 = get_prediction(
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
        
        sum_[var]={'fp':fp,'tp':tp,'fn':fn,'sens':sens,'ppv':ppv}
        parameters[var].update({
            'fp': fp, 'tp': tp, 'fn': fn,
            'ppv': ppv, 'sens': sens,
            'lon': lon_, 'lat': lat_,
            'int': int_, 'df': dfG,
            'dffp': dfFP, 'dftp': dfTP, 'dffn': dfFN,
            'dftp_0': dfTP0
        })
    return time,sum_,parameters#,time,events

#======================================= END FUNCTIONS ======================================

RESSIM0={}
RESSIM1={}
RESSIM2={}

# testing for 1 day
if DAYNUM is not None:
    day_max=day_min+DAYNUM

from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

def process_day(day, Z, radius, miles, detail):
    # Extracted operations from the original loop
    time_, sum_, parameters = getPred(day=day, Z=Z, radius=radius, miles=miles, detail=detail)
    # ... (the rest of the loop's operations) ...
    if FIGSHOW:
        getFig(time_)
    PRED=pd.concat([parameters[events[0]]['dftp'][['lat1','lat2','lon1','lon2']].drop_duplicates().assign(typ=var_title_map[events[0]]),
           parameters[events[1]]['dftp'][['lat1','lat2','lon1','lon2']]
                    .drop_duplicates().assign(typ=var_title_map[events[1]]),
                    parameters[events[2]]['dftp'][['lat1','lat2','lon1','lon2']]
                    .drop_duplicates().assign(typ=var_title_map[events[2]])]).drop_duplicates()
    PRED.to_csv(PREDPATH+'prediction_'+str(time_)+'.csv',index=None)
    return day, time_, sum_, parameters

# Parameters for the process_day function
Z = zx
radius = rx
miles = mx
detail = dx

# Modified for parallel execution
day_range = np.arange(day_min, day_max)
with ProcessPoolExecutor() as executor:
    results = list(tqdm(executor.map(process_day, day_range, repeat(Z),
                                     repeat(radius), repeat(miles),
                                     repeat(detail)), total=len(day_range)))

RESULTS=pd.concat([pd.DataFrame(result[2]).transpose().assign(date=result[1]) for result in results])
RESULTS.to_csv(OUTPUTDIR+'/res_day'+DESC+'.csv')
DF=None
for i in events:
    if DF is None:
        DF=pd.DataFrame(RESULTS.loc[i,:].mean(),columns=[i])
    else:
        DF=DF.join(pd.DataFrame(RESULTS.loc[i,:].mean(),columns=[i]))
        
DF.to_csv(OUTPUTDIR+'/output'+DESC+'.txt')
print(DF)
