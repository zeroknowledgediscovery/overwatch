#!/bin/bash

DRYRUN=0
NAME='IXCperf'

PROG=' module load python/anaconda-2021.05; python ./generate_sim_future.py -Z 0.15 -r 0.009 -m 0.005 -d 0.001 -f False'
echo -n '... launching '
T=1
LAUNCH='/project2/ishanu/LAUNCH_UTILITY/launcher_s.sh '
$LAUNCH -d $DRYRUN -P "$PROG"  -T $T -N 1 -C 28 -p  broadwl -J $NAME -M 30

rm *depx
rm *sbc


