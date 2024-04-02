#!/bin/bash

DRYRUN=0
NAME='IXCrec'

PROG=' module load python; python3 fnet_script.py'
echo -n '... launching '
T=20
LAUNCH='/project2/ishanu/LAUNCH_UTILITY/launcher_s.sh '
$LAUNCH -d $DRYRUN -P "$PROG"  -T $T -N 1 -C 28 -p  broadwl -J $NAME -M 30

rm *depx
rm *sbc
