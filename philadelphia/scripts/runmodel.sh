#!/bin/bash

DRYRUN=0
NAME='IXCsim'

PROG=' date; python digisimul.py; date'
echo -n '... launching '
T=30
LAUNCH='/project2/ishanu/LAUNCH_UTILITY/launcher_s.sh '
$LAUNCH -d $DRYRUN -P "$PROG"  -T $T -N 1 -C 28 -p  broadwl -J $NAME -M 30

rm *depx
rm *sbc
