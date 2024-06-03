# get data
getData.py

# runrecon.sh 
python3 fnet_script.py; cp program_calls.txt program_calls0.txt; ./phnx.sh

# runmodel.sh
python digisimul.py

# runperf.sh
./generate_sim_future.py -Z 0.17 -r 0.009 -m 0.005 -d 0.001 -f False

