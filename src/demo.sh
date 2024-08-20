# FSRnet baseline model (x2) + JPEG augmentation

ROOT_DIR=./experiment/
SUB_DIR=FSRnet/
DIR=$ROOT_DIR$SUB_DIR
if [ ! -d $DIR ];then 
    mkdir $DIR
fi

LOG_NAME=`date +%Y-%m-%d-%H-%M-%S`.log
LOG=$DIR$LOG_NAME
## denotes not compared 
# denotes selected as compared method
#======================================scale==============================================
#=======================================x2================================================
nohup python -u main.py --template FSRnet --print_every 100 --gpunum '3' --n_GPUs 1 --scale 2  --data_augment --reset --save_results 2>&1  | tee $LOG &

#======================================scale==============================================
#=======================================x4================================================
nohup python -u main.py --template FSRnet --print_every 100 --gpunum '3' --n_GPUs 1 --scale 4  --data_augment --reset --save_results 2>&1  | tee $LOG &

#======================================scale==============================================
#=======================================x8================================================
nohup python -u main.py --template FSRnet --print_every 100 --gpunum '3' --n_GPUs 1 --scale 8  --data_augment --reset --save_results 2>&1  | tee $LOG &
