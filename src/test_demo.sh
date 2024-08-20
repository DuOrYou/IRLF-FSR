# EDSR baseline model (x2) + JPEG augmentation

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

 nohup python -u test_only.py --template FSRnet --gpunum '0' --n_GPUs 1 --scale 2  2>&1  | tee $LOG &

