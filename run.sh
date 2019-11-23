#!/bin/sh
#current_dir=$(cd "$(dirname $0)";pwd)

#project_root=$(cd $current_dir;cd ../../;pwd)
#thrift_root=$(cd $current_dir;cd ../../;pwd)

#pip3 install scipy==1.1.0  -i https://pypi.tuna.tsinghua.edu.cn/simple
#pip3 install pypng
#export PYTHONPATH=$PYTHONPATH:/xxx/sunyasheng/projects/tfoptflow/tfoptflow
#export PYTHONPATH=$PYTHONPATH:/Users/yashengsun/ArnoldProjects/tfoptflow/tfoptflow
#python3 test_model.py
#rm /xxx/sunyasheng/nohup_dir/pwc_flow.out
#nohup python3 train.py --mode train > ~/vgg_stage2.out&
#python3 train.py --mode train
#python3 train.py --mode psnr
#python3 train.py --mode evaluate
#python3 train.py --mode test
for file in `ls middlebury-eval`
do
    python3 train.py --mode evaluate --img_dir ./middlebury-eval/$file --out_dir ./middlebury_eval_out/$file
    echo $file
done

#python3 train.py --mode evaluate --img_dir ./run_man/ --out_dir ./run_man_out/
