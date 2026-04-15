export CUDA_VISIBLE_DEVICES=3
# python tracking/train.py --script $1 --config $2 --mode multiple --nproc_per_node 2 

python tracking/train.py --script $1 --config $2 --save_dir ./output --mode multiple --nproc_per_node 1
# ./train.sh hivitr baseline_256
# ./train.sh hivitr baseline_mste_256
# ./train.sh hivitr_time baseline_mste_tpe_full_v2
# ./train.sh hivitr_time baseline_mstev2_tpe_256
# ./train.sh hivitr_time baseline_mstev2_tpe_256_got
# ./train.sh hivitr_time baseline_mste_tpe_mlpG_got
# ./train.sh hivitr_time baseline_mste_tpe_full_16_18_20
# ./train.sh hivitr_time baseline_1m
# ./train.sh hivitr_time baseline_atten
# ./train.sh hivitr_time baseline_got_384
# ./train.sh hivitr_time baseline_mste_tpe_full_16_18_20_got_384