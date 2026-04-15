export CUDA_VISIBLE_DEVICES=1,2,3
python tracking/test.py $1 $2 --dataset $3 --threads 16 --num_gpus 3 --runid 100
# python lib/test/utils/transform_got10k.py --tracker_name level_track --cfg_name  baseline_256_got
# python tracking/test.py hivitr_time baseline_mste_tpe_full --dataset lasot --threads 1 --num_gpus 1 --debug 1
## Got-10k
# python tracking/test.py odtrack baseline_got --dataset got10k_test  --runid 100 --threads 8 --num_gpus 2
# python lib/test/utils/transform_got10k.py --tracker_name level_track --cfg_name  baseline_256_v1_got
# ./test.sh hivitr_time baseline_mste_tpe_got_v2 got10k_test

## OTB
# python tracking/analysis_results.py
# ./test.sh hivitr_time baseline_mste_tpe_full_16_18_20 otb

## NFS
# python tracking/analysis_results.py
# ./test.sh hivitr_time baseline_mste_tpe_full nfs

## UAV123

# ./test.sh hivitr_time baseline_mste_tpe_full_16_18_20 uav
# python tracking/analysis_results.py

## TrackingNet
# python tracking/test.py odtrack baseline --dataset trackingnet  --runid 300 --threads 8 --num_gpus 2
# python lib/test/utils/transform_trackingnet.py --tracker_name hivitr --cfg_name  baseline_mste_256_294
# ./test.sh hivitr_time baseline_mste_tpe_full_16_18_20_384 trackingnet
# ./test.sh hivitr_time baseline_mste_tpe_full_16_18_20 trackingnet
# ./test.sh hivitr_time baseline_mstev2_tpe_256 trackingnet
# python lib/test/utils/transform_trackingnet.py --tracker_name hivitr_time --cfg_name  baseline_mste_tpe_full_16_18_20_299

# python tracking/test.py hivitr_time baseline_mste_tpe_full --dataset lasot --runid 299 --threads 1 --sequence 'car-6' --num_gpus 1 --debug 1
## LaSOT
# python tracking/test.py odtrack baseline --dataset lasot --runid 300 --threads 8 --num_gpus 2
# python tracking/analysis_results.py # need to modify tracker configs and na
# ./test.sh hivitr_time baseline_mste_tpe_full_16_18_20 lasot

# ./test.sh hivitr_time baseline_onlymamba lasot
# ./test.sh hivitr_time baseline_onlyattention lasot

# LaSOText
# ./test.sh hivitr_time baseline_2m lasot_extension_subset
# python tracking/analysis_results.py

# python tracking/test.py hivitr_time baseline_mste_tpe_full_16_18_20 --dataset lasot --threads 1 --num_gpus 1 --debug 1
# python tracking/test.py hivitr_time baseline_mste_tpe_full_16_18_20 --dataset lasot --runid 299 --sequence 'basketball-6' --threads 1 --num_gpus 1 --debug 1
# python tracking/test.py hivitr_time baseline_mste_tpe_full_16_18_20 --dataset vasttrack --runid 299 --threads 16 --num_gpus 2 --debug 0

#python tracking/test.py hivitr_time baseline_mste_tpe_full_16_18_20_384 --dataset got10k_test  --runid 292 --threads 16 --num_gpus 2