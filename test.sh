export CUDA_VISIBLE_DEVICES=1,2,3
python tracking/test.py $1 $2 --dataset $3 --threads 16 --num_gpus 3 --runid 100
## Got-10k
# ./test.sh m_step baseline_mste_tpe_full_16_18_20 got10k_test
# python lib/test/utils/transform_got10k.py --tracker_name level_track --cfg_name  baseline_256_v1_got

## OTB
# ./test.sh m_step baseline_mste_tpe_full_16_18_20 otb
# python tracking/analysis_results.py

## UAV123
# ./test.sh m_step baseline_mste_tpe_full_16_18_20 uav
# python tracking/analysis_results.py

## TrackingNet
# ./test.sh m_step baseline_mste_tpe_full_16_18_20 trackingnet
# python lib/test/utils/transform_trackingnet.py --tracker_name m_step --cfg_name  baseline_mste_tpe_full_16_18_20_299

## LaSOT
# ./test.sh m_step baseline_mste_tpe_full_16_18_20 lasot
# python tracking/analysis_results.py # need to modify tracker configs and na

# LaSOText
# ./test.sh m_step baseline_mste_tpe_full_16_18_20 lasot_extension_subset
# python tracking/analysis_results.py

# VastTrack
# ./test.sh m_step baseline_mste_tpe_full_16_18_20 vasttrack
# python tracking/analysis_results.py
