export CUDA_VISIBLE_DEVICES=3
python tracking/train.py --script $1 --config $2 --save_dir ./output --mode multiple --nproc_per_node 1
# ./train.sh m_step baseline_mste_tpe_full_16_18_20
