CUDA_VISIBLE_DEVICES=1 python calc_metrics.py --network=/home/lwlhsa/coding/layoutGAN/results_600/00000-stylegan2--gpus1-batch16/network-snapshot.pkl \
                        --metrics=is50k,fid50k_full  --data=/home/lwlhsa/coding/data/dataset_test_v5 \
                        --data1=/home/lwlhsa/coding/data/dataset4/
