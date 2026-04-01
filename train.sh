#NUM=600
CUDA_VISIBLE_DEVICES=0 python train2.py --data=../data/dataset4/ --data2=../data/dataset3 \
                                        --outdir=results_all \
                                        --gpus=1 --batch=16 --cfg=stylegan2

                                        # --subset=$NUM \