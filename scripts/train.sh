python ddp_train.py --batch_size 100 --lr 0.001 --timesteps 20 --patch_size 4 --dataset_fraction 0.25 --device cuda:0

# for distributed training
# CUDA_VISIBLE_DEVICES=0,1 python ddp_train.py --batch_size=100 --lr=0.001 --timesteps=20 --patch_size=4 --dataset_fraction=0.25 --ddp
