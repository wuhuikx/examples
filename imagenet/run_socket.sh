KMP_BLOCKTIME=1 KMP_HW_SUBSET=1t KMP_AFFINITY=granularity=fine,compact,1,0 OMP_NUM_THREADS=28 numactl -C0-27 -m0 python -u main.py -a resnet50 --mkldnn /lustre/dataset/imagenet/img_raw -b 128 -j 0 --world-size=2 --rank=0 --dist-backend=gloo --dist-url="tcp://192.168.20.58:7689" & KMP_BLOCKTIME=1 KMP_HW_SUBSET=1t KMP_AFFINITY=granularity=fine,compact,1,0 OMP_NUM_THREADS=28 numactl -C28-55 -m1 python -u main.py -a resnet50 --mkldnn /lustre/dataset/imagenet/img_raw -b 128 -j 0 --world-size=2 --rank=1 --dist-backend=gloo --dist-url="tcp://192.168.20.58:7689"
