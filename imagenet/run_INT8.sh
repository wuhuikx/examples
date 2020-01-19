model=$1 #resnext101_32x4d #resnet50
num_cores=$2  #56
num_threads=$3 #56
batch_sizes=$4 #"1"
use_mkldnn=--mkldnn
pretrained=--pretrained
# qengine="all"
evaluation=-e
image_path=/lustre/dataset/imagenet/img_raw/
iterations=$5 #100
#qengine="mkldnn"
qengine="fbgemm"
if (($# > 5)); then
    qengine=$6
fi
warmup=50
iter_calib=2500
#INT8="INT8_and_fp32"
#INT8="no_INT8"
INT8="INT8_only"
#INT8="FP32_only"
qscheme="perChannel"
#qscheme="perTensor"
log_level="info"
# profiling=-t
profiling=""
reduce_range=""
workers=0
model_url=https://download.pytorch.org/models/
# export GLOG_logtostderr=1; export GLOG_v=1000
# export MKLDNN_VERBOSE=1

# export LD_PRELOAD=/opt/intel/compilers_and_libraries_2018.1.163/linux/compiler/lib/intel64_lin/libiomp5.so
if [ $INT8 != 'no_INT8']
    echo "INT8=$INT8"
    then
    declare -A dict
    dict=([resnet18]='resnet18-5c106cde.pth'
        [resnet34]='resnet34-333f7ec4.pth'
        [resnet50]='resnet50-19c8e357.pth'
        [resnet101]='resnet101-5d3b4d8f.pth'
        [resnet152]='resnet152-b121ed2d.pth'
        [resnext50_32x4d]='resnext50_32x4d-7cdf4587.pth'
        [resnext101_32x4d]='checkpoint.pth.tar'
        [resnext101_32x8d]='resnext101_32x8d-8ba56ff5.pth'
        [wide_resnet50_2]='wide_resnet50_2-95faca4d.pth'
        [wide_resnet101_2]='wide_resnet101_2-32ee1156.pth'
    )
    if [ -z "${dict[$model]}" ];then
        echo "Unsupport this model "$model
        exit
    fi
    model_path="./models/"$model"/"${dict[$model]}
    model_download_path=$model_url${dict[$model]}
    if [ ! -f $model_path ]; then
        echo "download model state dict to ""./models/"$model"/"${dict[$model]}
        mkdir -p "./models/"$model
        wget -O $model_path $model_download_path
    fi
fi
echo "
export OMP_NUM_THREADS=$num_threads  KMP_AFFINITY=proclist=[$startid-$endid],granularity=fine,explicit
python -u main.py $pretrained $evaluation $reduce_range $profiling $use_mkldnn -j $workers -a $model -b $batch_sizes --INT8 $INT8 -qs $qscheme --iter-calib $iter_calib -w $warmup -qe $qengine  -i $iterations $image_path &  " >> command.sh

rm -rf logs
mkdir logs
# for bs in $batch_sizes; do
for i in $(seq 0 $(($num_cores / $num_threads - 1)))
do
echo $i "instance"
startid=$(($i*$num_threads))
endid=$(($i*$num_threads+$num_threads-1))
export OMP_SCHEDULE=STATIC OMP_NUM_THREADS=$num_threads OMP_DISPLAY_ENV=TRUE OMP_PROC_BIND=TRUE GOMP_CPU_AFFINITY="$startid-$endid"  
export OMP_NUM_THREADS=$num_threads  KMP_AFFINITY=proclist=[$startid-$endid],granularity=fine,explicit
python -u main.py $pretrained $evaluation $reduce_range $profiling $use_mkldnn -j $workers -a $model -b $batch_sizes --INT8 $INT8 -qs $qscheme --iter-calib $iter_calib -w $warmup -qe $qengine  -i $iterations $image_path 2>&1 | tee logs/instance_$i &  

done
wait

# report results
instances=$(($num_cores / $num_threads))
fps=$(grep 'performance' logs/instance_* | \
	awk '{ fps_sum += $2 } END { print fps_sum }')
latency=$(grep 'latency' logs/instance_* | \
	awk -v instances=$instances \
	'{ latency_sum += $2 } END { print latency_sum / instances }')

echo ---------*----------*---------
echo Performance result report
echo ---------*----------*---------
echo Engine: "$qengine"
echo Threads: $num_threads
echo Instances: $instances
echo Threads per instance: $num_cores
echo Iterations: $iterations
echo Batch size: $batch_sizes
echo Average latency for one image: ${latency}ms
echo FPS: $fps images/sec

