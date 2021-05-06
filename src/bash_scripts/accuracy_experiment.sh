cd ../scripts/
echo "Current working dir: $PWD"

DATASETS=("../../data/character_trajectories/dataset_steps-20_timesteps-206.pickle" \
            "../../data/anomaly_new/anomaly_dataset.pickle" \
            "../../data/FordA/dataset_classes-2_timesteps-500.pickle" \
            "../../data/ElectricDevices/dataset_classes-7_timesteps-96.pickle" \
            "../../data/daily_and_sport_activites/dataset_classes-19_timesteps-60.pickle")

STRIDESET=("10" "10" "5" "5,10")
LENGTHSET=("20" "20" "10" "10,20")
ZEROSET=(1 1 1 1)
ATTACHSET=(1 1 1 1)
NOTEMPSET=(0 1 0 0)
len=${#STRIDESET[@]}

for DATASET in ${DATASETS[@]}
do
    echo "=================================================="
    echo "Start experiments: $DATASET"
    echo "=================================================="

    for (( i=0; i<$len; i++ ))
    do 
        STRIDES=${STRIDESET[$i]}
        LENGTH=${LENGTHSET[$i]}
        ZERO=${ZEROSET[$i]} 
        ATTACH=${ATTACHSET[$i]}
        NOTEMP=${NOTEMPSET[$i]}

        echo "=================================================="
        echo "Strides: $STRIDES | Length: $LENGTH | Zero: $ZERO | Attach: $ATTACH | Notemp: $NOTEMP"
        echo "=================================================="
        python3 main.py --path $DATASET \
            --strides $STRIDES --length $LENGTH --zero $ZERO --attach $ATTACH --notemp $NOTEMP \
            --include_blackbox --get_statistics --save_statistics
    done
done

echo "=================================================="
echo "Finished"
echo "=================================================="