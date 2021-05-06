cd ../scripts/
echo "Current working dir: $PWD"

DATASETS=("../../data/character_trajectories/dataset_steps-20_timesteps-206.pickle" \
            "../../data/anomaly_new/anomaly_dataset.pickle" \
            "../../data/FordA/dataset_classes-2_timesteps-500.pickle" \
            "../../data/ElectricDevices/dataset_classes-7_timesteps-96.pickle" \
            "../../data/daily_and_sport_activites/dataset_classes-19_timesteps-60.pickle")

lenD=${#DATASETS[@]}
TRIVIALMODES=("majority" "occurance" "majority" "majority" "majority")
FEATURES=(1 1 1 1 0)

STRIDESET=("10" "5")
LENGTHSET=("20" "10")
ZEROSET=(1 1)
ATTACHSET=(1 1)
NOTEMPSET=(0 0)
len=${#STRIDESET[@]}

for (( j=0; j<$lenD; j++ ))
do
    DATASET=${DATASETS[$j]}

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
        TRIVIALMODE=${TRIVIALMODES[$j]}

        echo "=================================================="
        echo "Strides: $STRIDES | Length: $LENGTH | Zero: $ZERO | Attach: $ATTACH | Notemp: $NOTEMP | TrivialMode: $TRIVIALMODE"
        echo "=================================================="
        
        FEATURE=${FEATURES[$j]}
        if (($FEATURE))
        then        
            python3 main.py --path $DATASET --store_times \
                --strides $STRIDES --length $LENGTH --zero $ZERO --attach $ATTACH --notemp $NOTEMP \
                --include_l1 --include_l2 --clf_type "svm" \
                --include_trivial --trivial_mode $TRIVIALMODE \
                --include_blackbox --include_simple_clf \
                --force_compute_features --use_dense --include_feature_blackbox --include_feature_simple_clf \
                --get_statistics --save_statistics
        else
            python3 main.py --path $DATASET --store_times \
                --strides $STRIDES --length $LENGTH --zero $ZERO --attach $ATTACH --notemp $NOTEMP \
                --include_l1 --include_l2 --clf_type "svm" \
                --include_trivial --trivial_mode $TRIVIALMODE \
                --include_blackbox --include_simple_clf \
                --get_statistics --save_statistics
        fi
    done
done

echo "=================================================="
echo "Finished"
echo "=================================================="