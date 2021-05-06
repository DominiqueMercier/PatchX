cd ../scripts/
echo "Current working dir: $PWD"

DATASETS=("../../data/character_trajectories/dataset_steps-20_timesteps-206.pickle" \
            "../../data/anomaly_new/anomaly_dataset.pickle" \
            "../../data/FordA/dataset_classes-2_timesteps-500.pickle" \
            "../../data/ElectricDevices/dataset_classes-7_timesteps-96.pickle")

for DATASET in ${DATASETS[@]}
do
    echo "=================================================="
    echo "Create Features: $DATASET"
    echo "=================================================="
    
    python3 main.py --path $DATASET \
        --force_compute_features
done

echo "=================================================="
echo "Finished"
echo "=================================================="
