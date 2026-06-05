# !/bin/bash

# Force the terminal to use the directory where this script is saved
cd "$(dirname "$0")" || exit

# source .venv/bin/activate

echo "=========================================="
echo "Starting Segmentation Inference Queue..."
echo "=========================================="

echo "Running rotation experiments \n"

for model in ceVAE RPCA RVAE RDA RDDPM Opus; do
    for dataset in CDNet Hazelnut Metal_nut; do
        echo "Running $model on $dataset with rotation parameter \n"
        python gen_res.py --model $model --param rotation --start 0.0 --end 180.0 --step 18.0 --dataset $dataset
    done
done

echo "Running dense_noise experiments \n"

for model in ceVAE RPCA RVAE RDA RDDPM Opus; do
    for dataset in CDNet Hazelnut Metal_nut; do
        echo "Running $model on $dataset with dense_noise parameter \n"
        python gen_res.py --model $model --param dense_noise --start 0.0 --end 0.3 --step 0.03 --dataset $dataset
    done
done

echo "Running tps experiments \n"

for model in ceVAE RPCA RVAE RDA RDDPM Opus; do
    for dataset in CDNet Hazelnut Metal_nut; do
        echo "Running $model on $dataset with tps parameter \n"
        python gen_res.py --model $model --param tps --start 0.0 --end 0.8 --step 0.08 --dataset $dataset
    done
done

echo ""
echo "=========================================="
echo "All experiments have finished successfully!"
echo "=========================================="

# deactivate