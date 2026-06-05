import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import os
import sys
import argparse
from zipfile import ZipFile

if os.path.basename(os.getcwd()) != 'P114-Annotation-Free-Segmentation':
    raise RuntimeError("Please run this script from the project root directory 'P114-Annotation-Free-Segmentation' to ensure correct paths.")

sys.path.append(os.getcwd())
print(f"Current working directory: {os.getcwd()}\n")

from models import get_model
from trainers import get_trainer
from evaluation import *
from data import Augmentor

def main():
    parser = argparse.ArgumentParser(description="Run Segmentation Inference Queue")
    parser.add_argument('--model', type=str, required=True, choices=['ceVAE', 'RPCA', 'RVAE', 'RDA', 'RDDPM', 'Opus'], help="The model architecture to run")
    parser.add_argument('--param', type=str, required=True, help="The augmentation parameter (e.g., dense_noise)")
    parser.add_argument('--start', type=float, default=0.0, help="Starting value for the parameter")
    parser.add_argument('--end', type=float, default=0.6, help="Ending value for the parameter")
    parser.add_argument('--step', type=float, default=0.1, help="Step size for the parameter")
    parser.add_argument('--dataset', type=str, default='CDNet', choices=['CDNet', 'YaleB', 'Brats', 'Hazelnut', 'Metal_nut'], help="The dataset to use (e.g., CDNet, YaleB)")
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    zip_paths = {
        'CDNet': "./datasets/CDNet",
        'YaleB': "./datasets/yaleB",
        'Brats': "./datasets/brats",
        'Hazelnut': "./datasets/hazelnut",
        'Metal_nut': "./datasets/metal_nut",
    }

    dataset_names = {
        'CDNet': "highway",
        'YaleB': "01",
        'Brats': "BraTS2020_Train",
        'Hazelnut': "train",
        'Metal_nut': "train"
    }

    DATA_ZIP_PATH = zip_paths[args.dataset]
    DATASET_NAME = dataset_names[args.dataset]

    if os.path.exists(DATA_ZIP_PATH) is False:
        with ZipFile(f"{DATA_ZIP_PATH}.zip", 'r') as zip:
            zip.extractall("./datasets/")



    DATASET_PATH= os.path.join(DATA_ZIP_PATH, DATASET_NAME)
    DIMS = (128, 128)

    MODEL = args.model
    PARAM = args.param

    BATCH_SIZE = {
        "ceVAE": 256,#64,
        "RVAE": 512,#128,
        "RDA": 512,
        "RDDPM": 4,
        "Opus": 256,#64
    }
    
    MODEL_ARGS = {
        "ceVAE":{"latent_channels": 1024, "with_r": True},
        "RPCA":{"max_iter": 6000, "lambda_": None, "tol": 1e-7},
        "RVAE":{"latent_dim": 2},
        "RDA":{ 'latent_dim': 16, 'hidden_dim': 1024},
        "RDDPM": { "img_channels": 1 },
        "Opus":{"latent_channels": 1024, "with_r": True},
    }
    
    FIT_ARGS = {
        "ceVAE": {'epochs': 60, 'lr': 2e-4, "lambda_": 0.85},
        "RVAE": {'epochs': 20, 'lr': 2e-3, "beta": 5e-4},
        "RDA": {'lr': 1.5e-4, 'lambda_': (1.0 / torch.sqrt(torch.tensor(DIMS[0])))*1, 'outer_epochs': 10, 'inner_epochs': 10},
        "RDDPM": {"lr": 2e-4, "epochs": 10, "loss_type": "huber", "robust_param": 0.1},
        "Opus": {'outer_epochs': 6, 'inner_epochs': 10, 'lr': 2e-4, "lambda_ce": 0.85, "lambda_": 1.0/torch.sqrt(torch.tensor(DIMS[0]))}, ####tune
    }

    print(f"--- Starting run for Model: {MODEL} | Param: {PARAM} | Range: {args.start} to {args.end} (Step: {args.step}) ---")
    print(f"Using device: {DEVICE}\n")
    print(torch.version.cuda)

    auroc_metric = BinaryAUROC()
    auprc_metric = BinaryAveragePrecision()

    dice_results = {}
    auroc_results = {}
    auprc_results = {}

    # Safely generate floating-point steps
    intensities = []
    current = args.start
    while current <= args.end + 1e-9:
        intensities.append(round(current, 4))
        current += args.step

    for intensity in intensities:
        print(f"Processing dataset: {DATASET_NAME} with intensity {intensity}...")
        aug = Augmentor(source_root=DATASET_PATH, device=DEVICE, **{PARAM: intensity})

        data = aug.get_dataset(
            transform=transforms.Compose([
                transforms.Resize(DIMS),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor()
            ])
        )
        
        if MODEL != 'RPCA':
            train_loader = DataLoader(data, batch_size=BATCH_SIZE[MODEL], shuffle=True, num_workers=0)
            print(f"Train loader has {len(train_loader)} batches of size {BATCH_SIZE[MODEL]}.")

        eval_loader = DataLoader(data, batch_size=len(data), shuffle=False, num_workers=0)

        print(f"Eval loader has {len(eval_loader)} batches of size {len(data)}.")

        if MODEL != 'Opus':
            model = get_model(MODEL, **MODEL_ARGS[MODEL]).to(DEVICE)
        else:
            model = get_model('ceVAE', **MODEL_ARGS[MODEL]).to(DEVICE)

        models_dict = {MODEL: model}

        if MODEL != 'RPCA':
            trainer = get_trainer(MODEL, model, train_loader)
            trainer.fit(**FIT_ARGS[MODEL])

            del trainer
            del train_loader
            os.makedirs(f"saved_models/{MODEL}/{DATASET_NAME}/", exist_ok=True)
            torch.save(model.state_dict(), f"saved_models/{MODEL}/{DATASET_NAME}/{PARAM}_{intensity}.pth")

        torch.cuda.empty_cache()

        batch = next(iter(eval_loader))
        del eval_loader
        X_input, _, _ = batch
        
        if MODEL == 'RPCA':
            results_root = os.path.abspath(f"./results/{aug.dest_root.split('/')[-1]}/")
            _, L_rpca, S_rpca = get_rpca_decomposition(X_input, model, results_root, force_recompute=True, target_size=aug.init_dims, exact=False)
            model_results = {MODEL: (L_rpca, S_rpca)}
        else:
            model_results = run_deep_models_inference(X_input, models_dict, target_size=aug.init_dims, batch_size=BATCH_SIZE[MODEL]//2)

        del models_dict
        del model
        del X_input
        torch.cuda.empty_cache()
        
        truth = aug.get_gt_images()
        
        del aug

        if truth is not None:
            shuffle_idx = torch.randperm(len(truth))
            shuffled_truth = truth[shuffle_idx]

            validation_truth = shuffled_truth[:len(shuffled_truth)//2]
            test_truth = shuffled_truth[len(shuffled_truth)//2:]

            y_true = torch.as_tensor(truth).flatten().abs().int()

            for name, (L, S) in model_results.items():

                shuffled_S = S[shuffle_idx]
                t, dice = find_optimal_dice(validation_truth, shuffled_S[:len(shuffled_truth)//2], bins=1000)
                test_dice = calculate_dice(test_truth, shuffled_S[len(shuffled_truth)//2:], threshold=t)

                print(f"Dice Score for {name} on test set:\t {test_dice} at threshold {t:.4f}")
                dice_results[intensity] = test_dice

                y_scores = torch.as_tensor(S).flatten().abs()

                auroc_metric.update(y_scores, y_true)
                auroc = auroc_metric.compute().item() 

                print(f"AUROC for {name}: {auroc:.4f}")
                auroc_results[intensity] = auroc
                auroc_metric.reset()

                auprc_metric.update(y_scores, y_true)
                auprc = auprc_metric.compute().item()

                print(f"Average Precision for {name}: {auprc:.4f}")
                auprc_results[intensity] = auprc
                auprc_metric.reset()

        del model_results
        del truth

        torch.cuda.empty_cache()


    os.makedirs(f"demos/results/{DATASET_NAME}/{PARAM}", exist_ok=True)

    with open(f"demos/results/{DATASET_NAME}/{PARAM}/{MODEL}.txt", "w") as f:
        f.write(f"Results for {MODEL} with varying {PARAM} on {DATASET_NAME}:\n")
        for intensity, dice in dice_results.items():
            f.write(f"Intensity: {intensity}, Dice: {dice}, AUROC: {auroc_results[intensity]}, AUPRC: {auprc_results[intensity]}\n")

if __name__ == "__main__":
    main()