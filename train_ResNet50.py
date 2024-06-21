import torch
import wandb
import argparse

import torch.nn as nn
import torchvision.models as models

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights
from sklearn.metrics import average_precision_score, precision_score, recall_score, precision_recall_curve, roc_curve, auc
import numpy as np

from data.ksdd2 import KolektorSDD2

class KSDD2ResNet50(nn.Module):
    def __init__(self):
        super(KSDD2ResNet50, self).__init__()

        # Load the pre-trained ResNet-50 model from torchvision.models.
        self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        # Change the output layer to output 1 class score instead of 1000 classes.
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 1)

    def forward(self, x):
        return self.model(x)
    

def evaluate(model, criterion, test_loader, device, log_dict):
    t_loss = 0
    correct = 0
    targets = []
    predictions = []

    model.eval()
    with torch.no_grad():
        for _, data in (tepoch := tqdm(enumerate(test_loader), unit='batch',
                                       total=len(test_loader), desc='Validation')):
            x, y = data[0].to(device), data[1].to(device)

            # This gets the prediction from the network.
            output = model(x)
            output = output.squeeze(1)
            # Sum up batch loss.
            t_loss += criterion(output, y.float()).item()

            # Get the prediction
            pred = output
            
            predictions.extend(pred.cpu().numpy())
            targets.extend(y.cpu().numpy())

    t_loss /= len(test_loader)

    precision_, recall_, thresholds = precision_recall_curve(targets, predictions)
    f_measures = 2 * (precision_ * recall_) / (precision_ + recall_ + 0.0000000001)

    # Select best threshold based on F2 score. Following previous works procedure.
    ix_best = np.argmax(f_measures)
    if ix_best > 0:
        best_threshold = (thresholds[ix_best] + thresholds[ix_best - 1]) / 2
    else:
        best_threshold = thresholds[ix_best]
    precision = precision_[ix_best]
    recall = recall_[ix_best]

    classifications = predictions > best_threshold

    FPR, TPR, _ = roc_curve(targets, predictions)
    AUC = auc(FPR, TPR)
    AP = average_precision_score(targets, predictions)

    # Calculate predictions based on best threshold.
    correct = np.sum(classifications == targets)
    accuracy = 100. * correct / len(classifications)

    print('AVG loss: {:.4f}, ACC: {}/{} ({:.0f}%), Precision: {:.4f}, Recall: {:.4f}, AP: {:.4f}'.format(
            t_loss, correct, len(test_loader.dataset), accuracy, precision, recall, AP))
    
    # log metrics
    log_dict['val_ACC'] = accuracy
    log_dict['val_PRECISION'] = precision
    log_dict['val_RECALL'] = recall
    log_dict['val_AP'] = AP

    return log_dict


def main(args):
    # Set the seed for reproducibility.
    torch.manual_seed(args.seed)
    # Set the device.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    add_augmented = args.add_augmented
    num_augmented = args.num_augmented
    zero_shot = args.zero_shot
    logging = args.logging

    run_name = f'KSDD2ResNet50-zero_shot_{zero_shot}-add_augmented_{add_augmented}-num_augmented_{num_augmented}-bs_{args.batch_size}-epochs_{args.epochs}'
    tags = [f'{args.epochs}epochs', f'{num_augmented}augmented']
    if args.zero_shot:
        tags.append('zero_shot')
    else:
        tags.append('full_shot')
    if args.add_augmented:
        tags.append('augmented')
    else:
        tags.append('not_augmented')
    
    if logging:
        # Start a new wandb run to track this script.
        wandb.init(
            name=run_name,
            config=args,
            tags=tags
        )

    # Dataset.
    print('Loading KolektorSDD2 training set...')
    train_data = KolektorSDD2(dataroot=args.dataset_path, split='train', add_augmented=add_augmented, num_augmented=num_augmented, zero_shot=zero_shot)
    print('Number of samples:', len(train_data))

    print('Loading KolektorSDD2 test set...')
    test_data = KolektorSDD2(dataroot=args.dataset_path, split='test')
    print('Number of samples:', len(test_data))

    # DataLoaders.
    train_loader = DataLoader(train_data, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_data, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)

    # Define the model.
    model = KSDD2ResNet50()
    model.to(device)

    # Define the loss function and the optimizer
    criterion = nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training step.
    print(f'Start training on {device} [...]')
    model.train()
    log_dict = {'train_loss': 0, 'val_ACC': 0, 'val_PRECISION': 0, 'val_RECALL': 0, 'val_AP': 0, 'epoch': 0}
    for e in range(args.epochs):
        epoch_loss = 0
        for _, data in (tepoch := tqdm(enumerate(train_loader), unit='batch',
                                       total=len(train_loader))):
            tepoch.set_description(f'Epoch {e}')
            x, y = data[0].to(device), data[1].to(device)

            # Training step for the single batch.
            model.zero_grad()
            outputs = model(x)
            outputs = outputs.squeeze(1)
            loss = criterion(outputs, y.float())
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

            # Print statistics.
            tepoch.set_postfix(loss=loss.item())
            if logging:
                wandb.log({'train_loss':loss.item()})
        epoch_loss /= len(train_loader)
        log_dict['epoch_loss'] = epoch_loss
        log_dict['epoch'] = e

        # Evaluation step after each epoch.
        eval_dict = evaluate(model, criterion, test_loader, device, log_dict)
        if logging:
            wandb.log(eval_dict)
    
    if logging:
        wandb.finish()
    print('Training finished.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DIAG training')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--add_augmented', action='store_true', help='Add augmented images to the training set')
    parser.add_argument('--num_augmented', type=int, default=120)
    parser.add_argument('--zero_shot', action='store_true', help='Train the model without true positives in the training set')
    parser.add_argument('--logging', action='store_true', help='Log the stats to wandb')
    

    args = parser.parse_args()
    main(args)