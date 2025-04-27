import os
import numpy as np
from colorama import Fore
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import argparse
import pickle

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from PolypClassificationModel import PolypClassificationModel
from PolypDataset import PolypDataset

import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
EXPERIMENT_NAME = "baseline_classification_model"
mlflow.set_experiment(EXPERIMENT_NAME)

assert torch.cuda.is_available(), "GPU is not enabled"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def preprocess_files(one_vs_rest=False):
    if one_vs_rest:
        train_file_path = './data/train_set_AD_rest.pkl'
        val_file_path = './data/val_set_AD_rest.pkl'
        test_file_path = './data/test_set_AD_rest.pkl'
    
    else:
        train_file_path = './data/train_set_complete_imgs.pkl'
        val_file_path = './data/val_set_complete_imgs.pkl'
        test_file_path = './data/test_set_complete_imgs.pkl'


    # TRAINING SET
    # if os.path.exists(train_file_path):
    #     print("File already exsits. Loading...")
    #     with open(train_file_path, 'rb') as f:
    #         train_set = pickle.load(f)
    #     print(f"Loaded file {train_file_path}")

    # else:
    train_set = PolypDataset(image_dir="./data/m_train2/m_train/images",
                            csv_file="./data/m_train2/m_train/train.csv",
                            # mask_dir="./data/m_train2/m_train/masks",
                            transformations=True,
                            one_vs_rest=one_vs_rest
    )
    with open(train_file_path, "wb") as f:
        pickle.dump(train_set, f)
    
    print(f"File saved at {train_file_path}")



    # VALIDATION SET
    # if os.path.exists(val_file_path):
    #     print("File already exsits. Loading...")
    #     with open(val_file_path, 'rb') as f:
    #         val_set = pickle.load(f)
    #     print(f"Loaded file {val_file_path}")
        
    # else:
    val_set = PolypDataset(image_dir="./data/m_valid/m_valid/images",
                        csv_file="./data/m_valid/m_valid/valid.csv",
                        # mask_dir="./data/m_valid/m_valid/masks",
                        transformations=True,
                        one_vs_rest=one_vs_rest
        )
    with open(val_file_path, "wb") as f:
        pickle.dump(val_set, f)
            
    print(f"File saved at {val_file_path}")
        
        
    
    # TEST SET
    # if os.path.exists(test_file_path):
    #     print("File already exsits. Loading...")
    #     with open(test_file_path, 'rb') as f:
    #         test_set = pickle.load(f)
    #     print(f"Loaded file {test_file_path}")
        
    # else:
    test_set = PolypDataset(image_dir="./data/m_test/m_test/images",
                        csv_file="./data/m_test/m_test/gt_test.csv",
                        transformations=True,
                        one_vs_rest=one_vs_rest
    )
    with open(test_file_path, "wb") as f:
        pickle.dump(test_set, f)
            
    print(f"File saved at {test_file_path}")
    
    return train_set, val_set, test_set


def get_class_weights(dataset):
    classes = np.unique(dataset.labels)  # or [0, 1, 2]

    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=dataset.labels)
    class_weight_dict = dict(zip(classes, class_weights))
    print(class_weight_dict)
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    print(class_weights)
    
    return class_weights, class_weight_dict


def train(num_epochs, model, train_loader, val_loader, optimizer, criterion, n_early_stopping, path_model=None, scheduler=None):
    best_val_loss = float('inf')
    best_val_acc = 0.0
    train_loss_hist = []
    val_loss_hist = []
    early_stopping = 0
    

    model.to(device)

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        train_loss = 0.0
        correct_preds = 0
        total_preds = 0

        for batch in train_loader:
            optimizer.zero_grad()  # Clear gradients
            
            # Extract inputs and labels
            images = batch[0].to(device)  # Assuming first element is images
            labels = batch[1].to(device)  # Assuming second element is labels

            logits = model(images)
            loss = criterion(logits, labels)

            _, predicted_labels = torch.max(logits, dim=1)  
            correct_preds += (predicted_labels == labels).sum().item()
            total_preds += labels.size(0)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_acc = correct_preds / total_preds
        train_loss_hist.append(avg_train_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, Accuracy: {train_acc:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct_preds = 0
        val_total_preds = 0

        with torch.no_grad():

            for batch in val_loader:
                images = batch[0].to(device)
                labels = batch[1].to(device)

                logits = model(images)
                loss = criterion(logits, labels)

                _, predicted_labels = torch.max(logits, dim=1)
                val_correct_preds += (predicted_labels == labels).sum().item()
                val_total_preds += labels.size(0)

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct_preds / val_total_preds
        val_loss_hist.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_acc:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_acc = val_acc
            torch.save(model.state_dict(), path_model)
        else:
            early_stopping += 1
            print(Fore.YELLOW + f"Early stopping: {early_stopping}")

        if early_stopping == n_early_stopping:
            print(Fore.RED + f"Stopping early at epoch {epoch+1} with best loss of {best_val_loss}.\nBest model: {path_model}")
            break

        if scheduler:
            scheduler.step()

    print(Fore.GREEN + f"Training completed successfully!")
    return train_loss_hist, val_loss_hist, best_val_acc


def plot_loss(train_loss_hist, val_loss_hist, plot_path):
    plt.figure(figsize=(10, 6))
    
    # Plotting training loss
    plt.plot(range(1, len(train_loss_hist) + 1), train_loss_hist, label="Training Loss", color="blue", linestyle='-', marker='o')
    
    # Plotting validation loss
    plt.plot(range(1, len(val_loss_hist) + 1), val_loss_hist, label="Validation Loss", color="red", linestyle='--', marker='o')
    
    plt.title("Training and Validation Losses")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_path)
    plt.show()
    
    
def evaluate_model(model, path_model, test_loader, dataset, run_id, timestamp):
    model.load_state_dict(torch.load(path_model))
    print(Fore.GREEN + f"Loaded {os.path.basename(path_model)}")
    
    model.to(device)
    model.eval()
    predicted_labels_idxs = []
    true_labels_idxs = []
    
    with torch.no_grad():
        
        for batch in test_loader:
            images = batch[0].to(device)
            labels_idxs = batch[1].to(device)
            
            logits = model(images)
            _, pred_labels_idxs = torch.max(logits, dim=1)
            
            predicted_labels_idxs.extend(pred_labels_idxs.cpu().numpy())
            true_labels_idxs.extend(labels_idxs.cpu().numpy())

    predicted_labels = [dataset.dic_idx2label[idx] for idx in predicted_labels_idxs]
    true_labels = [dataset.dic_idx2label[idx] for idx in true_labels_idxs]
    
    
    with mlflow.start_run(run_id=run_id):
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)
        recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=0)
        f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=0)
        mlflow.log_metric("test_accuracy", f"{accuracy:.4f}")
        mlflow.log_metric("precision", f"{precision:.4f}")
        mlflow.log_metric("recall", f"{recall:.4f}")
        mlflow.log_metric("f1_score", f"{f1:.4f}")
        
        all_metrics = classification_report(true_labels, predicted_labels, labels=sorted(list(set(true_labels))), output_dict=True)
        df_metrics = pd.DataFrame(all_metrics).transpose()
        path_report = f'./results/metrics_report_{timestamp}.csv'
        df_metrics.to_csv(path_report)
        mlflow.log_artifact(path_report, "results")
        
        # Confusion Matrix
        conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=sorted(list(set(true_labels))))
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=sorted(list(set(true_labels))), yticklabels=sorted(list(set(true_labels))))
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        path_cm = f'./results/confusion_matrix_{timestamp}.png'
        plt.savefig(path_cm)
        plt.show()
        mlflow.log_artifact(path_cm, "results")

        # Print metrics
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        metrics = {
            "accuracy": f"{accuracy:.4f}",
            "precision": f"{precision:.4f}",
            "recall": f"{recall:.4f}",
            "f1_score": f"{f1:.4f}"
        }
    
    return metrics, conf_matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--hidden_features", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--one_vs_all", action="store_true")
    parser.add_argument("--weighted_loss", action="store_true")
    parser.add_argument("--weighted_sampling", action="store_true")
    
    args = parser.parse_args()
    
    techniques = []
    
    # ================ Construct dataset ==================
    print("Constructing datasets...")
    if args.one_vs_all:
        train_set, val_set, test_set = preprocess_files(one_vs_rest=True)
        techniques.append("ad vs rest")
    else:
        train_set, val_set, test_set = preprocess_files()
    print("Transformations to apply:", train_set.transformations_list)
    print("Datasets created")

    # ========= Construct dataloader ========================
    print("Constructing dataloaders...")
    batch_size = args.batch_size
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=4)
    
    if args.weighted_sampling:
        _, class_weights_dict = get_class_weights(train_set)
        sample_weights = [class_weights_dict[label] for label in train_set.labels]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

        train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, num_workers=4, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=4)
        test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=4)
        
        techniques.append('weighted sampling')
        
    print(Fore.GREEN + f"Dataloaders created")
     
    # ====================== Set hyperparameters ==========================
    print("Setting hyperparameters...")
    learning_rate = args.learning_rate
    num_epochs = 100
    early_stopping = 10
    weight_decay = args.weight_decay
    hidden_features = args.hidden_features
    dropout = args.dropout

    polyp_model = PolypClassificationModel(num_classes=len(train_set.dic_label2idx), dropout=dropout, hidden_features=hidden_features)
    if args.weighted_loss:
        class_weights_tensor, _ = get_class_weights(train_set)
        class_weights_tensor = class_weights_tensor.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        techniques.append('weighted loss')
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(polyp_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    if len(techniques) == 0:
        params = {
            "transformations": train_set.transformations_list,
            "criterion": "CrossEntropy",
            "optimizer": "Adam",
            "hidden_features": hidden_features,
            "batch_size": batch_size,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "num_epohcs": num_epochs,
            "early_stopping": early_stopping,
            "other_techniques": "None"
        }
    else:
        params = {
            "transformations": train_set.transformations_list,
            "criterion": "CrossEntropy",
            "optimizer": "Adam",
            "hidden_features": hidden_features,
            "batch_size": batch_size,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "num_epohcs": num_epochs,
            "early_stopping": early_stopping,
            "other_techniques": techniques
        }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"classifier_{timestamp}.pth"
    model_path = f'./models/baseline_classification/{model_name}'
    if os.path.exists(os.path.dirname(model_path)):
        print(True)
        print(model_path)
    else:
        print(False)
        
    print("Hyperparameters set")
        
    # =============== TRAINING ===============================
    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_params(params)
        print("Training...")
        train_loss_hist, val_loss_hist, val_accuracy = train(num_epochs, 
                                            polyp_model, 
                                            train_loader, 
                                            val_loader, 
                                            optimizer, 
                                            criterion,
                                            early_stopping,
                                            path_model=model_path)
        
        mlflow.log_metric("best_val_accuracy", f"{val_accuracy:.4f}") 
        mlflow.pytorch.log_model(polyp_model, "model")
        
        RUN_ID = run.info.run_id
        
        print("Run ID:", RUN_ID)
        
    
    print("Creating plot loss...")
    path = f"./results/loss_{timestamp}.png"
    plot_loss(train_loss_hist, val_loss_hist, path)

    with mlflow.start_run(run_id=RUN_ID):
        mlflow.log_artifact(path, 'results')
        
    print(Fore.GREEN + f"Plot loss saved at {path}")
        

    # ===================== TESTING ====================================
    print("Evaluating model...")
    metrics, conf_matrix = evaluate_model(polyp_model, model_path, test_loader, train_set, RUN_ID, timestamp)
    print("Finished model evaluation")
    
    print("Registering parameters...")
    df = pd.read_csv('./classifier_model/parameters_register.csv')
    new_row = [model_name] + list(params.values()) + [str(metrics["f1_score"])]
    df.loc[len(df)] = new_row
    df.to_csv('./classifier_model/parameters_register.csv', index=False)
    print(Fore.GREEN + f"Parameters registered at ./classifier_model/parameters_register.csv")


if __name__ == "__main__":
    main()