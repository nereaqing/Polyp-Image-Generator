import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import argparse
import json

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from PolypClassificationModel import PolypClassificationModel
from AugmentedPolypDataset import AugmentedPolypClassificationDataset
from config_classification import ConfigClassification

import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")

assert torch.cuda.is_available(), "GPU is not enabled"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def preprocess_files(image_size, path_model_to_test, ad_vs_rest=False):
    if ad_vs_rest:
        train_set = AugmentedPolypClassificationDataset(dirs=[
                                                            ("../data/m_train2/m_train/images", "../data/m_train2/m_train/train.csv"),
                                                            (f"{path_model_to_test}/samples/AD", None),
                                                            (f"{path_model_to_test}/samples/REST", None),
                                                        ],
                                                        image_size=image_size,
                                                        transformations=True,
                                                        ad_vs_rest=ad_vs_rest
        )
    else:
        train_set = AugmentedPolypClassificationDataset(dirs=[
                                                            ("../data/m_train2/m_train/images", "../data/m_train2/m_train/train.csv"),
                                                            (f"{path_model_to_test}/samples/AD", None),
                                                            (f"{path_model_to_test}/samples/HP", None),
                                                            (f"{path_model_to_test}/samples/ASS", None),
                                                        ],
                                                        image_size=image_size,
                                                        transformations=True,
                                                        ad_vs_rest=ad_vs_rest
        )
                                                            


    val_set = AugmentedPolypClassificationDataset(dirs=[
                                                        ("../data/m_valid/m_valid/images", "../data/m_valid/m_valid/valid.csv")
                                                ],
                                                image_size=image_size,
                                                transformations=True,
                                                ad_vs_rest=ad_vs_rest
        )
        
        
    test_set = AugmentedPolypClassificationDataset(dirs=[
                                                        ("../data/m_test/m_test/images", "../data/m_test/m_test/gt_test.csv")
                                                ],
                                                image_size=image_size,
                                                transformations=True,
                                                ad_vs_rest=ad_vs_rest
    )
    
    return train_set, val_set, test_set


def get_class_weights(dataset):
    classes = np.unique(dataset.labels)  # or [0, 1, 2]
    print(classes)
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

        print(f"\nEpoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, Accuracy: {train_acc:.4f}")

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
            print(f"Early stopping: {early_stopping}")

        if early_stopping == n_early_stopping:
            print(f"Stopping early at epoch {epoch+1} with best loss of {best_val_loss}.\nBest model: {path_model}")
            break

        if scheduler:
            scheduler.step()

    print("Training completed successfully!")
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
    print(f"Loaded {os.path.basename(path_model)}")
    
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
    
    output_path = os.path.dirname(path_model)
    
    
    with mlflow.start_run(run_id=run_id):
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)
        recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=0)
        f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=0)
        
        all_metrics = classification_report(true_labels, predicted_labels, labels=sorted(list(set(true_labels))), output_dict=True)
        df_metrics = pd.DataFrame(all_metrics).transpose()
        path_report = os.path.join(output_path, f"metrics_report_{timestamp}.csv")
        df_metrics.to_csv(path_report)
        mlflow.log_artifact(path_report, "classifier/results")
        
        # Confusion Matrix
        conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=sorted(list(set(true_labels))))
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=sorted(list(set(true_labels))), yticklabels=sorted(list(set(true_labels))))
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        path_cm = os.path.join(output_path, f"confusion_matrix_{timestamp}.png")
        plt.savefig(path_cm)
        plt.show()
        mlflow.log_artifact(path_cm, "classifier/results")

        # Print metrics
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        mlflow.log_metric("f1_score_classifier", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("accuracy", accuracy)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--path_model", type=str, required=True)
    parser.add_argument("--ad_vs_rest", action='store_true')
    
    args = parser.parse_args()
    config = ConfigClassification()
    
    techniques = []
    path_classifier = f"{args.path_model}/classifier"
    os.makedirs(path_classifier, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"classifier_{timestamp}.pth"
    model_path = os.path.join(path_classifier, model_name)
    if os.path.exists(os.path.dirname(model_path)):
        print(True)
        print(model_path)
    else:
        print(False)
    
    
    EXPERIMENT_NAME = args.experiment_name
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # ================ Construct dataset ==================
    print("Constructing datasets...")
    augmented_train_set, val_set, test_set = preprocess_files(config.image_size, args.path_model, args.ad_vs_rest)
    print("Transformations to apply:", augmented_train_set.transformations_list)
    print("Datasets created")

    # ========= Construct dataloader ========================
    print("Constructing dataloaders...")
    augmented_train_loader = DataLoader(augmented_train_set, batch_size=config.batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, num_workers=4)
    
    if config.weighted_sampling:
        _, class_weights_dict = get_class_weights(augmented_train_set)
        sample_weights = [class_weights_dict[label] for label in augmented_train_set.labels]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

        augmented_train_loader = DataLoader(augmented_train_set, batch_size=config.batch_size, sampler=sampler, num_workers=4, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=config.batch_size, num_workers=4)
        test_loader = DataLoader(test_set, batch_size=config.batch_size, num_workers=4)
        techniques.append("weighted sampling")
        
        
    print("Dataloaders created")
     
    # ====================== Set hyperparameters ==========================
    print("Setting hyperparameters...")
    learning_rate = config.learning_rate
    num_epochs = config.num_epochs
    early_stopping = config.patience
    weight_decay = config.weight_decay
    hidden_features = config.hidden_features
    dropout = config.dropout

    polyp_model = PolypClassificationModel(num_classes=len(augmented_train_set.dic_label2idx), dropout=dropout, hidden_features=hidden_features)
    if config.weighted_loss:
        class_weights_tensor, _ = get_class_weights(augmented_train_set)
        class_weights_tensor = class_weights_tensor.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        techniques.append("weighted loss")
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(polyp_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    if len(techniques) == 0:
        params = {
            "transformations": augmented_train_set.transformations_list,
            "image_size": config.image_size,
            "criterion": "CrossEntropy",
            "optimizer": "Adam",
            "hidden_features": hidden_features,
            "batch_size": config.batch_size,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "num_epohcs": num_epochs,
            "early_stopping": early_stopping,
            "other_techniques": "None"
        }
    else:
        params = {
            "transformations": augmented_train_set.transformations_list,
            "image_size": config.image_size,
            "criterion": "CrossEntropy",
            "optimizer": "Adam",
            "hidden_features": hidden_features,
            "batch_size": config.batch_size,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "num_epohcs": num_epochs,
            "early_stopping": early_stopping,
            "other_techniques": techniques
        }

    path_params = f"./{path_classifier}/params.json"
    with open(path_params, "w") as f:
        json.dump(params, f, indent=2)
    
    print(params)
    print("Hyperparameters set")
        
    # =============== TRAINING ===============================
    with mlflow.start_run(run_id=args.run_id) as run:
        mlflow.log_artifact(path_params, artifact_path="classifier")
        
        print("Training...")
        train_loss_hist, val_loss_hist, _ = train(num_epochs, 
                                                polyp_model, 
                                                augmented_train_loader, 
                                                val_loader, 
                                                optimizer, 
                                                criterion,
                                                early_stopping,
                                                path_model=model_path)
        
        mlflow.pytorch.log_model(polyp_model, "classifier/model")
    
    print("Creating plot loss...")
    path = os.path.join(path_classifier, f"loss_{timestamp}.png")
    plot_loss(train_loss_hist, val_loss_hist, path)

    with mlflow.start_run(run_id=args.run_id):
        mlflow.log_artifact(path, 'classifier/results')
        
    print(f"Plot loss saved at {path}")
        

    # ===================== TESTING ====================================
    print("Evaluating model...")
    evaluate_model(polyp_model, model_path, test_loader, augmented_train_set, args.run_id, timestamp)
    print("Finished model evaluation")

if __name__ == "__main__":
    main()