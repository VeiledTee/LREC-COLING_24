import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.svm import SVC
from transformers import BertTokenizer, BertModel

from clean_data import label_mapping, encode_sentence
from variables import DEVICE

if __name__ == "__main__":
    train_df = label_mapping(
        df=pd.read_csv("Data/SICK/train_cleaned_ph.csv"),
        from_col="gold_label",
        to_col="label",
    )
    valid_df = label_mapping(pd.read_csv("Data/SICK/valid_cleaned_ph.csv"))
    test_df = pd.read_csv("Data/SICK/test_cleaned_ph.csv")

    for b in [True, False]:
        acc = []
        f1 = []
        precision = []
        recall = []
        for i in range(1):
            print(b, i)
            if b:
                bbu_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
                bbu_model = BertModel.from_pretrained("bert-base-uncased").to(DEVICE)

                dataset_descriptors: list = ["match", "mismatch"]
                dataframes: list = []

                pair_x = [str(s).strip() for s in train_df["sentence1"]]
                pair_y = [str(s).strip() for s in train_df["sentence2"]]
                X_train = np.array(
                    [
                        encode_sentence(
                            bbu_model, bbu_tokenizer, f"{x} [SEP] {y}", DEVICE
                        )
                        for x, y in zip(pair_x, pair_y)
                    ]
                )
                y_train = train_df["gold_label"].tolist()
                pair_x = [str(s).strip() for s in valid_df["sentence1"]]
                pair_y = [str(s).strip() for s in valid_df["sentence2"]]
                X_val = np.array(
                    [
                        encode_sentence(
                            bbu_model, bbu_tokenizer, f"{x} [SEP] {y}", DEVICE
                        )
                        for x, y in zip(pair_x, pair_y)
                    ]
                )
                y_val = valid_df["gold_label"].tolist()
                pair_x = [str(s).strip() for s in test_df["sentence1"]]
                pair_y = [str(s).strip() for s in test_df["sentence2"]]
                X_test = np.array(
                    [
                        encode_sentence(
                            bbu_model, bbu_tokenizer, f"{x} [SEP] {y}", DEVICE
                        )
                        for x, y in zip(pair_x, pair_y)
                    ]
                )
                y_test = test_df["gold_label"].tolist()
                # Initialize lists to store results
                validation_accuracies = []
                validation_precisions = []
                validation_recalls = []
                validation_f1_scores = []
                test_accuracies = []

                final_clf = SVC(kernel="linear", C=0.1)

                # Train the final model on the entire training dataset
                final_clf.fit(X_train, y_train)
                y_val_pred = final_clf.predict(X_val)
                # Calculate evaluation metrics
                validation_accuracy = accuracy_score(y_val, y_val_pred)
                validation_precision = precision_score(
                    y_val, y_val_pred, average="weighted"
                )
                validation_recall = recall_score(y_val, y_val_pred, average="weighted")
                validation_f1 = f1_score(y_val, y_val_pred, average="weighted")
                # Evaluate the final model on the test set (unseen data)
                y_test_pred = final_clf.predict(X_test)
                # Count unique values and their counts
                unique_values, counts = np.unique(y_test_pred, return_counts=True)
                # Print unique values and their counts
                for value, count in zip(unique_values, counts):
                    print(f"Class {value}: {count} predictions")
                # Calculate test set evaluation metrics
                test_accuracy = accuracy_score(y_test, y_test_pred)
                test_precision = precision_score(
                    y_test, y_test_pred, average="weighted"
                )
                test_recall = recall_score(y_test, y_test_pred, average="weighted")
                test_f1 = f1_score(y_test, y_test_pred, average="weighted")

                acc.append(test_accuracy)
                f1.append(test_f1)
                precision.append(test_precision)
                recall.append(test_recall)
                print(
                    f"{100 * sum(acc) / len(acc):.2f}% | F1: {sum(f1) / len(f1):.4f} | "
                    f"P: {sum(precision) / len(precision):.4f} | R: {sum(recall) / len(recall):.4f}"
                )
            else:
                # Load the Sentence-BERT model
                model = SentenceTransformer("all-MiniLM-L6-v2")

                dataset_descriptors: list = ["match", "mismatch"]
                dataframes: list = []

                pair_x = [str(s).strip() for s in train_df["sentence1"]]
                pair_y = [str(s).strip() for s in train_df["sentence2"]]
                X_train = model.encode([(x, y) for x, y in zip(pair_x, pair_y)])
                y_train = train_df["gold_label"].tolist()
                pair_x = [str(s).strip() for s in valid_df["sentence1"]]
                pair_y = [str(s).strip() for s in valid_df["sentence2"]]
                X_val = model.encode([(x, y) for x, y in zip(pair_x, pair_y)])
                y_val = valid_df["gold_label"].tolist()
                pair_x = [str(s).strip() for s in test_df["sentence1"]]
                pair_y = [str(s).strip() for s in test_df["sentence2"]]
                X_test = model.encode([(x, y) for x, y in zip(pair_x, pair_y)])
                y_test = test_df["gold_label"].tolist()
                # print(f"Percent Positive: {100 * sum([1 if int(i) == 2 else 0 for i in y_test]) / len(y_test):.4f}%")

                final_clf = SVC(kernel="linear", C=1)
                # Train the final model on the entire training dataset
                final_clf.fit(X_train, y_train)
                y_val_pred = final_clf.predict(X_val)
                # Calculate evaluation metrics
                validation_accuracy = accuracy_score(y_val, y_val_pred)
                validation_precision = precision_score(
                    y_val, y_val_pred, average="weighted"
                )
                validation_recall = recall_score(y_val, y_val_pred, average="weighted")
                validation_f1 = f1_score(y_val, y_val_pred, average="weighted")

                # Evaluate the final model on the test set (unseen data)
                y_test_pred = final_clf.predict(X_test)
                # Count unique values and their counts
                unique_values, counts = np.unique(y_test_pred, return_counts=True)
                # Print unique values and their counts
                for value, count in zip(unique_values, counts):
                    print(f"Class {value}: {count} predictions")
                # Calculate test set evaluation metrics
                test_accuracy = accuracy_score(y_test, y_test_pred)
                test_precision = precision_score(
                    y_test, y_test_pred, average="weighted"
                )
                test_recall = recall_score(y_test, y_test_pred, average="weighted")
                test_f1 = f1_score(y_test, y_test_pred, average="weighted")

                acc.append(test_accuracy)
                f1.append(test_f1)
                precision.append(test_precision)
                recall.append(test_recall)
                print(
                    f"{100 * sum(acc) / len(acc):.2f}% | F1: {sum(f1) / len(f1):.4f} | "
                    f"P: {sum(precision) / len(precision):.4f} | R: {sum(recall) / len(recall):.4f}"
                )
