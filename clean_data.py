from typing import List, Optional

import pandas as pd
import numpy as np
from ripser import ripser
from transformers import AutoModel, AutoTokenizer, BertTokenizer

import torch
import concurrent.futures

from tqdm import tqdm
import csv

from variables import DEVICE


def load_txt_file_to_dataframe(dataset_description: str) -> pd.DataFrame:
    """
    Load MultiNLI data into dataframe for use
    :param dataset_description: Which dataset to load and work with
    :return: Cleaned data contained in a dataframe
    """
    to_drop: list = [
        "label1",
        "sentence1_binary_parse",
        "sentence2_binary_parse",
        "sentence1_parse",
        "sentence2_parse",
        "promptID",
        "pairID",
        "genre",
        "label2",
        "label3",
        "label4",
        "label5",
    ]
    if dataset_description.lower().strip() == "train":
        data_frame = pd.read_csv(
            "Data/MultiNLI/multinli_1.0_train.txt", sep="\t", encoding="latin-1"
        ).drop(columns=to_drop)
    elif dataset_description.lower().strip() == "match":
        data_frame = pd.read_csv(
            "Data/MultiNLI/multinli_1.0_dev_matched.txt", sep="\t"
        ).drop(columns=to_drop)
    elif dataset_description.lower().strip() == "mismatch":
        data_frame = pd.read_csv(
            "Data/MultiNLI/multinli_1.0_dev_mismatched.txt", sep="\t"
        ).drop(columns=to_drop)
    else:
        raise ValueError("Pass only 'train', 'match', or 'mismatch' to the function")

    data_frame.dropna(inplace=True)
    return data_frame


def top_k_holes(
    ph_diagrams: List[np.ndarray], k: Optional[List[int]] = None
) -> List[np.ndarray]:
    """
    Find the k holes in each dimension that persist the longest and return their information
    :param ph_diagrams: Diagrams generated using the "ripser" library
    :param k: A list of k values to use in each dimension.
    :return: List of top k holes in each dimension
    """
    if k is None:
        k = [260, 50]
    if len(k) > len(ph_diagrams):
        print(
            Warning(
                f"You provided more k values than dimensions. There are {len(ph_diagrams)} dimensions but {len(k)} k values. Only the first {len(ph_diagrams)} k values will be used"
            )
        )
    elif len(k) < len(ph_diagrams):
        raise ValueError(
            f"Less k values than there are dimensions. You provided {len(k)} k values for {len(ph_diagrams)} dimensions. Ensure there is a k value for every dimension."
        )

    top_holes: List[np.ndarray] = []
    # Iterate over each dimension
    for dimension, diagram_array in enumerate(ph_diagrams):
        # Initialize an empty list to store the hole indices and their persistence values
        holes = []
        # Iterate over each feature in the diagram
        for j in range(diagram_array.shape[0]):
            feature_birth = diagram_array[j, 0]
            feature_death = diagram_array[j, 1]
            persistence = feature_death - feature_birth
            holes.append(np.array([persistence]))

        # Sort the holes based on their persistence values in descending order
        holes.sort(key=lambda x: x[2], reverse=True)

        # Select the top k holes and add to list
        top_holes.append(np.array(holes[: k[dimension]]))

    # return list of top k holes in each dimension
    return top_holes


def count_negations(sentences: List[str]) -> int:
    negation_count: int = 0
    for sentence in sentences:
        # Load the BERT tokenizer
        tokenizer: BertTokenizer = AutoModel.from_pretrained("bert-base-uncased")
        # Count the negation words
        negation_words: List[str] = [
            "not",
            "no",
            "never",
            "none",
            "nobody",
            "nowhere",
            "nothing",
            "neither",
            "nor",
        ]
        # Preprocess contractions
        sentence: str = sentence.replace("n't", " not")
        sentence: str = sentence.replace("'re", " are")
        sentence: str = sentence.replace("'ll", " will")
        # tokenize
        words: tokenizer = tokenizer.tokenize(sentence)
        # Count the negation words
        negation_count += sum(1 for word in words if word.lower() in negation_words)

    return negation_count


def get_sentence_embedding(model_name: str, sentence: str) -> torch.Tensor:
    # Load pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(DEVICE)

    # Tokenize the input sentence
    tokens = tokenizer(sentence, padding=True, truncation=True, return_tensors="pt")
    tokens = {key: value.to(DEVICE) for key, value in tokens.items()}

    # Get the model output
    with torch.no_grad():
        outputs = model(**tokens)

    # Get the representation of [CLS] token (sentence embedding)
    sentence_embedding = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]

    # Move the sentence embedding tensor to CPU before returning
    return sentence_embedding.squeeze().cpu()


def persistent_homology_features(
    model_name: str, phrases: List[str]
) -> List[List[np.ndarray]]:
    features: List[List[np.ndarray]] = []

    def process_phrase(sentence: str) -> List[np.ndarray]:
        embedding = [get_sentence_embedding(model_name, s) for s in sentence]
        all_embeddings = np.array(embedding).T
        ph = ripser(all_embeddings, maxdim=1)
        ph_diagrams = ph["dgms"]
        phrase_holes = top_k_holes(ph_diagrams)
        return phrase_holes

    with concurrent.futures.ThreadPoolExecutor() as executor:
        phrase_futures = [
            executor.submit(process_phrase, sentence) for sentence in phrases
        ]

        for future in concurrent.futures.as_completed(phrase_futures):
            phrase_holes: List[np.ndarray] = future.result()
            features.append(phrase_holes)

    return features


def label_mapping(
    df: pd.DataFrame,
    from_col: str = "gold_label",
    to_col: str = "label",
    str_to_int: bool = True,
) -> pd.DataFrame:
    if str_to_int:
        mapping = {
            "neutral": 0,
            "entailment": 1,
            "contradiction": 2,
        }
    else:
        mapping = {0: "neutral", 1: "entailment", 2: "contradiction"}
    df[to_col] = df[from_col].map(mapping)
    return df


def encode_sentence(language_model, tokenizer, sentence, device):
    tokens = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True).to(
        device
    )
    with torch.no_grad():
        outputs = language_model(**tokens)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()


def create_subset_with_ratio(input_df, subset_percentage, label_column):
    # Calculate the size of the desired subset
    total_count = input_df[label_column].count()
    subset_size = int(subset_percentage * total_count)

    # Create an empty DataFrame to store the subset
    subset_df = pd.DataFrame(columns=input_df.columns)

    # Iterate through the unique labels and sample data based on the proportion of the original count
    unique_labels = input_df[label_column].unique()
    for label in unique_labels:
        if label != "-":
            label_subset_count = int(
                (input_df[label_column] == label).sum() * subset_percentage
            )
            label_subset = input_df[input_df[label_column] == label].sample(
                label_subset_count
            )
            subset_df = pd.concat([subset_df, label_subset], ignore_index=True)

    # Shuffle the subset DataFrame to randomize the order
    subset_df = subset_df.sample(frac=1).reset_index(drop=True)

    return subset_df


def embed_and_ph(df_for_cleaning: pd.DataFrame, output_csv_path: str) -> None:
    """
    Given a dataframe, retrieve the data's BERT embeddings and PH features for dimensions 0 and 1, then save it. If the
    output file already exists, pick up where it left off
    :param df_for_cleaning: df containing data
    :param output_csv_path: The path to the output file
    :return: None
    """
    # Check if the output CSV file already exists
    try:
        with open(output_csv_path, "r") as csvfile:
            reader = csv.reader(csvfile)
            # Count the number of rows in the existing CSV file
            existing_records = sum(1 for _ in reader) - 1  # Subtract 1 for the header
    except FileNotFoundError:
        existing_records = 0

    print(f"\tWriting to:  {output_csv_path}")
    print(f"Starting at Record: {existing_records + 1}")

    for i, row in tqdm(df_for_cleaning.iterrows(), total=len(df_for_cleaning)):
        if i >= existing_records:
            if row["gold_label"].lower() == "contradiction":
                label = 2
            elif row["gold_label"].lower() == "entailment":
                label = 1
            else:
                label = 0

            # Apply the get_sentence_embedding function to generate embeddings
            if "sentence1_embeddings" not in row:
                row["sentence1_embeddings"] = get_sentence_embedding('bert-base-uncased', row["sentence1"])

            if "sentence2_embeddings" not in row:
                row["sentence2_embeddings"] = get_sentence_embedding('bert-base-uncased', row["sentence2"])

            # Calculate negation count
            if "negation" not in row:
                row["negation"] = count_negations(
                    [str(row["sentence1"]).strip(), str(row["sentence2"]).strip()]
                )

            if output_csv_path[-6:-4] == "ph":
                if "sentence1_ph_a" not in row:
                    s1_ph_features = persistent_homology_features('bert-base-uncased',
                        [row["sentence1"].strip()]
                    )
                    row["sentence1_ph_a"] = s1_ph_features[0][0]
                    row["sentence1_ph_b"] = s1_ph_features[0][1]

                if "sentence2_ph_a" not in row:
                    s2_ph_features = persistent_homology_features('bert-base-uncased',
                        [row["sentence2"].strip()]
                    )
                    row["sentence2_ph_a"] = s2_ph_features[0][0]
                    row["sentence2_ph_b"] = s2_ph_features[0][1]

                # Create a new row for the result DataFrame
                result_row = {
                    "gold_label": row["gold_label"].lower().strip(),
                    "sentence1": row["sentence1"].strip(),
                    "sentence2": row["sentence2"].strip(),
                    "label": label,
                    "sentence1_embeddings": row["sentence1_embeddings"],
                    "sentence2_embeddings": row["sentence2_embeddings"],
                    "sentence1_ph_a": row["sentence1_ph_a"],
                    "sentence1_ph_b": row["sentence1_ph_b"],
                    "sentence2_ph_a": row["sentence2_ph_a"],
                    "sentence2_ph_b": row["sentence2_ph_b"],
                    "negation": row["negation"],
                }
            else:
                # Create a new row for the result DataFrame
                result_row = {
                    "gold_label": row["gold_label"].strip(),
                    "sentence1": str(row["sentence1"]).strip(),
                    "sentence2": str(row["sentence2"]).strip(),
                    "label": label,
                    # "sentence1_embeddings": row["sentence1_embeddings"],
                    # "sentence2_embeddings": row["sentence2_embeddings"],
                    "negation": row["negation"],
                }

            with open(output_csv_path, "a", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=result_row.keys())
                if i == 0:
                    writer.writeheader()  # Write the header only for the first row
                try:
                    writer.writerow(result_row)
                except UnicodeError:
                    print(row["sentence1"])
                    print(row["sentence2"])
                    raise UnicodeError


def ph_to_tensor(array_str):
    # Remove newline characters and strip brackets
    array_str = array_str.replace("\n", "").strip("[]")

    # Split rows and convert to a list of lists of floats
    rows = [list(map(float, row.split())) for row in array_str.split("] [")]

    # Check for 'inf' values and replace them
    for row in rows:
        for i in range(len(row)):
            if np.isinf(row[i]):
                row[i] = np.finfo(
                    np.float32
                ).max  # Replace 'inf' with a large finite value

    # Convert the list of lists to a NumPy array
    array = np.array(rows)

    # Convert the NumPy array to a PyTorch tensor
    tensor = torch.tensor(array, dtype=torch.float32)

    return tensor
