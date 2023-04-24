import pandas as pd
import numpy as np
import config
import os
import clean_text
import argparse
import pickle


def process_data(data_path, save_dir):
    """The function is to process and clean the raw text and split into train, test, validate

    Args:
        data_path (str): The path to comments dataset
        save_dir (str): The path to store preprocessed datasets along the with spliting
    """

    data = pd.read_csv(data_path, index_col=0)

    ## Extracting the relevant columns
    data = data[config.columnns]

    ## Binariziation of target column
    data["target_label"] = np.where(data["target"] >= 0.5, 1, 0)

    ## Droping target column as it's not needed
    data.drop(columns="target", inplace=True)

    ## Clean the comments
    data["comment_text"] = data["comment_text"].apply(
        lambda x: clean_text.preprocess_text(x)
    )
    print(
        "All the text is cleaned"
    )  ## As a logging step. In future, we can add logger here

    ## Splitting the text into train test validate split
    train, validate, test = np.split(
        data.sample(frac=1),
        [
            int(config.train_size * len(data)),
            int((config.train_size + config.test_size) * len(data)),
        ],
    )

    ## Generating Pandas dataframe
    df_train = pd.DataFrame(train)
    df_validate = pd.DataFrame(validate)
    df_test = pd.DataFrame(test)

    ## Saving data frame into csv
    df_train.to_csv(os.path.join(save_dir, "train_split.csv"))
    df_validate.to_csv(os.path.join(save_dir, "validate.csv"))
    df_test.to_csv(os.path.join(save_dir, "test.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Path of data file
    parser.add_argument("--data_path", type=str, required=True)

    ## Path for the save directory
    parser.add_argument("--save_dir", type=str, required=True)

    args = parser.parse_args()

    process_data(args.data_path, args.save_dir)
    print(f"Data Saved to {args.save_dir}")
