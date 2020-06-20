import pandas as pd
import tensorflow as tf
import os
import re
import csv

# Load all files from a directory in a DataFrame.
def load_directory_data(directory):
    data = {}
    data["sentence"] = []
    # data['sentiment'] = []
    data["polarity"] = []

    for file_path in os.listdir(directory):
        with tf.gfile.GFile(os.path.join(directory, file_path), "r") as f:
            # txt
            # data["sentence"].append(f.read())
            # csv
            reader = csv.reader(f, delimiter=",")
            for line in reader:
                data["sentence"].append(line[0])
                # data['sentiment'].append(line[2])
                data['polarity'].append(int(line[1]))
            # data["sentiment"].append(re.match("(\w+)\.csv", file_path).group(1))
    return pd.DataFrame.from_dict(data)


# Merge positive and negative examples, add a polarity column and shuffle.
def load_dataset(directory):
  sample = load_directory_data(directory)
  # print(pos_df)
  # neg_df = load_directory_data(directory)
  # print(neg_df)
  # pos_df["polarity"] = 1
  # neg_df["polarity"] = 0
  return pd.DataFrame(sample).sample(frac=1).reset_index(drop=True)


# Download and process the dataset files.
def download_and_load_datasets(force_download=False):
    # dataset = tf.keras.utils.get_file(
    #     fname="aclImdb.tar.gz",
    #     origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
    #     extract=True)
    dataset = "D:/project/bert_use-master/data/"
    train_df = load_dataset(os.path.join(os.path.dirname(dataset),"train"))
    test_df = load_dataset(os.path.join(os.path.dirname(dataset),"test"))
    # train_df = load_dataset(os.path.join(os.path.dirname(dataset), "aclImdb", "train"))
    # test_df = load_dataset(os.path.join(os.path.dirname(dataset), "aclImdb", "test"))

    return train_df,test_df



if __name__ == "__main__":
    train,test = download_and_load_datasets()
    train = train.sample(20000)
    print(train)
    test = test.sample(5000)
