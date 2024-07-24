
### split dataframe: train and val

import pandas as pd

def manual_split_training_data():

    train_df = pd.read_csv("training.csv")
    # test_df = pd.read_csv("facial-keypoints-detection/test.csv")

    train_15_landmarks = train_df[~(train_df.isnull().sum(axis=1) > 20)].copy()
    train_4_landmarks = train_df[(train_df.isnull().sum(axis=1) > 20)].copy()

    # train_15_landmarks.shape, train_4_landmarks.shape

    train_15_landmarks.iloc[0:100, :].to_csv("val_15.csv", index=False)
    train_15_landmarks.iloc[100:, :].to_csv("train_15.csv", index=False)

    train_4_landmarks.iloc[0:200, :].to_csv("val_4.csv", index=False)
    train_4_landmarks.iloc[200:, :].to_csv("train_4.csv", index=False)
