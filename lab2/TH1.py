import numpy as np
from sklearn.naive_bayes import GaussianNB
import pandas as pd
def sol():
  df = pd.read_csv("Data/Education.csv")
  NB = GaussianNB()
  print(df)
if __name__ == "__main__":
  sol()