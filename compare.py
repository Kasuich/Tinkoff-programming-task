import argparse
import pickle
import os
import numpy as np
import pandas as pd
import catboost as cb
from sklearn.model_selection import train_test_split
from process_code import process_two_directories, process_two_files

class Compare():

    def __init__(self, model_pkl):
        self.model = pickle.load(model_pkl)

    def compare(self, input_file, output_file):

        ans = []

        with open(input_file) as in_file:
            for line in in_file.readlines():
                file1, file2 = line.split()[0], line.split()[1]
                with open(file1) as f:
                    code1 = f.read()
                with open(file2) as f:
                    code2 = f.read()

            df = process_two_files(code1, code2)
            prediction = self.model.predict(df)
            ans.append(prediction)
    
        with open(output_file, 'w') as f:
            for a in ans:
                f.write(a + '\n')


parser = argparse.ArgumentParser(description='Compare Files')
parser.add_argument('input', type=str, help='Input file')
parser.add_argument('output', type=str, help='Output file')
parser.add_argument('--model', type=str, help='Pickle for model')
args = parser.parse_args()

input_path = os.path(args.input)
output_path = os.path(args.output)
model = Compare(args.model)
model.compare(input_path, output_path)
