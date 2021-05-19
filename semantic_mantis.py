# -*- coding: utf-8 -*-
"""
Created on Wed May 19 09:14:21 2021

@author: asus
"""
import argparse
import pandas as pd
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('paraphrase-distilroberta-base-v1')


# required arg
parser = argparse.ArgumentParser()
parser.add_argument("--text", required = True,  help="Please enter your text")
args = parser.parse_args()


#Load the Data
df = pd.read_csv("mantis_copie.csv")

#preprocessing
#keep only tuples we need
df = df[df['id']==0]
df = df.reset_index(drop=True)

df = df[['bug_id','id','note']]

# list of sentences
sentences = df['note']
sentences = sentences.values.tolist()

#Encode all sentences
embeddings = model.encode(sentences)

#sentence 2
embeddings2 = model.encode(args.text)
embeddings2

#Compute cosine similarity between all pairs
cos_sim = util.pytorch_cos_sim(embeddings, embeddings2)
cos_sim

#Add all pairs to a list with their cosine similarity score
all_sentence_combinations = []
for i in range(len(cos_sim)):
      all_sentence_combinations.append([cos_sim[i], i])

all_sentence_combinations

#Sort list by the highest cosine similarity score
all_sentence_combinations = sorted(all_sentence_combinations, key=lambda x: x[0], reverse=True)

print("Top-5 most similar pairs:")
for score, i in all_sentence_combinations[0:5]:
  print(i )
  print(df['bug_id'][i], cos_sim[i])










