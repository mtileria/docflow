from sentence_transformers import SentenceTransformer, util, InputExample, losses, evaluation
from torch.utils.data import DataLoader
import math
import pandas as pd
model = SentenceTransformer('all-mpnet-base-v2')
df = pd.read_csv('../inputs/pairs_fine_tunning.csv',index_col=None)

train_examples = []
for _,row in df.iterrows():
    train_examples.append(InputExample(texts=[row['doc1'],row['doc2']],label=row['labels']))

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)


# This list can be customised and extended 
sentences1 = ['random access file read line reads the next line of text from this file',
'session get peer host returns the host name of the peer in this session',
'cast session sends a message to the currently connected application',
'channel opens a channel to exchange data with a remote node',
'channel sends a file from output to the side channel']
sentences2 = ['console read line reads a single line of text from the console',
'engine get peer port returns the port number of the peer',
'message client send message sends byte data to the specified node',
'channel closes this channel, passing an application-defined error code to the remote node',
'engine that receives bytes from a buffer into a file']
scores = [0.8, 0.7, 0.6,0.4,0.2]
evaluator = evaluation.EmbeddingSimilarityEvaluator(sentences1, sentences2, scores)

# select the parameters as needed. By default, the params provided here generate good results

train_batch_size = 16
num_epochs = 10  # or 4 tested
model_save_path = '../embeddings/fine_tunned'
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up

# Fine-tunning the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=model_save_path)