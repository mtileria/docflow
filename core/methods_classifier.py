from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from torch.utils.data import DataLoader
import math
import pandas as pd

from sklearn import metrics
from sklearn.model_selection import cross_val_predict,train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier


def fine_tunning(model_path,df_path):
    '''
        Runs fine tunning of the Sentence-BERT
        Arguments:
        model_path (str) path to read the pre-trained model
        df (dataframe) dataset of methods pairs
    '''

    model = SentenceTransformer(model_path)
    df = pd.read_csv(df_path,index_col=None)

    train_examples = []
    for _,row in df.iterrows():
        train_examples.append(InputExample(texts=[row['doc1'],row['doc2']],label=row['labels']))

    train_batch_size = 16
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=train_batch_size)
    train_loss = losses.CosineSimilarityLoss(model)

    # example of method pairs for evaluation
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

    
    num_epochs = 10
    model_save_path = '/home/marcos/nlp-documentation/embeddings/fine_tunned_model_10'
    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up

    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=num_epochs,
            evaluation_steps=1000,
            warmup_steps=warmup_steps,
            output_path=model_save_path)


def get_metrics(target,predicted):
    '''
        Calculates metrics for the prediction using the labels (target)
    '''
    accuracy =  metrics.accuracy_score(target, predicted)
    metrics_cv = metrics.classification_report(target, predicted,output_dict=True)
    precision = metrics_cv['macro avg']['precision']
    recall = metrics_cv['macro avg']['recall']
    f1 = metrics_cv['macro avg']['f1-score']
    return accuracy,precision,recall,f1

def classify_methods(embedding,classifier,df):
    '''
        Classify methods using the embedding to calculate the vector representation of methods
        and then classify using the classifier algorithm
    '''
    X_original = embedding.encode(df['docs'].values)
    label_encoder = LabelEncoder()
    label_encoder = label_encoder.fit(df['real'].values)
    label_encoded_y = label_encoder.transform(df['real'].values)
    x_train, x_test, y_train, y_test = train_test_split(X_original,
                                     label_encoded_y, test_size=0.25, random_state=0)
    classifier.fit(x_train,y_train)
    predicted = classifier.predict(x_test)
    accuracy,precision,recall,f1 =  get_metrics(y_test, predicted)
    return accuracy,precision,recall,f1 



if __name__ == '__main__':
    # use a 1-layer neural network as example
    learning = 'NN'
    nn = MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(64), random_state=1,max_iter=500)
    # example of a pre-trained embedding model
    model = SentenceTransformer('all-mpnet-base-v2')
    documents = ['docuement_1','document_2']
    classify_methods(model,nn,documents)