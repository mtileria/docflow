from transformers import pipeline

def read_labels():
    labels = ['logging', 'nfc communication','nfc information', 'audio information', 'user account', 'user information','text message',
       'file information', 'network connection','network information' ,'geographical location', 'bluetooth','system settings','calendar information',
       'text message', 'contact information','media','media information','database','database information','notification']
    return labels

def classify(model,text,labels):
    out = model(text, labels)
    return((out['labels'][0],out['scores'][0]))



if __name__ == '__main__':
    # use the facebook/bart-large-mnli as example
    learning = 'zero-shot'
    classifier = pipeline("zero-shot-classification",
                        model="facebook/bart-large-mnli")
    documents = ["this class provides access to the system location services these services allow applications to obtain periodic updates of the device's geographical location, or to be notified when the device enters the proximity of a given geographical location"]
    labels = read_labels()
    for doc in documents:
        label,score = classify(classifier,doc,labels)