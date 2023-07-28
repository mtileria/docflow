import numpy as np
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import gensim.downloader as api
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from gensim.parsing.preprocessing import remove_stopwords, preprocess_string, strip_punctuation, stem_text, STOPWORDS
from sklearn.decomposition import IncrementalPCA    # inital reduction
from sklearn.manifold import TSNE
from nltk.corpus import words
import json
import re
from re import split
import os
from sklearn import metrics
import pandas as pd


tmp_labels = ['logging', 'nfc communication', 'nfc information', 'audio information', 'user account', 'user information', 'text message',
              'file information', 'network connection', 'network information', 'geographical location', 'bluetooth', 'system settings', 'calendar information',
              'text message', 'contact information', 'media', 'media information', 'database', 'database information', 'notification']
inverse_map_pred = {'logging': 'log', 'nfc communication': 'nfc', 'nfc information': 'nfc', 'audio information': 'audio',
                    'user account': 'user account', 'user information': 'user account', 'text message': 'text message',
                    'file information': 'files', 'network connection': 'network', 'geographical location': 'geolocation', 'bluetooth': 'bluetooth',
                    'system settings': 'system settings',  'calendar information': 'calendar',
                    'text message': 'text message', 'contact information': 'contact information', 'media': 'media', 'media information': 'media',
                    'network information': 'network', 'database': 'database', 'database information': 'database', 'notification': 'notification'}
inverse_map_real = {'calendar information': 'calendar',
                    'network connection': 'network', 'network information': 'network',
                    'nfc communication': 'nfc', 'nfc information': 'nfc',
                    'text message': 'text message',  'notification': 'notification',
                    'files': 'files',  'geolocation': 'geolocation',
                    'calendar': 'calendar',  'system settings': 'system settings',
                    'media': 'media',  'bluetooth': 'bluetooth',
                    'database': 'database',  'contact information': 'contact information',
                    'user account': 'user account',  'log': 'log',  'audio': 'audio'
                    }
inverse_map_susi = {'calendar information': 'calendar', 'network connection': 'network', 'network information': 'network',
                    'nfc communication': 'nfc', 'nfc information': 'nfc', 'no category': 'no category'}


def reduce_dimensions_tsne(model):
    num_dimensions = 2
    # extract the words & their vectors, as numpy arrays
    vectors = np.asarray(model.wv.vectors)
    labels = np.asarray(model.wv.index_to_key)  # fixed-width numpy strings
    # reduce using t-SNE
    tsne = TSNE(n_components=num_dimensions, random_state=0)
    vectors = tsne.fit_transform(vectors)
    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]

    return x_vals, y_vals, labels


def save_model(model_name, path_emb):
    model = api.load(model_name)
    model.save(path_emb+model_name)


def load_model(model_name, path_emb):
    return KeyedVectors.load(path_emb + model_name, mmap='r')


def get_file_number(path):
    num = 0
    while os.path.isfile(path + '_' + str(num)) == True:
        num = num + 1
    return '_' + str(num)


def save_clustering_results(emb_name, algo, params, cluster_values, num=0, options=''):
    path = '/path/' + emb_name + '_' + num + '.json'
    if os.path.isfile(path):
        print('File exists, try different name')
        return
    output = {'algo': algo, 'params': params,
              'options': options, 'cluster': cluster_values}
    with open(path, "w") as outfile:
        json.dump(output, outfile, indent=4)


def get_tokens_from_name(signature):
    name = signature.split('(')[0]
    # camelCase method names regex
    matches = re.finditer(
        '.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', name)
    tokens = ([m.group(0).lower() for m in matches])
    return [token for token in tokens if filter_token(token)]


def filter_token(token):
    if token in STOPWORDS or len(token) < 2:
        return False
    return True


def get_tokens_for_frequency(description):
    CUSTOM_FILTERS = [remove_stopwords, strip_punctuation]
    clean_text = preprocess_string(description, CUSTOM_FILTERS)
    return clean_text


def remove_key_words(description, keywords, en_dict):
    '''Remove class and method names from description if word not in english dicionary'''
    desc_final = []
    if len(description[0]) > 1:
        desc_final.append(description[0].lower())
    for item in description[1:]:
        if len(item) in range(2, 22):
            if not any(ele.isupper() for ele in item):
                desc_final.append(item)
            elif item.lower() in en_dict:
                desc_final.append(item.lower())
            elif item not in keywords:
                desc_final.append(item.lower())
    return desc_final


def get_description_tokens(description, keywords):
    '''First remove stop words and punctuation. The remove keywords and short words.
        Return a lost of tokens
    '''
    CUSTOM_FILTERS = [remove_stopwords, strip_punctuation]
    clean_text = preprocess_string(description, CUSTOM_FILTERS)
    description_filtered = remove_key_words(clean_text, keywords)
    return description_filtered


def get_list_names_descriptions(json_data):
    text = []  # [method name,description]
    for lib_dict in json_data:
        for clazz in lib_dict['classes']:
            for method in clazz['methods']:
                # if len(method['description']) > 0:
                text.append([method['name'], method['description']])
    return text


def get_type_token(text, type='list'):
    '''removes undesirable characters and returns lower cased text'''
    if text.startswith(' '):
        text = text.split(' ')[-1]
    text = get_return_type(text)
    return text.lower()


def get_return_type(text):
    '''Remove modifiers and parametrized types, and split camel cased text'''
    text = text.replace('static', '').replace('abstract', '')
    if '<' in text and '>' in text:
        text = re.sub('<.*?>', '', text).strip()
    if any(x.isupper() for x in text[1:]) and not all(x.isupper() for x in text):
        text_tokens = split(r'(?=[A-Z])', text)
        text = ' '.join(text_tokens[1:])
    return text


def get_params_token(text, type='list'):
    '''returns a lower cased description of the parameters'''
    text = text.replace('.', '_')
    txt_sp = text.split('_')
    return [get_type_token(x, type) for x in txt_sp]


def lower_init(text):
    if len(text) > 0:
        return text[0].lower() + text[1:]
    else:
        return text


def clean_desc(text):
    '''removes everything between brackets'''
    if all(x in text for x in ['(', ')']):
        text = re.sub(r"\([^()]*\)", "", text)
    elif '(' in text:
        i = text.index('(')
        text = text[:i - 1]
    return text


def split_description(text):
    '''split camel cased words in a description'''
    text = text.replace('.', ' ')
    tokens = text[:-1].split(' ')
    out = []
    for token in tokens:
        if len(token) > 1 and not all(x.isupper() for x in token) and any(x.isupper() for x in token):
            tmp = split(r'(?=[A-Z])', token)
            tmp = [x for x in tmp if x != '']
            out.extend(tmp)
        else:
            out.append(token)
    return ' '.join(out)


def generate_docs_bert_A_verbs(df, verbs_list=['no_verbs']):
    ''' Format: description verbs weight'''
    docs = []
    keys = []
    classification = []
    for i, row in df.iterrows():
        verbs = verbs_list[i]
        if verbs[0] != 'no_verbs':
            if len(row['description']) > 0:
                text = row['description']
                description = text if '(' not in text else clean_desc(text)
                tmp = lower_init(description)
                tmp = split_description(description)
                docs.append(tmp)
                classification.append(row['classification'])
                keys.append('_'.join([row['class'], row['method_name']]))
        else:
            text = row['description']
            description = text if '(' not in text else clean_desc(text)
            description = description.split()
            if verbs[0] in description:
                ind = description.index(verbs[0])
                description = ' '.join(description[:ind + 6]) + '.'
            else:
                description = ' '.join(description) + '.'

            tmp = lower_init(description)
            docs.append(tmp)
            classification.append(row['classification'])
            keys.append('_'.join([row['class'], row['method_name']]))
    return keys, docs, classification


def generate_docs_FA(df):
    ''' Format: description'''
    docs = []
    for _, row in df.iterrows():
        if len(row['description']) > 0:
            text = row['description']
            description = text if '(' not in text else clean_desc(text)
            tmp = lower_init(description)
            tmp = split_description(description)
            docs.append(tmp.lower())
    return docs


def generate_docs_FB(df):
    ''' Format: method_name + description'''
    docs = []
    for _, row in df.iterrows():
        if len(row['description']) > 0:
            m_name = split(r'(?=[A-Z])', row['method_name'])
            m_name_tokens = ' '.join([x.lower() for x in m_name])
            text = row['description']
            description = text if '(' not in text else clean_desc(text)
            description = lower_init(description)
            description = split_description(description)
            tmp = ' '.join([m_name_tokens, description.lower()])
            docs.append(tmp)
    return docs


def generate_docs_FC(df):
    ''' Format: method_name + description + class name'''
    docs = []
    for _, row in df.iterrows():
        if len(row['description']) > 0:
            c_name = split(r'(?=[A-Z])', row['class'].replace('.', ''))
            class_tokens = ' '.join([x.lower()
                                    for x in c_name[1:] if len(x) > 1])
            m_name = split(r'(?=[A-Z])', row['method_name'])
            m_name_tokens = ' '.join([x.lower() for x in m_name])
            text = row['description']
            description = text if '(' not in text else clean_desc(text)
            description = lower_init(description)
            description = split_description(description)
            tmp = ' '.join(
                [m_name_tokens, 'from', class_tokens, description.lower()])
            docs.append(tmp)
    return docs


def generate_docs_bert_ft_C(df):
    ''' Format: method_name + description + class name'''
    docs = []
    keys = []
    for _, row in df.iterrows():
        if len(row['description']) > 0:
            c_name = split(r'(?=[A-Z])', row['class'].replace('.', ''))
            class_tokens = ' '.join([x.lower()
                                    for x in c_name[1:] if len(x) > 1])
            m_name = split(r'(?=[A-Z])', row['method_name'])
            m_name_tokens = ' '.join([x.lower() for x in m_name])
            text = row['description'].lower()
            description = text if '(' not in text else clean_desc(text)
            description = lower_init(description)
            description = split_description(description)
            tmp = ' '.join([m_name_tokens, 'from', class_tokens, description])
            docs.append(tmp)
            keys.append('_'.join([row['class'], row['method_name']]))
    return keys, docs


def generate_docs_FD(df):
    ''' Format: method_name + class name + description + class description and method signature'''
    docs = []
    for _, row in df.iterrows():
        c_des = row['class_description'].replace('-', ' ')
        c_name = split(r'(?=[A-Z])', row['class'].replace('.', ''))
        class_tokens = ' '.join([x.lower() for x in c_name[1:]])
        m_name = split(r'(?=[A-Z])', row['method_name'])
        m_name_tokens = ' '.join([x.lower() for x in m_name])
        class_tokens_des = c_des if '(' not in c_des else clean_desc(c_des)
        class_tokens_des = lower_init(class_tokens_des)
        class_tokens_des = split_description(class_tokens_des)
        if len(row['description']) > 0:
            text = row['description'].replace('.', '')
            description = text if '(' not in text else clean_desc(text)
            description = lower_init(description)
            description = split_description(description)
            tmp = ' '.join(
                [m_name_tokens, 'from', class_tokens, description.lower()])
            r = get_return_type(row['return'])
            if len(row['args_string']) > 0:
                arg_tokens = 'parameters are ' + \
                    ' '.join(get_params_token(row['args_string'], 'string'))
            else:
                arg_tokens = 'no argument'
            tmp = tmp + f'. Return type is {r} and {arg_tokens}. '
        else:
            tmp = ' '.join([m_name_tokens, 'from', class_tokens, '. '])

        if len(c_des) > 10:
            tmp = tmp + class_tokens_des.lower()
        docs.append(tmp.strip())
    return docs


def generate_docs_FE(df):
    ''' Format: description + class description'''
    docs = []
    for _, row in df.iterrows():
        # method description
        text = row['description']
        description = text if '(' not in text else clean_desc(text)
        description = lower_init(description)
        description = split_description(description)
        # class description
        c_name = split(r'(?=[A-Z])', row['class'].replace('.', ''))
        class_tokens = ' '.join([x.lower() for x in c_name[1:] if len(x) > 1])
        c_des = row['class_description'].replace('-', ' ')
        if len(c_des) > 10:
            class_tokens_des = c_des if '(' not in c_des else clean_desc(c_des)
            class_tokens_des = lower_init(class_tokens_des)
            class_tokens_des = split_description(class_tokens_des)
        else:
            class_tokens_des = class_tokens
        # join all parts
        tmp = ' '.join([description.lower(), '. ', class_tokens_des.lower()])
        docs.append(tmp)
    return docs


def generate_docs_FF(df):
    ''' Format: class description or (class name + method name) '''
    docs = []
    for _, row in df.iterrows():
        # method name
        m_name = split(r'(?=[A-Z])', row['method_name'])
        m_name_tokens = ' '.join([x.lower() for x in m_name])
        # class description
        c_name = split(r'(?=[A-Z])', row['class'].replace('.', ''))
        class_tokens = ' '.join([x.lower() for x in c_name[1:] if len(x) > 1])
        c_des = row['class_description'].replace('-', ' ')
        if len(c_des) > 10:
            class_tokens_des = c_des if '(' not in c_des else clean_desc(c_des)
            class_tokens_des = lower_init(class_tokens_des)
            class_tokens_des = split_description(class_tokens_des)
        else:
            class_tokens_des = class_tokens + ' ' + m_name_tokens
        # join all parts
        tmp = ' '.join([class_tokens_des.lower()])
        docs.append(tmp)
    return docs


def generate_docs_FG(df):
    ''' Format: class description + method name'''
    docs = []
    for _, row in df.iterrows():
        # method name
        m_name = split(r'(?=[A-Z])', row['method_name'])
        m_name_tokens = ' '.join([x.lower() for x in m_name])
        # class description
        c_name = split(r'(?=[A-Z])', row['class'].replace('.', ''))
        class_tokens = ' '.join([x.lower() for x in c_name[1:] if len(x) > 1])
        c_des = row['class_description'].replace('-', ' ')
        if len(c_des) > 10:
            class_tokens_des = c_des if '(' not in c_des else clean_desc(c_des)
            class_tokens_des = lower_init(class_tokens_des)
            class_tokens_des = split_description(class_tokens_des)
        else:
            class_tokens_des = class_tokens
        # join all parts
        tmp = ' .'.join([m_name_tokens, class_tokens_des.lower()])
        docs.append(tmp)
    return docs


def generate_docs_format_A(df):
    '''class_name + method_name + method_description
    '''
    docs = []
    keys = []
    for _, row in df.iterrows():
        if len(row['description']) > 0:
            c_name = split(r'(?=[A-Z])', row['class'].replace('.', ''))
            class_tokens = ' '.join([x.lower()
                                    for x in c_name[1:] if len(x) > 1])
            tmp = class_tokens + ' '
            m_name = split(r'(?=[A-Z])', row['method_name'])
            m_name_tokens = ' '.join([x.lower() for x in m_name])
            text = row['description']
            description = text if '(' not in text else clean_desc(text)
            tmp = tmp + m_name_tokens + ' ' + lower_init(description)
            docs.append(tmp)
            keys.append('_'.join([row['class'], row['method_name']]))
    return keys, docs


def generate_docs_format_C(df):
    '''class_description + method_name + method_description + argument types + return type.
     Split camel cased words'''
    docs = []
    for _, row in df.iterrows():
        if len(row['description']) > 0:
            c_des = row['class_description'].replace('-', ' ')
            if len(c_des) == 0:
                c_name = split(r'(?=[A-Z])', row['class'].replace('.', ''))
                class_tokens = ' '.join([x.lower() for x in c_name[1:]])
                tmp = class_tokens + ' '
            else:
                class_tokens = c_des if '(' not in c_des else clean_desc(c_des)
                tmp = split_description(class_tokens) + ' '
            m_name = split(r'(?=[A-Z])', row['method_name'])
            m_name_tokens = ' '.join([x.lower() for x in m_name])
            text = row['description'].replace('.', '')
            description = text if '(' not in text else clean_desc(text)
            description = split_description(description)
            tmp = tmp + m_name_tokens + ' ' + lower_init(description) + ' '
            r = get_return_type(row['return'])
            tmp = tmp + r + ' '
            if len(row['args_string']) > 0:
                arg_tokens = ' '.join(get_params_token(
                    row['args_string'], 'string'))
                tmp = tmp + arg_tokens
            docs.append(tmp.strip())
    return docs


def generate_input_training(text_data, keywords, target='description'):
    '''
        set the target to either description or name
    '''
    # tokenize method name and description
    sentences = []
    control = set()
    for ele in text_data:
        if target == 'description' and ele[1] != '':
            if ele[1] not in control:
                control.add(ele[1])
                tokens = get_description_tokens(ele[1], keywords)
                sentences.append(tokens)
        elif target == 'name':
            tokens = get_tokens_from_name(ele[0])
            sentences.append(tokens)
    return sentences


def gen_training_sentences(text_data, keywords):
    '''
        prepare descriptions for training 
    '''
    sentences = []
    for ele in set(text_data):
        if ele != '':
            tokens = get_description_tokens(ele, keywords)
            sentences.append(tokens)
    return sentences


def get_aosp_classes_methods(classes, methods, en_dict):
    '''
    Get classes and method names from Android AOSP documentation.
    Names are filtered by the english dictionary and lower cased
    '''
    classes_aosp = []
    # en_dict = set(words.words())
    for text in classes:
        name = text.split('/')[-1]
        name_tmp = name[0].lower() + name[1:]
        if name_tmp not in en_dict:
            classes_aosp.append(name)  # lower()
    methods_aosp = [
        method for method in methods if method not in en_dict]  # lower()
    return classes_aosp, methods_aosp


def get_embeddings(words, model):
    embeddings = []
    missings = []
    vocabulary = []
    for word in set(words):
        if model.has_index_for(word):
            vocabulary.append(word)
            embeddings.append(model[word])
        else:
            missings.append(word)
    print(f'OOO words: {len(missings)}')
    with open('OOO_words.txt', 'w') as f:
        for x in missings:
            f.write(f'{x}\n')
    return vocabulary, embeddings


''' Plotting K-distance Graph'''


def plot_k_distance_graph(X, n=2):
    print(f'avg distance k = {n}')
    neigh = NearestNeighbors(n_neighbors=n)
    nbrs = neigh.fit(X)
    distances, indices = nbrs.kneighbors(X)
    distances_pl = np.sort(distances, axis=0)
    distances_pl = distances_pl[:, 1]
    plt.figure(figsize=(8, 6))
    plt.plot(distances_pl)
    plt.title('K-distance Graph', fontsize=10)
    plt.xlabel('Data Points sorted by distance', fontsize=10)
    plt.ylabel('Epsilon', fontsize=10)
    plt.show()


def reduce_dimensions_pca(embeddings, dim=2):
    scaler = MinMaxScaler()
    data_rescaled = scaler.fit_transform(embeddings)
    pca = PCA(n_components=dim)
    X_principal = pca.fit_transform(data_rescaled)
    return X_principal


def get_metrics(target, predicted):
    accuracy = metrics.accuracy_score(target, predicted)
    metrics_cv = metrics.classification_report(
        target, predicted, output_dict=True)
    precision = metrics_cv['macro avg']['precision']
    recall = metrics_cv['macro avg']['recall']
    f1 = metrics_cv['macro avg']['f1-score']
    return accuracy, precision, recall, f1


def add_cat_map(out, inverse_map_susi):
    susi_cat_map = []
    for x in out['category_map']:
        if x in inverse_map_susi:
            susi_cat_map.append(inverse_map_susi[x])
        else:
            susi_cat_map.append(x)
    return susi_cat_map

# function to obtain the full qualified name of a class


def get_qualify_name_gps(df_full, library, cl_name):
    query = df_full.loc[(df_full['lib'] == library) & (
        df_full['name'] == cl_name), 'fqname']
    if query.shape[0] == 0:
        return ''
    else:
        return query.values[0]


def get_qualify_name_aosp(aosp_map, cl_name):
    if cl_name in aosp_map:
        return aosp_map[cl_name]
    else:
        return ''


def get_qualify_name(cl_name, library, aosp_map, gps_df):
    # basic object and popular, store for efficiency
    basic_map = {'void': 'void', 'short': 'short', 'int': 'int', 'long': 'long', 'float': 'float', 'byte': 'byte', 'boolean': 'boolean',
                 'String': 'java.lang.String', 'ArrayList': 'java.util.ArrayList', 'Intent': 'android.content.Intent', 'Uri': 'android.net.Uri',
                 'ByteBuffer': 'java.nio.ByteBuffer', 'CharSequence': 'java.lang.CharSequence', 'Bundle': 'android.os.Bundle',
                 'IBinder': 'android.os.IBinder', 'List': 'java.util.List', 'Log': 'android.util.Log', 'Context': 'android.content.Context'}

    full_class_name = cl_name
    if cl_name in basic_map:
        return basic_map[cl_name]
    if library != '':
        res = get_qualify_name_gps(gps_df, library, cl_name)
        if res != '':
            return res
    res = get_qualify_name_aosp(aosp_map, cl_name)
    full_class_name = res if res != '' else full_class_name
    return full_class_name


def get_aosp_map():
    with open('../inputs/classes_aosp_cleaned.txt') as f:
        lines = f.read().splitlines()
    aosp_class_map = dict()
    duplicated = []
    for line in lines:
        key = line.split('/')[-1]
        value = line.replace('/', '.')
        if key in aosp_class_map:
            duplicated.append([key, value])
        else:
            aosp_class_map[key] = value
    return aosp_class_map


def get_gps_map():
    return pd.read_csv('../inputs/gps_classes_qname.csv')
