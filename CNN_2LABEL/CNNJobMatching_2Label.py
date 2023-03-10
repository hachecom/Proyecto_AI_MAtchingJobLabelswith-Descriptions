import warnings

warnings.simplefilter("ignore")
with warnings.catch_warnings():
    warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
    warnings.filterwarnings(action='ignore', category=DeprecationWarning, module='gensim')
    warnings.filterwarnings(action='ignore', category=RuntimeWarning, module="gensim")
    import gensim
    import xlrd
    import spacy
    import numpy as np
    from gensim.utils import simple_preprocess
    import gensim.corpora as corpora
    import xlsxwriter
    import os
    import errno

    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    from keras.utils import to_categorical
    from keras.layers import Input, Dense, Embedding, merge, Convolution2D, MaxPooling2D, Dropout, Conv2D, concatenate,\
        Reshape, Flatten
    from keras.optimizers import Adam
    from keras.models import Model
    from keras.callbacks import ModelCheckpoint
    from sklearn.model_selection import train_test_split

word2vec_training = "wiki-news-300d-1M.vec"  # Download a pre trained word2vec model
model_word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_training, binary=False)  # Load the model

input_file = "labeled- computer_science - Half.xlsx"

no_below = 21
no_above = 0.3


def makedir_p(path):
    """Returns a directory"""

    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def get_data():
    workbook = xlrd.open_workbook(input_file)
    sheet = workbook.sheet_by_index(0)

    list_job_labels = []
    list_job_descriptions = []

    for row in range(sheet.nrows):
        list_job_labels.append(sheet.cell_value(row, 1))

    for row in range(sheet.nrows):
        list_job_descriptions.append(sheet.cell_value(row, 0))

    return list_job_descriptions, list_job_labels


def sent_to_words(sentences):
    """Tokenize and remove accent marks"""

    for sentence in sentences:
        yield (simple_preprocess(str(sentence), deacc=True))


def remove_stopwords(texts, nlp):
    """Removes stopwords using Spacy"""

    # Spacy stopwords
    return [[word for word in doc if not nlp.vocab[word].is_stop] for doc in texts]


def lemmatization(texts, nlp, allowed_postags):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])

    return texts_out


def preprocess_descriptions(answers, nlp):
    """Pre process of the data
    Keyword Arguments:
    answers -- answers to the survey questions, raw data
    """

    # Removing accents and simple pre-process
    data_words = list(sent_to_words(answers))

    data_words = remove_stopwords(data_words, nlp)

    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words, nlp, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    # remove empty lists and words with len < 3
    """
    data_lemmatized_aux = []
    data_lemmatized_cleaned = []
    for l_words in data_lemmatized:
        if len(l_words) > 0:
            for word in l_words:
                if len(word) > 3:
                    data_lemmatized_aux.append(word)

            # if it has something after removal of 3 letters words
            if len(data_lemmatized_aux) > 0:
                data_lemmatized_cleaned.append(data_lemmatized_aux)

            data_lemmatized_aux = []
    """
    # filter very uncommon and very common words with gensim
    id2word = corpora.Dictionary(data_lemmatized)
    id2word.filter_extremes(no_below=no_below, no_above=no_above)

    # place the words not filtered in list
    list_id2word = []
    for word in id2word.values():
        list_id2word.append(word)

    # compare the words in the corpus against the words not filtered.
    data_final = []
    data_final_aux = []

    for lista in data_lemmatized:
        for word in lista:
            if word in list_id2word:
                data_final_aux.append(word)
        data_final.append(data_final_aux)
        data_final_aux = []

    return data_final


def padding(job_descriptions):
    """ Pad the sentences that don´t have enough words.
    Keywords Arguments --
    job_descriptions - list of lists where each list is a tokenized and cleaned job description.
    """
    # Find the longest description
    list_lengths = []
    for sentence in job_descriptions:
        list_lengths.append(len(sentence))
    maxi = max(list_lengths)

    # Pad the sentences so they all have equal length
    for sentence in job_descriptions:
        while len(sentence) < maxi:
            sentence.append("<PAD>")

    return job_descriptions


def train_cnn(job_desc_uncleaned, job_descriptions, job_labels, wv):
    """Create the CNN architecture.
       Keyword Arguments--
       job_desc_uncleaned- list of Job descriptions taken straight from file
       job_descriptions- list of job descriptions after being cleaned
       job_labels- list of job_labels taken straight from file
    """
    # Place the labels in the correct format, and create a set of unique labels to be used later
    list_labels = []
    list_labels_categorized = []
    counter = 0
    for label in job_labels:
        if label not in list_labels:
            list_labels.append(label)
            list_labels_categorized.append(counter)
            counter += 1
        elif label in list_labels:
            indexed_label = list_labels.index(label)
            list_labels_categorized.append(indexed_label)

    # categorize the labels (convert into matrix format)
    list_labels_categorized = to_categorical(np.asarray(list_labels_categorized))

    tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'',
                          lower=True)

    # give each word a token
    tokenizer.fit_on_texts(job_descriptions)
    # convert job descriptions into token descriptions
    job_descriptions_sequence = tokenizer.texts_to_sequences(job_descriptions)

    # pad each sentence
    job_descriptions_padded = pad_sequences(job_descriptions_sequence, padding='post')
    print(job_descriptions_padded[0])

    # create a dictionary of word: token
    word_index = tokenizer.word_index

    # split the data for training and testing
    x_train, x_test, y_train, y_test = train_test_split(job_descriptions_padded, list_labels_categorized, test_size=0.20,
                                                        random_state=42)
    print(x_train.shape)
    # split the data to get the descriptions and match them later
    x2_train, x2_test, y2_train, y2_test = train_test_split(job_desc_uncleaned, list_labels_categorized, test_size=0.20,
                                                            random_state=42)

    embedding_dim = 300
    filter_sizes = [2, 3, 4]
    num_filters = 100
    drop = 0.5
    nb_epoch = 1
    batch_size = 5

    vocabulary_size = len(word_index) + 1
    embedding_matrix = np.zeros((vocabulary_size, embedding_dim))
    print(word_index)
    # create the embedding matrix
    for word, i in word_index.items():
        try:
            embedding_vector = wv[word]
            embedding_matrix[i] = embedding_vector
        except KeyError:
            pass

    sequence_length = x_train.shape[1]

    print(embedding_matrix)
    inputs = Input(shape=(sequence_length,))

    embedding_layer = Embedding(input_length=sequence_length, output_dim=embedding_dim, input_dim=vocabulary_size,
                                weights=[embedding_matrix], trainable=False)(inputs)
    print(embedding_layer)
    reshape = Reshape((sequence_length, embedding_dim, 1))(embedding_layer)
    print(reshape)
    conv_0 = Convolution2D(num_filters, (filter_sizes[0], embedding_dim), border_mode='valid', init='normal',
                           activation='sigmoid', dim_ordering='tf')(reshape)
    conv_1 = Convolution2D(num_filters, (filter_sizes[1], embedding_dim), border_mode='valid', init='normal',
                           activation='sigmoid', dim_ordering='tf')(reshape)
    conv_2 = Convolution2D(num_filters, (filter_sizes[2], embedding_dim), border_mode='valid', init='normal',
                           activation='sigmoid', dim_ordering='tf')(reshape)

    maxpool_0 = MaxPooling2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1, 1), border_mode='valid',
                             dim_ordering='tf')(conv_0)
    maxpool_1 = MaxPooling2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1), strides=(1, 1), border_mode='valid',
                             dim_ordering='tf')(conv_1)
    maxpool_2 = MaxPooling2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1), strides=(1, 1), border_mode='valid',
                             dim_ordering='tf')(conv_2)

    merged_tensor = concatenate([maxpool_0, maxpool_1, maxpool_2], axis=1)

    flatten = Flatten()(merged_tensor)

    dropout = Dropout(drop)(flatten)

    dense_1 = Dense(output_dim=1000, activation='sigmoid')(dropout)

    dense_2 = Dense(output_dim=1000, activation='sigmoid')(dense_1)

    output = Dense(output_dim=len(list_labels), activation='sigmoid')(dense_2)

    model = Model(input=inputs, output=output)

    checkpoint = ModelCheckpoint('./projects/weights.{epoch:03d}-{val_acc:.4f}.hdf5',
                                 monitor='val_acc', verbose=1,
                                 save_best_only=True, mode='auto')
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    print(model.summary())

    model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, callbacks=[checkpoint],
              validation_data=(x_test, y_test))
    model.save_weights('./projects/final weights')
    model.save('./projects/my_model.h5')
    outputs = model.predict(x_test)

    score, acc = model.evaluate(x_test, y_test,
                                batch_size=batch_size)
    print(model.metrics_names)
    print('Test score:', score)
    print('Test accuracy:', acc)

    return outputs, y_test, x2_test, list_labels


def write_results(outputs, y_test, x2_test, list_labels):

    # initiate workbook
    workbook = xlsxwriter.Workbook('Outputs_CNN.xlsx')
    worksheet = workbook.add_worksheet()
    row = 0
    col = 0

    list_things = ["Description", "Human", "AI", "", "", "Label", 'acc', 'recall', 'precision', 'f1']
    for element in list_things:
        worksheet.write(row, col, element)
        col += 1

    col = 0
    row += 1
    outputs = np.array(outputs).tolist()
    y_test = np.array(y_test).tolist()
    # write the description and the AI and human decisions in excel
    for a, b, c in zip(x2_test, y_test, outputs):
        worksheet.write(row, col, a)
        col += 1
        max_y = b.index(max(b))
        worksheet.write(row, col, list_labels[max_y])
        max_output = c.index(max(c))
        col += 1
        worksheet.write(row, col, list_labels[max_output])
        row += 1
        col = 0

    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    row = 1
    col = 5
    # create the metrics for a multi-label classifier
    # first we measure the TP, FN, FP, TN
    for counter in range(len(list_labels)):
        for a, b in zip(y_test, outputs):
            if a.index(max(a)) == counter or b.index(max(b)) == counter:
                if a.index(max(a)) == counter:
                    if a.index(max(a)) == b.index(max(b)):
                        true_positives += 1
                    else:
                        false_negatives += 1
                elif b.index(max(b)) == counter:
                    if b.index(max(b)) != a.index(max(a)):
                        false_positives += 1
            elif a.index(max(a)) != counter and b.index(max(b)) != counter:
                true_negatives += 1
        # then we calculate the metrics of each separate class
        acc = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
        if (true_positives + false_negatives) > 0:
            recall = true_positives / (true_positives + false_negatives)
        else:
            recall = 0
        if (true_positives + false_positives) > 0:
            precision = true_positives / (true_positives + false_positives)
        else:
            precision = 0
        if (recall + precision) > 0:
            f1 = (2 * recall * precision) / (recall + precision)
        else:
            f1 = 0

        # create a matrix with the scores and labels
        worksheet.write(row, col, list_labels[counter])
        col += 1
        worksheet.write(row, col, acc)
        col += 1
        worksheet.write(row, col, recall)
        col += 1
        worksheet.write(row, col, precision)
        col += 1
        worksheet.write(row, col, f1)

        row += 1
        col = 5

        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0

    workbook.close()


def main():
    # get original descriptions and labels
    job_descriptions, job_labels = get_data()
    # load spacy
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    # clean descriptions
    descriptions_cleaned = preprocess_descriptions(job_descriptions, nlp)
    # pad descriptions
    descriptions_cleaned_padded = padding(descriptions_cleaned)
    # create a directory where you will save weights
    makedir_p("projects")
    # train cnn

    outputs, y_test, x2_test, list_labels = train_cnn(job_descriptions, descriptions_cleaned_padded, job_labels, model_word2vec)
    # write results with metrics in excel
    write_results(outputs, y_test, x2_test, list_labels)


if __name__ == "__main__":
    main()
