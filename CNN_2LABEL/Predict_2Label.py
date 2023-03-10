import warnings

warnings.simplefilter("ignore")
with warnings.catch_warnings():
    warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
    warnings.filterwarnings(action='ignore', category=DeprecationWarning, module='gensim')
    warnings.filterwarnings(action='ignore', category=RuntimeWarning, module="gensim")

    import glob
    import operator
    import os
    import time
    import xlrd
    import spacy
    import xlsxwriter
    import numpy as np
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    from keras.optimizers import Adam
    from keras.models import Model, load_model

    from CNNJobMatching_2Label import preprocess_descriptions, padding, get_data


input_file = "Data_Chengyao.xlsx"
# get the job labels from the training to then find all unique labels
x, job_labels = get_data()


def get_oldest_file(files, _invert=False):
    """ Find and return the oldest file of input file names.
    Only one wins tie. Values based on time distance from present.
    Use of `_invert` inverts logic to make this a youngest routine,
    to be used more clearly via `get_youngest_file`.
    """
    gt = operator.lt if _invert else operator.gt
    # Check for empty list.
    if not files:
        return None
    # Raw epoch distance.
    now = time.time()
    print(files)
    # Select first as arbitrary sentinel file, storing name and age.
    oldest = files[0], now - os.path.getctime(files[0])
    # Iterate over all remaining files.
    for f in files[1:]:
        age = now - os.path.getctime(f)
        if gt(age, oldest[1]):
            # Set new oldest.
            oldest = f, age
    # Return just the name of oldest file.
    return oldest[0]


def get_youngest_file(files):
    return get_oldest_file(files, _invert=True)


def unique_labels(labels):
    list_labels = []
    for label in labels:
        if label not in list_labels:
            list_labels.append(label)
    return list_labels


def extract_data():
    workbook = xlrd.open_workbook(input_file)
    sheet = workbook.sheet_by_index(0)
    list_job_descriptions = []

    for row in range(sheet.nrows):
        list_job_descriptions.append(sheet.cell_value(row, 0))

    return list_job_descriptions


def get_data_ready(list_job_descriptions):
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    job_descriptions = preprocess_descriptions(list_job_descriptions, nlp)
    job_descriptions = padding(job_descriptions)

    tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'',
                          lower=True)

    # give each word a token and then convert the job descriptions into token descriptions
    tokenizer.fit_on_texts(job_descriptions)
    job_descriptions_sequence = tokenizer.texts_to_sequences(job_descriptions)
    # pad each sentence
    job_descriptions_padded = pad_sequences(job_descriptions_sequence, padding='post')

    return job_descriptions_padded


def load_cnn_run(descriptions, file):
    model = load_model('./projects/my_model.h5')
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    model.load_weights(file)

    outputs = model.predict(descriptions)

    return outputs


def write_excel(descriptions, outputs, list_labels):
    workbook = xlsxwriter.Workbook('prediction_CNN.xlsx')
    worksheet = workbook.add_worksheet()
    row = 0
    col = 0

    outputs = np.array(outputs).tolist()
    for description, output in zip(descriptions, outputs):
        worksheet.write(row, col, description)
        col += 1
        max_output = output.index(max(output))
        worksheet.write(row, col, list_labels[max_output])
        row += 1
        col = 0

    workbook.close()


def main():
    # load the unique labels
    list_labels = unique_labels(job_labels)
    # extract the data from the file
    descriptions = extract_data()
    print(descriptions)
    # get the data ready for the model to predict on it
    descriptions_ready = get_data_ready(descriptions)
    # find the youngest file in the carpet projects
    path = "./projects/"
    list_of_files = glob.glob(path + 'weights.*.hdf5')
    youngest_file = get_youngest_file(list_of_files)
    # predict
    outputs = load_cnn_run(descriptions_ready, youngest_file)
    # outputs in excel
    write_excel(descriptions, outputs, list_labels)


if __name__ == "__main__":
    main()
