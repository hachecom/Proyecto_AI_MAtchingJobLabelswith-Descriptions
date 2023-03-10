import warnings

warnings.simplefilter("ignore")
with warnings.catch_warnings():
    warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
    warnings.filterwarnings(action='ignore', category=DeprecationWarning, module='gensim')
    warnings.filterwarnings(action='ignore', category=RuntimeWarning, module="gensim")

    import glob
    import operator
    import os
    import xlrd
    import time
    import spacy
    import xlsxwriter
    import numpy as np
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    from keras.optimizers import Adam
    from keras.models import Model, load_model

    from CNNJobMatching_3Label import preprocess_descriptions, padding, get_data

input_file = "labeled -general - Half.xlsx"
training_descriptions, list_job_sectors, list_job_labels = get_data()


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


def load_cnn_run(descriptions, file, path):

    model = load_model(path + "my_model.h5")
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    model.load_weights(file)

    outputs = model.predict(descriptions)

    return outputs


def write_excel(worksheet, descriptions, ai_label, ai_sublabel, row):

    col = 0
    row += 1

    for description, label, sublabel in zip(descriptions, ai_label, ai_sublabel):
        worksheet.write(row, col, description)
        col += 1
        worksheet.write(row, col, label)
        col += 1
        worksheet.write(row, col, sublabel)
        row += 1
        col = 0


def main():
    # extract the descriptions from the prediction file
    list_descriptions = extract_data()
    # get the data ready to place into the predictor CNN, tokenize, pad, etc
    list_data_ready = get_data_ready(list_descriptions)
    # get a list of all of the unique sectors( or primary labels )
    list_unique_sectors = unique_labels(list_job_sectors)
    # create workbook and begin writing
    workbook = xlsxwriter.Workbook('predict_CNN.xlsx')
    worksheet = workbook.add_worksheet()
    list_elements = ["Description", "Label", "Sub-Label"]
    col = 0
    row = 0
    for element in list_elements:
        worksheet.write(row, col, element)
        col += 1
    # create path for projects
    path = "./projects/General/"
    list_of_files = glob.glob(path + 'weights.*.hdf5')
    youngest_file = get_youngest_file(list_of_files)
    outputs = load_cnn_run(list_data_ready, youngest_file, path)
    outputs = np.array(outputs).tolist()

    for unique_sector in list_unique_sectors:
        list_desc_sector = []
        list_labels_sector = []
        list_data_ready_sector = []
        list_sublabels_sector = []
        for description, sector, data_ready, job_labels, output in zip(list_descriptions, list_job_sectors,
                                                                       list_data_ready, list_job_labels, outputs):
            index_max = output.index(max(output))
            if unique_sector == list_unique_sectors[index_max]:
                list_desc_sector.append(description)
                list_labels_sector.append(list_unique_sectors[index_max])
                list_data_ready_sector.append(data_ready)

        for training_desc, training_sec, training_label in zip(training_descriptions, list_job_sectors, list_job_labels):
            if training_sec == unique_sector:
                list_sublabels_sector.append(training_label)

        if list_desc_sector:
            list_data_ready_sector = np.asarray(list_data_ready_sector)
            path = "./projects/" + unique_sector + "/"
            list_of_files = glob.glob(path + "weights.*.hdf5")
            youngest_file = get_youngest_file(list_of_files)
            outputs_sublabels = load_cnn_run(list_data_ready_sector, youngest_file, path)
            outputs_sublabels = np.array(outputs_sublabels).tolist()

            unique_sublabels = []
            for sublabel in list_sublabels_sector:
                if sublabel not in unique_sublabels:
                    unique_sublabels.append(sublabel)

            list_sublabels_ai = []
            for output_sub in outputs_sublabels:
                index_sub = output_sub.index(max(output_sub))
                sublabel_ai = unique_sublabels[index_sub]
                list_sublabels_ai.append(sublabel_ai)

            write_excel(worksheet, list_desc_sector, list_labels_sector, list_sublabels_ai, row)
    workbook.close()


if __name__ == "__main__":
    main()

