import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score as kappa_score
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler

kappa_scorer = make_scorer(kappa_score)

from sklearn.preprocessing import MinMaxScaler

train_df = pd.read_csv("/Users/viteka/final_project/all/train.csv")
test_df = pd.read_csv("/Users/viteka/final_project/all/test/test.csv")

cat_cols = ['Type', 'Age', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2', 'Color3', 'MaturitySize',
            'FurLength', 'Vaccinated', 'Dewormed', 'Sterilized', 'Health', 'Quantity', 'State', 'VideoAmt', 'PhotoAmt']

num_cols = ['Fee']

text_cols = ['Description']

embed_sizes = [len(train_df[col].unique()) + 1 for col in cat_cols]

print(embed_sizes)

print('scaling num_cols')
for col in num_cols:
    print('scaling {}'.format(col))
    col_mean = train_df[col].mean()
    train_df[col].fillna(col_mean, inplace=True)
    test_df[col].fillna(col_mean, inplace=True)
    scaler = StandardScaler()
    train_df[col] = scaler.fit_transform(train_df[col].values.reshape(-1, 1))
    test_df[col] = scaler.transform(test_df[col].values.reshape(-1, 1))

from tqdm import tqdm
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

print('getting embeddings')


def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')


embeddings_index = dict(
    get_coefs(*o.rstrip().rsplit(' ')) for o in tqdm(open('/Users/viteka/final_project/wiki-news-300d-1M-subword.vec')))

num_words = 20000
maxlen = 80
embed_size = 300

train_df['Description'] = train_df['Description'].astype(str).fillna('no text')
test_df['Description'] = test_df['Description'].astype(str).fillna('no text')

print("   Fitting tokenizer...")
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(train_df['Description'].values.tolist())

train_df['Description'] = tokenizer.texts_to_sequences(train_df['Description'])
test_df['Description'] = tokenizer.texts_to_sequences(test_df['Description'])

word_index = tokenizer.word_index
nb_words = min(num_words, len(word_index))
embedding_matrix = np.zeros((nb_words, embed_size))
for word, i in word_index.items():
    if i >= num_words: continue
    try:
        embedding_vector = embeddings_index[word]
    except KeyError:
        embedding_vector = None
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


def get_input_features(df):
    X = {'description': pad_sequences(df['Description'], maxlen=maxlen)}
    X['numerical'] = np.array(df[num_cols])
    for cat in cat_cols:
        X[cat] = np.array(df[cat])
    return X


from keras.layers import Input, Embedding, Concatenate, Flatten, Dense, Dropout, BatchNormalization, LSTM, CuDNNLSTM, \
    SpatialDropout1D
from keras.layers import Bidirectional, GlobalAveragePooling1D, GlobalMaxPool1D
from keras.models import Model
from keras.optimizers import Adam

categorical_inputs = []
for cat in cat_cols:
    categorical_inputs.append(Input(shape=[1], name=cat))

categorical_embeddings = []
for i, cat in enumerate(cat_cols):
    categorical_embeddings.append(
        Embedding(embed_sizes[i], 10)(categorical_inputs[i]))

categorical_logits = Concatenate()([Flatten()(cat_emb) for cat_emb in categorical_embeddings])
categorical_logits = Dense(256, activation='relu')(categorical_logits)

numerical_inputs = Input(shape=[len(num_cols)], name='numerical')
numerical_logits = numerical_inputs
numerical_logits = BatchNormalization()(numerical_logits)
numerical_logits = Dense(128, activation='relu')(numerical_logits)

text_inp = Input(shape=[maxlen], name='description')
text_embed = Embedding(nb_words, embed_size, weights=[embedding_matrix], trainable=False)(text_inp)
text_logits = SpatialDropout1D(0.2)(text_embed)
text_logits = Bidirectional(LSTM(64, return_sequences=True))(text_logits)
avg_pool = GlobalAveragePooling1D()(text_logits)
max_pool = GlobalMaxPool1D()(text_logits)
text_logits = Concatenate()([avg_pool, max_pool])

x = Concatenate()([categorical_logits, text_logits, numerical_logits])
x = BatchNormalization()(x)

x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
out = Dense(1, activation='sigmoid')(x)

model = Model(inputs=[text_inp] + categorical_inputs + [numerical_inputs], outputs=out)
model.compile(optimizer=Adam(lr=0.0001), loss='mse')

from sklearn.model_selection import train_test_split

# for i, l in enumerate(tr_df['AdoptionSpeed'].values):
#    y_train[i,l] = 1
# for i, l in enumerate(val_df['AdoptionSpeed'].values):
#    y_valid[i,l] = 1
tr_df, val_df = train_test_split(train_df, test_size=0.2, random_state=23)

from keras.utils import np_utils

print(tr_df['AdoptionSpeed'].values.shape)

y_train = tr_df['AdoptionSpeed'].values / 4
y_valid = val_df['AdoptionSpeed'].values / 4

y_train = np_utils.to_categorical(tr_df['AdoptionSpeed'], num_classes=5)
y_valid = np_utils.to_categorical(val_df['AdoptionSpeed'], num_classes=5)
y_test = np_utils.to_categorical(test_df['AdoptionSpeed'], num_classes=5)

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1337)  # for reproducibility
from keras.models import Sequential
from keras.layers import Activation
from keras.optimizers import RMSprop

continuous = ['Type', 'Age', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2', 'Color3', 'MaturitySize',
              'FurLength', 'Vaccinated', 'Dewormed', 'Sterilized', 'Health', 'Quantity', 'State', 'VideoAmt',
              'PhotoAmt']

cs = MinMaxScaler()
trainContinuous = cs.fit_transform(tr_df[continuous])
trainContinuous2 = cs.fit_transform(val_df[continuous])
trainContinuous3 = cs.fit_transform(test_df[continuous])

x_train = np.hstack([trainContinuous])
x_valid = np.hstack([trainContinuous2])
x_test = np.hstack([trainContinuous3])
model = Sequential([
    Dense(32, input_dim=18),
    Activation('relu'),
    Dense(5),
    Activation('sigmoid'),
])
rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
# We add metrics to get more results you want to see
# categorical_crossentropy--mse
model.compile(optimizer=rmsprop,
              loss='mse',
              metrics=['accuracy'])
model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=260, batch_size=10)

print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(x_test, y_test)

print('test loss: ', loss)
print('test accuracy: ', accuracy)
