import keras.utils.np_utils as np_utils
import numpy as np
import pandas as pd
from keras.layers import Bidirectional, GlobalAveragePooling1D, GlobalMaxPool1D
from keras.layers import Input, Embedding, Concatenate, Flatten, Dropout, BatchNormalization, LSTM, \
    SpatialDropout1D
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import cohen_kappa_score as kappa_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

np.random.seed(1337)  # for reproducibility
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop


def prepare_data(data):
    pet_id = data.PetID

    # Remove unused features
    data.drop(['RescuerID', 'PetID', 'State'], axis=1, inplace=True)

    # Apply binning to ages
    data['Age'] = pd.cut(data['Age'], [-1, 2, 3, 6, 255], labels=[0, 1, 2, 3])

    # Apply binning to fee
    data['Fee'] = pd.cut(data['Fee'], [-1, 50, 100, 200, 3000], labels=[0, 1, 2, 3])

    # Apply binning to photo amount
    data['PhotoAmt'] = pd.cut(data['PhotoAmt'], [-1, 1, 5, 10, 100], labels=[0, 1, 2, 3])

    # Apply binning to video amount
    data['VideoAmt'] = pd.cut(data['VideoAmt'], [-1, 1, 100], labels=[0, 1])

    # Replace names with 1 is present, 0 if not present
    data.loc[data['Name'].notnull(), 'Name'] = 1
    data.loc[data['Name'].isnull(), 'Name'] = 0

    # Fill missing continuous data
    data_continuous = data.select_dtypes(exclude=['object'])
    data_continuous.fillna(0, inplace=True)

    # Fill missing string data
    data_categorical = data.select_dtypes(include=['object'])
    data_categorical.fillna('NONE', inplace=True)

    final_data = data_continuous.merge(data_categorical, left_index=True, right_index=True)

    return final_data, data_categorical, data_continuous, pet_id


kappa_scorer = make_scorer(kappa_score)

train_df = pd.read_csv("/Users/viteka/final_project/all/train.csv")
test_df = pd.read_csv("/Users/viteka/final_project/all/test/test.csv")

cat_cols = ['Type', 'Age', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2', 'Color3', 'MaturitySize',
            'FurLength', 'Vaccinated', 'Dewormed', 'Sterilized', 'Health', 'Quantity', 'VideoAmt', 'PhotoAmt']
num_cols = ['Fee']
text_cols = ['Description']
embed_sizes = [len(train_df[col].unique()) + 1 for col in cat_cols]
print(embed_sizes)

print('scaling num_cols')

train, train_categorical, train_continuous, train_pet_id = prepare_data(train_df)
test, test_categorical, test_continuous, test_pet_id = prepare_data(test_df)

for col in num_cols:
    print('scaling {}'.format(col))
    # col_mean = train_df[col].mean()
    # train_df[col].fillna(col_mean, inplace=True)
    # test_df[col].fillna(col_mean, inplace=True)
    scaler = StandardScaler()
    train_df[col] = scaler.fit_transform(train[col].reshape(-1, 1))
    test_df[col] = scaler.transform(test[col].reshape(-1, 1))

print('getting embeddings')


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


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
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector


def get_input_features(df):
    X = {'description': pad_sequences(df['Description'], maxlen=maxlen)}
    X['numerical'] = np.array(df[num_cols])
    for cat in cat_cols:
        X[cat] = np.array(df[cat])


categorical_inputs = train_categorical

for cat in cat_cols:
    categorical_inputs.append(Input(shape=[1], name=cat))

categorical_embeddings = []

for i, cat in enumerate(cat_cols):
    categorical_embeddings.append(
        Embedding(embed_sizes[i], 10)(categorical_inputs[i]))

categorical_logits = Concatenate()([Flatten()(cat_emb) for cat_emb in categorical_embeddings])
categorical_logits = Dense(256, activation='relu')(categorical_logits)

numerical_inputs = train_continuous
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

tr_df, val_df = train_test_split(train_df, test_size=0.2, random_state=23)
y_train = np_utils.to_categorical(tr_df['AdoptionSpeed'], num_classes=5)
y_valid = np_utils.to_categorical(val_df['AdoptionSpeed'], num_classes=5)

cs = MinMaxScaler()
trainContinuous = cs.fit_transform(tr_df[cat_cols])
trainContinuous2 = cs.fit_transform(val_df[cat_cols])
trainContinuous3 = cs.fit_transform(test_df[cat_cols])
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
