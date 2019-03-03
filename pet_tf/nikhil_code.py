# libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Dense, Input, Reshape
from keras.layers import Embedding, Dropout, Concatenate
from keras.models import Model
from keras.utils import to_categorical
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')


breeds = pd.read_csv('/Users/viteka/final_project/all/breed_labels.csv')
colors = pd.read_csv('/Users/viteka/final_project/all/color_labels.csv')
states = pd.read_csv('/Users/viteka/final_project/all/state_labels.csv')

data = pd.read_csv('/Users/viteka/final_project/all/train.csv')

all_data = data

data['Breed1'] = data['Breed1'].map(breeds.set_index('BreedID')['BreedName'])
data['Breed2'] = data['Breed2'].map(breeds.set_index('BreedID')['BreedName'])

data['State'] = data['State'].map(states.set_index('StateID')['StateName'])

data['Color1'] = data['Color1'].map(colors.set_index('ColorID')['ColorName'])
data['Color2'] = data['Color2'].map(colors.set_index('ColorID')['ColorName'])
data['Color3'] = data['Color3'].map(colors.set_index('ColorID')['ColorName'])

genderDict = {1: 'Male', 2: 'Female', 3: 'Mixed'}
typeDict = {1: 'Dog', 2: 'Cat'}
maturityDict = {1: 'Small', 2: 'Medium', 3: 'Large', 4: 'Extra Large', 0: 'Not Specified'}
healthDict = {1: 'Healthy', 2: 'Minor Injury', 3: 'Serious Injury', 0: 'Not Specified'}
furDict = {1: 'Short', 2: 'Medium', 3: 'Long', 0: 'Not Specified'}

data['Gender'] = data['Gender'].map(genderDict)
data['Type'] = data['Type'].map(typeDict)
data['MaturitySize'] = data['MaturitySize'].map(maturityDict)
data['Health'] = data['Health'].map(healthDict)
data['FurLength'] = data['FurLength'].map(furDict)

data_label = data.AdoptionSpeed
# We drop name because it creates a huge embedding vector and we know that name is not very useful anyway
data = data.drop(['AdoptionSpeed', 'Name'], axis=1)

train, test, train_label, test_label = train_test_split(data, data_label, test_size=0.33, random_state=9)

# Turn labels into n dimensional vectors for loss calculation
train_label = to_categorical(train_label, num_classes=None)
test_label = to_categorical(test_label, num_classes=None)

train.drop('Description', axis=1, inplace=True)

categorical_vars = ['Type', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2',
                    'Color3', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed',
                    'Sterilized', 'Health', 'State']
numerical_vars = ['Age', 'Quantity', 'Fee', 'VideoAmt', 'PhotoAmt']


# Creating a Embedding model for categorical variables using the fast.ai approach
def createEmbeddingsModel(data, categorical_vars, numerical_vars):
    embeddings = []
    inputs = []
    for categorical_var in categorical_vars:
        i = Input(shape=(1,))
        no_of_unique_cat = data[categorical_var].nunique()
        embedding_size = min(np.ceil((no_of_unique_cat) / 2), 50)
        embedding_size = int(embedding_size)
        vocab = no_of_unique_cat + 1
        embedding = Embedding(vocab, embedding_size, input_length=1)(i)
        embedding = Reshape(target_shape=(embedding_size,))(embedding)
        embeddings.append(embedding)
        inputs.append(i)

    input_numeric = Input(shape=(len(numerical_vars),))
    embedding_numeric = Dense(16)(input_numeric)
    inputs.append(input_numeric)
    embeddings.append(embedding_numeric)

    x = Concatenate()(embeddings)
    x = Dense(80, activation='relu')(x)
    x = Dropout(.35)(x)
    x = Dense(20, activation='relu')(x)
    x = Dropout(.15)(x)
    x = Dense(10, activation='relu')(x)
    x = Dropout(.15)(x)
    output = Dense(5, activation='sigmoid')(x)

    model = Model(inputs, output)
    model.compile(metrics=['accuracy'], loss='categorical_crossentropy', optimizer='adam')
    return model


model = createEmbeddingsModel(train, categorical_vars, numerical_vars)


def preproc(X_train, X_test, embed_cols, num_cols):
    input_list_train = []
    input_list_test = []
    m = MinMaxScaler()

    # the cols to be embedded: rescaling to range [0, # values)
    for c in embed_cols:
        raw_vals = np.unique(X_train[c].astype(str))
        val_map = {}
        for i in range(len(raw_vals)):
            val_map[raw_vals[i]] = i
        m.fit(X_train[c].map(val_map).values.reshape(-1, 1))
        input_list_train.append(m.transform(X_train[c].map(val_map).values.reshape(-1, 1)))

        m.fit(X_test[c].map(val_map).fillna(0).values.reshape(-1, 1))
        input_list_test.append(m.transform(X_test[c].map(val_map).fillna(0).values.reshape(-1, 1)))

    # the numerical columns
    m.fit(X_train[num_cols].values)
    input_list_train.append(m.transform(X_train[num_cols].values))

    m.fit(X_test[num_cols].values)
    input_list_test.append(m.transform(X_test[num_cols].values))

    return input_list_train, input_list_test


X_train, X_test = preproc(train, test, categorical_vars, numerical_vars)

hist = model.fit(X_train, train_label, batch_size=64, epochs=50, validation_split=0.1, shuffle=True)

plt.plot(hist.history['acc'])

model.evaluate(X_test, test_label)
