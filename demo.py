from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import *
from keras.utils import to_categorical
import pandas as pd
import jieba
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, f1_score, recall_score
 
if __name__=='__main__':
    dataset = pd.read_csv('smsspamcollection/SMSSpamCollection.txt', sep='\t',names=['label', 'review']).astype(str)
    class_mapping = {'ham':1, 'spam':0}
    dataset['label'] = dataset['label'].map(class_mapping)
    cw = lambda x: list(jieba.cut(x))
    dataset['words'] = dataset['review'].apply(cw)
    tokenizer=Tokenizer()  #Create a Tokenizer object
    #The fit_on_texts function can number each word in the input text. The number is based on the word frequency. The higher the word frequency, the smaller the number
    tokenizer.fit_on_texts(dataset['words'])
    vocab=tokenizer.word_index #Get the number of each word
    x_train, x_test, y_train, y_test = train_test_split(dataset['words'], dataset['label'], test_size=0.1)
    # Convert each word in each sample into a list of numbers, using the number of each word for numbering
    x_train_word_ids=tokenizer.texts_to_sequences(x_train)
    x_test_word_ids = tokenizer.texts_to_sequences(x_test)
    #Cut off the part that exceeds the fixed value, and fill in the front with 0 for the insufficient
    x_train_padded_seqs=pad_sequences(x_train_word_ids,maxlen=50) 
    x_test_padded_seqs=pad_sequences(x_test_word_ids, maxlen=50)
    #CNN model
    model = Sequential()
    model.add(Embedding(len(vocab) + 1, 300, input_length=50)) #Use Embeding layer to convert each word encoding into word vector
    model.add(Conv1D(256, 5, padding='same'))
    model.add(MaxPooling1D(3, 3, padding='same'))
    model.add(Conv1D(128, 5, padding='same'))
    model.add(MaxPooling1D(3, 3, padding='same'))
    model.add(Conv1D(64, 3, padding='same'))
    model.add(Flatten())
    model.add(Dropout(0.1))
    model.add(BatchNormalization())  
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    one_hot_labels = to_categorical(y_train, num_classes=2)  # Convert tags to one-hot encoding
    model.fit(x_train_padded_seqs, one_hot_labels,epochs=5, batch_size=800)
    y_predict = model.predict_classes(x_test_padded_seqs).astype("int64")  # The prediction is the category
    print('accuracy', accuracy_score(y_test, y_predict))
    print('recall', recall_score(y_test, y_predict))
    print('f-measure:', f1_score(y_test, y_predict, average='weighted'))
