import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer

class DataPreprocess:
    def __init__(self, df, is_test_input=False, is_vectorize=False, vectorizer=None, analyzer="char"):
        self.is_vectorize = is_vectorize
        self.is_test_input = is_test_input
        self.is_tfidf_vect = False
        self.label_enc = None
        if is_vectorize:
            self.X_data = df.sequence
            self.y_data = df.target
            self.vect = vectorizer(analyzer=analyzer).fit(self.X_data)
            if "CountVectorizer" not in str(type(vectorizer())):
                self.is_tfidf_vect = True
        else:
            self.X_data, self.y_data = self.__make_new_df(df)
        
    def __make_new_df(self, df):
        y_data = df.target
        def convert_to_int(x):
            return self.letter_bags.index(x)

        tmp = df.sequence.apply(lambda seq: list(seq))
        new_df = np.zeros((len(tmp), df.seq_len.max()))

        str_tmp = ""
        for letter in df.seq_unique_letters.unique():
            str_tmp += letter
        self.letter_bags = list(set(list(str_tmp)))

        for idx, _ in enumerate(new_df):
            converted = map(convert_to_int, list(tmp.values[idx]))
            new_df[idx] = list(converted)
        
        return new_df, y_data
    
    def __vectorize(self, X_data):
        if self.is_tfidf_vect:
            tmp_arr = self.vect.transform(X_data).toarray()
            tmp_arr = tmp_arr*100
            return tmp_arr.astype(int)
        return self.vect.transform(X_data).toarray()
    
    def get_vectorized_data(self):
        return self.__vectorize(self.X_data)
    
    def get_vectorized_data_to_eval(self, x, is_tfidf_vect):
        if is_tfidf_vect:
            tmp_arr = self.vect.transform(x).toarray()
            tmp_arr = tmp_arr*100
            return tmp_arr.astype(int)
        return self.vect.transform(x).toarray()

    def get_inversed_target_data(self, y):
        return self.label_bin.inverse_transform(y)
    
    def get_preprocessed_data(self, X_data=None, y_data=None, test_size=0.4, random_state=0, is_for_nn=False):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_data, self.y_data, test_size=test_size, random_state=random_state)
        if self.is_vectorize:
            self.X_train = self.__vectorize(self.X_train)
            self.X_test = self.__vectorize(self.X_test)
            
        if is_for_nn:
            self.label_bin = LabelBinarizer().fit(self.y_data)
            self.y_train = self.label_bin.transform(self.y_train)
            self.y_test = self.label_bin.transform(self.y_test)
            
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def get_original_data(self):
        return self.X_data, self.y_data