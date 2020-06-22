# encoding=utf-8
from tensorflow.python.keras.models import Model
from deepctr.inputs import build_input_features, combined_dnn_input, create_embedding_matrix
from deepctr.layers.core import PredictionLayer, DNN
from deepmatch.inputs import input_from_feature_columns
from deepmatch.layers.core import Similarity
from keras.utils import plot_model


def DSSM(user_feature_columns, item_feature_columns, user_dnn_hidden_units=(64, 32),
         item_dnn_hidden_units=(64, 32),
         dnn_activation='tanh', dnn_use_bn=False,
         l2_reg_dnn=0, l2_reg_embedding=1e-6, dnn_dropout=0, init_std=0.0001, seed=1024, metric='cos'):

    embedding_matrix_dict = create_embedding_matrix(user_feature_columns + item_feature_columns, l2_reg_embedding,
                                                    init_std, seed,
                                                    seq_mask_zero=True)

    user_features = build_input_features(user_feature_columns)
    user_inputs_list = list(user_features.values())
    user_sparse_embedding_list, user_dense_value_list = input_from_feature_columns(user_features,
                                                                                   user_feature_columns,
                                                                                   l2_reg_embedding, init_std, seed,
                                                                                   embedding_matrix_dict=embedding_matrix_dict)
    user_dnn_input = combined_dnn_input(user_sparse_embedding_list, user_dense_value_list)

    item_features = build_input_features(item_feature_columns)
    item_inputs_list = list(item_features.values())
    item_sparse_embedding_list, item_dense_value_list = input_from_feature_columns(item_features,
                                                                                   item_feature_columns,
                                                                                   l2_reg_embedding, init_std, seed,
                                                                                   embedding_matrix_dict=embedding_matrix_dict)
    item_dnn_input = combined_dnn_input(item_sparse_embedding_list, item_dense_value_list)

    user_dnn_out = DNN(user_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                       dnn_use_bn, seed, )(user_dnn_input)

    item_dnn_out = DNN(item_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                       dnn_use_bn, seed)(item_dnn_input)

    score = Similarity(type=metric)([user_dnn_out, item_dnn_out])

    output = PredictionLayer("binary", False)(score)

    model = Model(inputs=user_inputs_list + item_inputs_list, outputs=output)

    plot_model(model, to_file='dnn.png',show_shapes=True)
    print("go")
    model.__setattr__("user_input", user_inputs_list)
    model.__setattr__("item_input", item_inputs_list)
    model.__setattr__("user_embedding", user_dnn_out)
    model.__setattr__("item_embedding", item_dnn_out)
    return model
#######################################################################################################################
import pandas as pd
import numpy as np
# from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
# from keras.utils import plot_model
samples_data = pd.read_csv("samples.txt", sep="\t", header = None)
samples_data.columns = ["user_id", "gender", "age", "hist_movie_id", "hist_len", "movie_id", "movie_type_id", "label"]
samples_data.head()
# samples_data = shuffle(samples_data)
X = samples_data[["user_id", "gender", "age", "hist_movie_id", "hist_len", "movie_id", "movie_type_id"]]
y = samples_data["label"]
"""
本示例中包含：7个特征。

user端特征有5个，分别为["user_id", "gender", "age", "hist_movie_id", "hist_len"]；
user_id 为 用户ID特征，离散特征，从1-3表示；
gender 为 用户性别特征，离散特征，从1-2表示；
age 为 用户年龄特征，离散特征，从1-3表示；
hist_movie_id 为 用户观看的movie序列特征，根据观看的时间倒排，即最新观看的movieID排在前面；
hist_len 为 用户观看的movie序列长度特征，连续特征；
movie端特征有2个，为 ["movie_id", "movie_type_id"]；
movie_id 为 movieID特征，离散特征，从1-208表示；
movie_type_id 为 movie类型ID特征，离散特征，从1-9表示；
"""
train_model_input = {"user_id": np.array(X["user_id"]), \
           "gender": np.array(X["gender"]), \
           "age": np.array(X["age"]), \
           "hist_movie_id": np.array([[int(i) for i in l.split(',')] for l in X["hist_movie_id"]]), \
           "hist_len": np.array(X["hist_len"]), \
           "movie_id": np.array(X["movie_id"]), \
           "movie_type_id": np.array(X["movie_type_id"])}

train_label = np.array(y)
#######################################################################################################################
import pandas as pd
from deepctr.inputs import SparseFeat, VarLenSparseFeat
from sklearn.preprocessing import LabelEncoder


embedding_dim = 32
SEQ_LEN = 50
user_feature_columns = [SparseFeat('user_id', max(samples_data["user_id"])+1, embedding_dim),
                        SparseFeat("gender", max(samples_data["gender"])+1, embedding_dim),
                        SparseFeat("age", max(samples_data["age"])+1, embedding_dim),
                        VarLenSparseFeat(SparseFeat('hist_movie_id', max(samples_data["movie_id"])+1, embedding_dim,
                                                    embedding_name="movie_id"), SEQ_LEN, 'mean', 'hist_len'),
                        ]

item_feature_columns = [SparseFeat('movie_id', max(samples_data["movie_id"])+1, embedding_dim),
                       SparseFeat('movie_type_id', max(samples_data["movie_type_id"])+1, embedding_dim)]
#模型训练
model = DSSM(user_feature_columns, item_feature_columns)

model.compile(optimizer='adagrad', loss="binary_crossentropy", metrics=['accuracy'])
model.summary()

# plot_model(model, 'doublednn.png')
history = model.fit(train_model_input, train_label,
                    batch_size=256, epochs=10, verbose=1, validation_split=0.2, )
#######################################################################################################################
#获取loss曲线
import matplotlib.pyplot as plt
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('./loss.png')
#获取user item向量
test_user_model_input = {"user_id": np.array(X["user_id"]), \
           "gender": np.array(X["gender"]), \
           "age": np.array(X["age"]), \
           "hist_movie_id": np.array([[int(i) for i in l.split(',')] for l in X["hist_movie_id"]]), \
           "hist_len": np.array(X["hist_len"])}

test_item_model_input = {"movie_id": np.array(X["movie_id"]), \
           "movie_type_id": np.array(X["movie_type_id"])}

user_embedding_model = Model(inputs=model.user_input, outputs=model.user_embedding)
item_embedding_model = Model(inputs=model.item_input, outputs=model.item_embedding)

user_embs = user_embedding_model.predict(test_user_model_input, batch_size=2 ** 12)
item_embs = item_embedding_model.predict(test_item_model_input, batch_size=2 ** 12)
print(user_embs)



