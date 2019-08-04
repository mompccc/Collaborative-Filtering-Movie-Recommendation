import pandas as pd
import time
import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction import DictVectorizer
from user import get_user_dict
from scipy.stats import pearsonr


def get_item_dict(df):
    isbn_dict = defaultdict(dict)
    isbn_group = df.groupby('ISBN')
    for s in isbn_group:
        isbn = s[0]
        temp_df = s[1]
        user_list = list(temp_df['User-ID'])
        rating_list = list(temp_df['Book-Rating'])
        for i in range(len(user_list)):
            isbn_dict[isbn][user_list[i]] = rating_list[i]
    return isbn_dict


def cos_sim(item_dict1, item_dict2):
    set_all = set(item_dict1) | set(item_dict2)
    set_and = set(item_dict1) & set(item_dict2)
    if len(set_and)/len(set_all) <= Threshold:
        return 0
    d = [item_dict1, item_dict2]
    v = DictVectorizer(sparse=False)
    x = v.fit_transform(d)
    score = cosine_similarity(x)[0][1]
    #score = pearsonr(x[0], x[1])[0]
    return score


def get_item_sim(item_dict):
    item_sim = defaultdict(dict)
    temp_dict = set()
    for key1 in item_dict:
        for key2 in item_dict:
            if key2 not in temp_dict:
                score = cos_sim(item_dict[key1], item_dict[key2])
                if score > 0 and key1 != key2:
                    item_sim[key1][key2] = score
        temp_dict.add(key1)
    return item_sim


def get_recommendation(user_id):
    user_dict = User_Dict[user_id]
    temp = Item_Sim.copy()
    temp[0] = user_dict
    movie_name = list(temp.keys())
    v = DictVectorizer(sparse=False)
    d = list(temp.values())
    x = v.fit_transform(d)

    user_dict_hot = x[-1]
    matrix_1 = x * user_dict_hot
    for i in range(len(user_dict_hot)):
        if user_dict_hot[i] != 0:
            user_dict_hot[i] = 1
    matrix_2 = x * user_dict_hot
    a = [matrix_1[i].sum() for i in range(matrix_1.shape[0])]
    b = [matrix_2[i].sum() for i in range(matrix_2.shape[0])]
    result = np.array(a)/np.array(b)
    recom_dict = {}
    for i in range(result.shape[0]-1):
        if not np.isnan(result[i]):
            recom_dict[movie_name[i]] = result[i]
    #print(recom_dict)

    rmse_list = []
    test_dict = Test_Dict[user_id]
    for key in recom_dict:
        if key in test_dict:
            rmse_list.append(recom_dict[key] - test_dict[key])
    if not rmse_list:
        return -1, -1, -1
    precision = len(rmse_list) / len(recom_dict)
    recall = len(rmse_list) / len(test_dict)
    rmse = np.sqrt(np.mean(np.array(rmse_list) ** 2))
    return rmse, precision, recall


if __name__ == '__main__':
    Threshold = 0.1
    # BX_Book_Rating = pd.read_csv('BX-CSV-Dump/BX-Book-Ratings.csv', encoding='ISO-8859-1', sep=";", nrows=10000)
    DF_Name = ['User-ID', 'ISBN', 'Book-Rating', 'TimeStamp']
    MovieLens_Rating = pd.read_csv('MovieLens/ml-100k/ub.base', sep="\t", header=None, names=DF_Name)
    MovieLens_Rating.drop(columns='TimeStamp', inplace=True)
    Test_Rating = pd.read_csv('MovieLens/ml-100k/ub.test', sep="\t", header=None, names=DF_Name)
    Test_Rating.drop(columns='TimeStamp', inplace=True)

    t0 = time.time()
    User_Dict = get_user_dict(MovieLens_Rating)
    Item_Dict = get_item_dict(MovieLens_Rating)
    Test_Dict = get_user_dict(Test_Rating)
    t1 = time.time()
    print("time for get_item_dict: {:.2f}".format(t1 - t0))

    Item_Sim = get_item_sim(Item_Dict)
    t2 = time.time()
    print("time for get_item_sim: {:.2f}".format(t2-t1))

    User_RMSE = {}
    User_Precision = {}
    User_Recall = {}
    for user in User_Dict:
        temp_rmse, temp_precision, temp_recall = get_recommendation(user)
        if temp_rmse != -1:
            User_RMSE[user] = temp_rmse
            User_Precision[user] = temp_precision
            User_Recall[user] = temp_recall
    print("RMSE:")
    print(np.array(list(User_RMSE.values())).mean())
    print("Precision:")
    print(np.array(list(User_Precision.values())).mean())
    print("Recall:")
    print(np.array(list(User_Recall.values())).mean())
    t3 = time.time()
    print("time fore get_recommendation: {:.2f}".format(t3 - t2))