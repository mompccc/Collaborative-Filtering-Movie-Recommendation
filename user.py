import pandas as pd
import time
import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import scale
from scipy.stats import pearsonr


def get_user_dict(df):
    user_dict = defaultdict(dict)
    user_group = df.groupby('User-ID')
    for s in user_group:
        user = s[0]
        temp_df = s[1]
        isbn_list = list(temp_df['ISBN'])
        rating_list = list(temp_df['Book-Rating'])
        for i in range(len(isbn_list)):
            user_dict[user][isbn_list[i]] = rating_list[i]
    return user_dict


def cos_sim(user_dict1, user_dict2):
    set_all = set(user_dict1) | set(user_dict2)
    set_and = set(user_dict1) & set(user_dict2)
    if len(set_and)/len(set_all) <= Threshold:
        return 0
    d = [user_dict1, user_dict2]
    v = DictVectorizer(sparse=False)
    x = v.fit_transform(d)
    #x[0], x[1] = scale(x[0]), scale(x[1])
    score = cosine_similarity(x)[0][1]
    #score = pearsonr(x[0], x[1])[0]
    return score


def get_user_sim(user_dict):
    user_sim = defaultdict(dict)
    temp_dict = set()
    for key1 in user_dict:
        for key2 in user_dict:
            if key2 not in temp_dict:
                score = cos_sim(user_dict[key1], user_dict[key2])
                if score > 0 and key1 != key2:
                    user_sim[key1][key2] = score
        temp_dict.add(key1)
    return user_sim


def get_recommendation(user_id):
    top_sim = [0]
    temp_matrix = {user_id: User_Dict[user_id]}
    temp_sorted = sorted(User_Sim[user_id].items(), key=lambda x: x[1], reverse=True)
    for i in range(min(Top_K, len(temp_sorted))):
        temp_matrix[temp_sorted[i][0]] = User_Dict[temp_sorted[i][0]]
        top_sim.append(temp_sorted[i][1])
    #print(top_sim)

    v = DictVectorizer(sparse=False)
    d = list(temp_matrix.values())
    x = v.fit_transform(d)
    feature_names = v.get_feature_names()

    recom_dict = {}
    for i in range(x.shape[1]):
        if x[0][i] == 0:
            sum_sim = 0
            for j in range(x.shape[0]):
                if x[j][i] != 0:
                    sum_sim += top_sim[j]
            result = np.matmul(x[..., i], np.array(top_sim)) / sum_sim
            recom_dict[feature_names[i]] = round(result, 2)
    #recom_sorted = sorted(recom_dict.items(), key=lambda x: x[1], reverse=True)
    #print(recom_sorted)
    rmse_list = []
    test_dict = Test_Dict[user_id]
    for key in recom_dict:
        if key in test_dict:
            rmse_list.append(recom_dict[key] - test_dict[key])
    if not rmse_list:
        return -1, -1, -1
    precision = len(rmse_list)/len(recom_dict)
    recall = len(rmse_list)/len(test_dict)
    rmse = np.sqrt(np.mean(np.array(rmse_list)**2))
    return rmse, precision, recall


if __name__ == '__main__':
    Threshold = 0.1
    Top_K = 10
    # BX_Book_Rating = pd.read_csv('BX-CSV-Dump/BX-Book-Ratings.csv', encoding='ISO-8859-1', sep=";", nrows=10000)
    DF_Name = ['User-ID', 'ISBN', 'Book-Rating', 'TimeStamp']
    MovieLens_Rating = pd.read_csv('MovieLens/ml-100k/u1.base', sep="\t", header=None, names=DF_Name)
    MovieLens_Rating.drop(columns='TimeStamp', inplace=True)
    Test_Rating = pd.read_csv('MovieLens/ml-100k/u1.test', sep="\t", header=None, names=DF_Name)
    Test_Rating.drop(columns='TimeStamp', inplace=True)

    t0 = time.time()
    User_Dict = get_user_dict(MovieLens_Rating)
    Test_Dict = get_user_dict(Test_Rating)
    t1 = time.time()
    #print(User_Dict)
    print("time for get_user_dict: {:.2f}".format(t1 - t0))

    User_Sim = get_user_sim(User_Dict)
    t2 = time.time()
    #print(User_Sim)
    print("time for get_user_sim: {:.2f}".format(t2-t1))

    User_RMSE = {}
    User_Precision = {}
    User_Recall = {}
    for user in User_Sim:
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