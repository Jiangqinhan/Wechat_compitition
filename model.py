from deepctr.feature_column import SparseFeat, VarLenSparseFeat, DenseFeat, get_feature_names
from deepctr.models import DeepFM
# from lightgbm
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import time
import pickle


epochs = 5
batch_size = 64
embedding_dim = 16
ROOT_PATH = "../data"
# 比赛数据集路径
DATASET_PATH = os.path.join(ROOT_PATH, "wechat_algo_data1")
# 训练集
USER_ACTION = os.path.join(DATASET_PATH, "user_action.csv")
FEED_INFO = os.path.join(DATASET_PATH, "feed_info.csv")
FEED_EMBEDDINGS = os.path.join(DATASET_PATH, "feed_embeddings.csv")
# 测试集
TEST_FILE = os.path.join(DATASET_PATH, "test_a.csv")
END_DAY = 15
SEED = 2021

# 初赛待预测行为列表
ACTION_LIST = ["read_comment", "like", "click_avatar",  "forward"]
# 复赛待预测行为列表
# ACTION_LIST = ["read_comment", "like", "click_avatar",  "forward", "comment", "follow", "favorite"]
# 用于构造特征的字段列表
FEA_COLUMN_LIST = ["read_comment", "like", "click_avatar",  "forward", "comment", "follow", "favorite"]
# 负样本下采样比例(负样本:正样本)
ACTION_SAMPLE_RATE = {"read_comment": 5, "like": 5, "click_avatar": 5, "forward": 10, "comment": 10, "follow": 10, "favorite": 10}
# 各个阶段数据集的设置的最后一天
STAGE_END_DAY = {"online_train": 14, "offline_train": 12, "evaluate": 13, "submit": 15}
# 各个行为构造训练数据的天数
ACTION_DAY_NUM = {"read_comment": 5, "like": 5, "click_avatar": 5, "forward": 5, "comment": 5, "follow": 5, "favorite": 5}

def main(stage):
    '''
    目前用到的特征有
    ['userid', 'feedid', 'device', 'authorid', 'bgm_song_id',
     'bgm_singer_id', 'videoplayseconds', 'read_comment', 'read_commentsum',
     'likesum', 'click_avatarsum', 'forwardsum', 'commentsum', 'followsum',
     'favoritesum', 'read_commentsum_user', 'likesum_user',
     'click_avatarsum_user', 'forwardsum_user', 'commentsum_user',
     'followsum_user', 'favoritesum_user']
    '''
    sparse_features = ['userid', 'feedid', 'device', 'authorid', 'bgm_song_id',
                       'bgm_singer_id']
    #sparse_features_vocab_size={'user_id':20000}

    dense_features = ['videoplayseconds']
    '''
        , 'read_commentsum',
                      'likesum', 'click_avatarsum', 'forwardsum', 'commentsum', 'followsum',
                      'favoritesum', 'read_commentsum_user', 'likesum_user',
                      'click_avatarsum_user', 'forwardsum_user', 'commentsum_user',
                      'followsum_user', 'favoritesum_user']
    '''
    test_file_name = "{stage}_{action}_{day}_concate_sample.csv".format(stage='submit', action="all",
                                                                        day=STAGE_END_DAY['submit'])
    submit_dir = os.path.join(ROOT_PATH, 'submit', test_file_name)
    predict_dict = {}
    for action in ACTION_LIST:
        stage_dir=""
        if stage in ["offline_train","online_train"]:
            file_name = "{stage}_{action}_{day}_concate_sample.csv".format(stage=stage, action=action,
                                                                           day=STAGE_END_DAY[stage])
            stage_dir = os.path.join(ROOT_PATH, stage, file_name)
        df = pd.read_csv(stage_dir)
        dense_feature_columns = [DenseFeat(feat, 1, ) for feat in dense_features]
        sparse_feature_columns = [SparseFeat(feat,df[feat].max()+1, embedding_dim) for feat in sparse_features]
        fixlen_feature_columns = sparse_feature_columns + dense_feature_columns
        dnn_feature_columns=fixlen_feature_columns
        linear_feature_columns=fixlen_feature_columns
        feature_names=get_feature_names(linear_feature_columns+dense_feature_columns)
        model_input={name:df[name] for name in feature_names}
        model_label=df[action]
        model=DeepFM(linear_feature_columns,dnn_feature_columns)
        model.compile("adagrad",loss='binary_crossentropy')
        for epoch in range(epochs):
            history=model.fit(model_input,model_label,batch_size=batch_size)
        # =====================predict part==============================

        test_data=pd.read_csv(submit_dir)
        test_input={name:test_data[name] for name in feature_names}
        logits=[x[0] for x in model.predict(test_input)]
        predict_dict[action]=logits


    submit=pd.read_csv('../data/wechat_algo_data1/test_a.csv')[['userid','feedid']]
    with open('../data/wechat_algo_data1/id_index_map.pkl', 'rb') as f:
        feed_id_map=pickle.load(f)
        user_id_map=pickle.load(f)
    actions = pd.DataFrame.from_dict(predict_dict)
    res = pd.concat([submit, actions], sort=False, axis=1)
    res['userid']=res['userid'].map(lambda x:user_id_map[x])
    res['feedid']=res['feedid'].map(lambda x:feed_id_map[x])
    file_name = "submit_" + str(int(time.time())) + ".csv"
    submit_file = os.path.join(ROOT_PATH, 'submit', file_name)
    print('Save to: %s' % submit_file)
    res.to_csv(submit_file, index=False)


if __name__ == '__main__':
    main('online_train')
