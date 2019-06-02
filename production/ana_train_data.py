
import sys
import os
import numpy as np
import pandas as pd
import operator


def get_input(input_train_file, input_test_file):
    '''
    Return:
        pd.Dataframe train_file
        pd.Dataframe test_file
    '''
    dtype_dict = {'age':np.int32,
                  'education-num':np.int32,
                  'captial-gain':np.int32,
                  'hours-per-week':np.int32,
                  'captial-gain':np.int32}
    use_list = list(range(15))
    use_list.remove(2)
    train_data_df = pd.read_csv(input_train_file, sep=",", header=0, dtype=dtype_dict, na_values=' ?', keep_default_na=False,usecols=use_list)
    #print(train_data_df.shape)
    train_data_df = train_data_df.dropna(axis=0, how='any')
    #print(train_data_df.shape)
    test_data_df = pd.read_csv(input_test_file, sep=",", header=0, dtype=dtype_dict, na_values=" ?", keep_default_na=False, usecols=use_list)
    test_data_df = test_data_df.dropna(axis=0, how='any')
    return train_data_df, test_data_df

def label_trans(x):
    '''
    Args:
        x: each element in fix col value
    '''
    if x == ' <=50K' or x == '<=50K.':
        return '0'
    if x == ' >50K' or x == ' >50K.':
        return '1'
    return 0

def process_label_feature(label_feature_str, df_in):
    '''
    Args:
        label_feature_str: label
        df_in: dataframe
    '''
    df_in.loc[:,label_feature_str] = df_in.loc[:,label_feature_str].apply(label_trans)

def dict_trans(dict_in):
    '''
    Args:
        dict_in: key str, value int
    Return:
        a dick key str value index
    '''
    output_dict = {}
    index = 0
    for zuhe in sorted(dict_in.items(), key=operator.itemgetter(1), reverse=True):
        output_dict[zuhe[0]] = index
        index += 1
    return output_dict

def dis_to_feature(x, feature_dict):
    '''
    Args:
        x: element
        feature_dict: pos dict
    Return:
        a str e.g '1,0,0'
    '''
    output_list = [0]*len(feature_dict)
    if x not in feature_dict:
        return ','.join(str(ele) for ele in output_list)
    else:
        index = feature_dict[x]
        output_list[index] = 1
    return ','.join(str(ele) for ele in output_list)


def process_dis_feature(feature_str, df_train, df_test):
    '''
    Args:
        feature_str: label in
        df_train: train_data_df
        df_test: test_data_df
    Return:
        the dim
    '''
    origin_dict = df_train.loc[:,feature_str].value_counts().to_dict()
    feature_dict = dict_trans(origin_dict)
    #print(feature_dict)
    df_train.loc[:,feature_str] = df_train.loc[:,feature_str].apply(dis_to_feature, args=(feature_dict,))
    df_test.loc[:,feature_str] = df_test.loc[:,feature_str].apply(dis_to_feature, args=(feature_dict,))
    #print(df_train.loc[:3,feature_str])
    return len(feature_dict)

def list_train(input_dict):
    '''
    Args:
        dict_in: eg.{'mean': 38.437901995888865, 'min': 17.0, 'count': 30162.0, '75%': 47.0, '25%': 28.0, 'max': 90.0, '50%': 37.0, 'std': 13.134664776855985}
    Retrun:
    a list [0,1,2,3,4]
    '''
    output_list = [0]*5
    key_list = ['min','25%','50%','75%','max']
    for index in range(len(key_list)):
        fix_key = key_list[index]
        if fix_key not in input_dict:
            print(fix_key)
            print(input_dict)
            print('error')
            sys.exit()
        else:
            output_list[index] = input_dict[fix_key]
    return output_list

def con_to_feature(x, feature_list):
    '''
    Args:
        x: element
        feature_list: list for feature trans
    :Return:
        str: 1,0,0,0,0
    '''
    feature_len = len(list(feature_list)) - 1
    result = [0] * feature_len
    for index in range(feature_len):
        if x >= feature_list[index] and x<=feature_list[index+1]:
            result[index] = 1
            return ','.join(str(ele) for ele in result)

    return ','.join(str(ele) for ele in result)

def process_con_feature(feature_str, df_train, df_test):
    '''
    Args:
        feature_str: feature string
        df_train: train_data_df
        df_test: test_data_df
    Return:
        the dim of feature output
    '''
    origin_dict = df_train.loc[:,feature_str].describe().to_dict()
    feature_list =  list_train(origin_dict)
    df_train.loc[:,feature_str] = df_train.loc[:,feature_str].apply(con_to_feature, args=(feature_list,))
    df_test.loc[:, feature_str] = df_test.loc[:, feature_str].apply(con_to_feature, args=(feature_list,))
    #print(feature_list)
    #print(df_train.loc[:3,feature_str])
    return len(feature_list)-1

def output_file(df_in, out_file):
    '''
    Args:
        df_in:
        out_file:
     write data of dataframe to out_file
    '''
    fw = open(out_file, 'w+')
    for row_index in df_in.index:
        outline = ','.join([str(ele) for ele in df_in.loc[row_index].values])
        fw.write(outline + '\n')
    fw.close()

def add(str_one, str_two):
    '''
    Args:
        str_one:"0,0,1,0"
        str_two:'1,0,0,0'
    Return:
        str such as '0,0,1,0'
    '''
    list_one = str_one.split(',')
    list_two = str_two.split(',')
    list_one_len = len(list_one)
    list_two_len = len(list_two)
    return_list = [0] * (list_one_len * list_two_len)
    try:
        index_one = list_one.index['1']
    except:
        index_one = 0

    try:
        index_two = list_two.index['1']
    except:
        index_two = 0
    return_list[index_one * list_two_len + index_two] = 1
    return ','.join(str(ele) for ele in return_list)

def combine_feature(feature_one, feature_two, new_feature, train_data_df, test_data_df, feature_num_dict):
    '''
    Args:
        feature_one:
        feature_two:
        new_feature: combine feature name
        train_data_df:
        test_data_df:
        feature_num_dict: ndim of every feature, key feature name value len of the dim
    Return:
        new_feature_num
    '''
    train_data_df[new_feature] = train_data_df.apply(lambda row:add(row[feature_one],row[feature_two]), axis=1)
    test_data_df[new_feature] = test_data_df.apply(lambda row:add(row[feature_one],row[feature_two]), axis=1)
    if feature_one not in feature_num_dict:
        print('error')
        sys.exit()
    if feature_two not in feature_num_dict:
        print('error')
        sys.exit()
    return feature_num_dict[feature_one] * feature_num_dict[feature_two]

def ana_train_data(input_train_data,
                   input_test_data,
                   output_train_file,
                   output_test_file,
                   feature_num_file
                   ):
    '''
    '''
    train_data_df, test_data_df = get_input(input_train_data, input_test_data)
    label_feature_str = 'income'
    process_label_feature(label_feature_str, train_data_df)
    process_label_feature(label_feature_str, test_data_df)

    dis_feature_list = ['workclass','education','marital-status','occupation','relationship','race','sex','native-country']
    con_feature_list = ['age','education-num','captial-gain','captial-loss','hours-per-week']
    index_list = ['age','workclass','education','education-num','marital-status','occupation','relationship','race'
                  ,'sex','captial-gain','captial-loss','hours-per-week','native-country']
    dis_feature_num = 0
    con_feature_num = 0
    for dis_feature in dis_feature_list:
        tmp_feature_num = process_dis_feature(dis_feature, train_data_df, test_data_df)
        dis_feature_num += tmp_feature_num
    for con_feature in con_feature_list:
        con_feature_num += 1

    output_file(train_data_df, output_train_file)
    output_file(test_data_df, output_test_file)
    fw = open(feature_num_file,'w+')
    fw.write('feature_num=' + str(dis_feature_num + con_feature_num))
    fw.close()

if __name__=='__main__':
    ana_train_data('../data/train.txt','../data/test.txt','../data/train_file','../data/test_file','../data/feature_num')
