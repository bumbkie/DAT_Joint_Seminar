import pandas as pd

def term_explain_preprocessing(csv_file):
    df1 = pd.read_csv(csv_file)
    df1 = df1.dropna()

    #용어에 대한 설명이 10개 이상인 용어만 사용
    count = df1['용어'].value_counts()
    over10 = list(count[count >= 10].index)

    df = df1[df1['용어'].isin(over10)]

    #one-hot encoding을 위한 mapping
    label_list = list(df['용어'].unique())
    label_mapping = {}
    result_dict = {}
    for idx, word in enumerate(label_list):
        label_mapping[term] = idx
        result_dict[idx] = term
        
    return df, label_mapping, result_dict
