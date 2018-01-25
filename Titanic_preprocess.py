
#coding=utf-8
import matplotlib.pyplot as plt
import numpy as np


def preprocess(train_df, test_df):
    #Overview the data
    train_df.head()
    test_df.head()
    train_df.info()
    test_df.info()
    
    #Remove useless columns in train_df and test_df
    train_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1, inplace = True)
    test_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1, inplace = True)
    
    #Age preprocess
    #Subplots
    fig, ([axis1, axis2], [axis3, axis4]) = plt.subplots(2, 2, figsize = (15, 10))
    axis1.set_title('train_df Age original')
    axis2.set_title('train_df Age afterwards')
    axis3.set_title('test_df Age original')
    axis4.set_title('test_df Age afterwards')
    #Plot the Age column before data-preprocessing
    train_df.Age.dropna().hist(bins = 60, ax = axis1)
    test_df.Age.dropna().hist(bins = 60, ax = axis3)
    #Fill the NAs in Age column for both train_df and test_df
    train_age_mean = train_df.Age.mean()
    test_age_mean = test_df.Age.mean()
    train_age_std = train_df.Age.std()
    test_age_std = test_df.Age.std()
    train_age_nan_len = train_df.Age.isnull().sum()
    test_age_nan_len = test_df.Age.isnull().sum()
    train_rand = np.random.randint(train_age_mean - train_age_std, train_age_mean + train_age_std, size =                                   train_age_nan_len)
    test_rand = np.random.randint(test_age_mean - test_age_std, test_age_mean + test_age_std, size =                                   test_age_nan_len)
    train_df.Age[np.isnan(train_df.Age)] = train_rand
    test_df.Age[np.isnan(test_df.Age)] = test_rand
    #Plot the Age column after data-preprocessing
    train_df.Age.hist(bins = 60, ax = axis2)
    test_df.Age.hist(bins = 60, ax = axis4)
    plt.show()
    
    #Embarked preprocess
    #Subplots
    fig, ([axis1, axis2], [axis3, axis4]) = plt.subplots(2, 2, figsize = (15, 10))
    axis1.set_title('train_df Embarked original')
    axis2.set_title('train_df Embarked afterwards')
    axis3.set_title('test_df Embarked original')
    axis4.set_title('test_df Embarked afterwards')
    #Plot the Embarked column before data-preprocessing
    train_df.Embarked.dropna().hist(bins = 60, ax = axis1)
    test_df.Embarked.dropna().hist(bins = 60, ax = axis3)
    #Fill the NAs in the Embarked column by the most frequent one, which is 'S'
    train_df.Embarked.fillna('S', inplace = True)
    test_df.Embarked.fillna('S', inplace = True)
    #Plot the Embarked column after data-preprocessing
    train_df.Embarked.hist(bins = 60, ax = axis2)
    test_df.Embarked.hist(bins = 60, ax = axis4)
    plt.show()
    #Or we can just drop the Embarked column
    train_df.drop(['Embarked'], axis = 1, inplace = True)
    test_df.drop(['Embarked'], axis = 1, inplace = True)
    
    #Fare
    #Fill in the one NA in the Fare column of test_df
    test_df.Fare.fillna(test_df.Fare.dropna().mean(), inplace = True)
    
    #Sex
    #male = 1, female = 0
    train_df.Sex[train_df.Sex == 'male'] = 1
    train_df.Sex[train_df.Sex == 'female'] = 0
    test_df.Sex[test_df.Sex == 'male'] = 1
    test_df.Sex[test_df.Sex == 'female'] = 0
    
    #Data post-process
    print('--train_df--')
    print(train_df.info())
    print('\n')
    print('--test_df--')
    print(test_df.info())
    return(train_df, test_df)

