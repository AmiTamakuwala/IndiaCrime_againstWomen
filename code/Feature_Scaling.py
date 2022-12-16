from source import State_UT_crime_df, crime_data, crime_in_years_df

# Featre Scaling:
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import warnings
from sklearn import linear_model
from sklearn.model_selection import train_test_split
warnings.simplefilter(action="ignore", category=FutureWarning)

if __name__ == "__main__":
    crime_data = pd.read_csv("D:\\Crime_against_Women\\42_District_wise_crimes_committed_against_women_2001_2012.csv")
    print(crime_data)
    print(State_UT_crime_df.head())


    x = State_UT_crime_df.iloc[:, 1:].values
    print("\n Original data values: \n", x)

    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))

    # Scaled feature:

    x_after_min_max_scaler = min_max_scaler.fit_transform(x)
    print("\n After min_max Scaling: \n", x_after_min_max_scaler)
    
    # Standardization:
    Standardisation = preprocessing.StandardScaler()

    # Scaled feature:
    x_after_Standerdisation = Standardisation.fit_transform(x)
    print("\n After Standardisation: \n", x_after_Standerdisation)

    print(State_UT_crime_df.describe())

    State_UT_crime_df.hist(bins=50, figsize=(20,15))
    plt.show()

    print(State_UT_crime_df.columns)

    # One-hot-encoding:

    one_hot_encoded_data = pd.get_dummies(State_UT_crime_df, columns = ['Rape', 'Kidnapping and Abduction', 'Dowry Deaths',
       'Assault on women with intent to outrage her modesty',
       'Insult to modesty of Women', 'Cruelty by Husband or his Relatives',
       'Importation of Girls'])
    print(one_hot_encoded_data)

    New_df = State_UT_crime_df.join(one_hot_encoded_data)
    print(New_df.head())

    print(State_UT_crime_df.corr())

    sns.heatmap(State_UT_crime_df.corr())
    plt.show()

    New_df.reset_index(inplace=True)
    print(New_df)

    print(New_df.iloc[:36, :])

    print(New_df.head())
    print(New_df.columns)

    X = New_df.drop(['STATE/UT'], 1)
    print(X.head())

    y = New_df["STATE/UT"]
    print(y.head())

    X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.8, test_size=0.2, random_state=50)

    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    scaler = MinMaxScaler()

    X_train[['Rape', 'Kidnapping and Abduction', 'Dowry Deaths',
       'Assault on women with intent to outrage her modesty',
       'Insult to modesty of Women', 'Cruelty by Husband or his Relatives',
       'Importation of Girls']] = scaler.fit_transform(X_train[['Rape', 'Kidnapping and Abduction', 'Dowry Deaths',
       'Assault on women with intent to outrage her modesty',
       'Insult to modesty of Women', 'Cruelty by Husband or his Relatives',
       'Importation of Girls']])

    print(X_train.head())

    print(New_df.corr())

    #############    LINEAR REGRESSION      ############     

    print(crime_in_years_df.head())

    ########    1. "Rape Cases"    ############
    xs = crime_in_years_df.iloc[:, 0]
    ys = crime_in_years_df.iloc[:, 1]

    X_train, X_test, y_train, y_test = train_test_split(xs, ys, test_size=0.3, random_state=0)
    print("Year\n" + str(X_train) + "\n\nRape\n" + str(y_train))

    len(xs), len(ys)

    # Sctter plot for "Rape" cases for 2001-2012:

    fig = plt.figure(figsize=(12,10))
    plt.scatter(xs, ys)
    plt.ylabel("Rape crime Rate")
    plt.xlabel("Year")
    plt.show()

    # intercept caluculation:

    def slope_intercept(x_val, y_val):
        """ calculation of intercept"""
        x = np.array(x_val)
        y = np.array(y_val)
        m = (((np.mean(x) * np.mean(y)) - np.mean(x*y)) /
        ((np.mean(x) * np.mean(x)) - np.mean(x*x)))
        m = round(m,2)
        b = (np.mean(y) - np.mean(x) * m)
        b = round(b,2)
        
        return m, b
    print(slope_intercept(xs, ys))

    # creating regression line for the "Rape crime rate":

    fig = plt.figure(figsize=(12,10))
    m,b = slope_intercept(xs, ys)
    reg_line = [(m*x)+b for x in xs]
    plt.scatter(xs, ys, color = "brown")
    plt.plot(xs, reg_line)
    plt.ylabel("Rape Crime Rate")
    plt.xlabel("Year")
    plt.title("Making a regression line")
    # plt.show()

    ######  "Rape" cases after prediction:

    X1 = crime_in_years_df.iloc[:, :-7].values
    y1 = crime_in_years_df.iloc[:, 1].values

    from sklearn.model_selection import train_test_split
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.3, random_state=0)

    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(X1_train, y1_train)

    print("Regression Intercept: " + str(m))
    print("Regression Coefficient: " + str(b))

    print("Years:")
    for x in np.nditer(xs):
        print(x)

    print("\n\n Rape Values:")

    for x in np.nditer(ys):
        print(x)

    # prediction for the next 5 years:

    years = np.array([2013, 2014, 2015, 2016, 2017, 2018])
    predict_crime_in_years = pd.DataFrame(years)
    print(predict_crime_in_years)
    ypredd = (m * years + b)

    print("\n\n Rape Values will be....")

    for x in np.nditer(ypredd):
        print(int(x)) 

    # merging the values and year with predicted values and year.

    merge_list = np.concatenate((xs, years), axis=0)
    print("Years: " + str(merge_list))


    reg_line = [(m*x)+b for x in merge_list]

    transform_list = np.concatenate((ys, ypredd), axis=0)
    print("Rape: "+str(transform_list))

    fig = plt.figure(figsize=(15,12))
    plt.scatter(merge_list, transform_list, color = "Green")
    plt.plot(merge_list, reg_line)
    plt.xlim((2001, 2018))
    plt.ylabel("Rape Crime Rate")
    plt.xlabel("Years")
    plt.title("Rape Crime regression Line from 2001 to 2018")
    plt.show()  

    ######-- 2. Kidnapping & Abduction Cases --############

    xs1 = crime_in_years_df.iloc[:, 0]
    ys1 = crime_in_years_df.iloc[:, 2]

    X_train1, X_test1, y_train1, y_test1 = train_test_split(xs1, ys1, test_size=0.3, random_state=0)
    print(str(X_train1) + "\n\n" + str(y_train1))

    plt.scatter(X_train1, y_train1)
    plt.ylabel("Kidnapping & Abduction Crime Reate")
    plt.xlabel("Year")
    plt.show()

    print(slope_intercept(xs1, ys1))

    # making regression line:

    m1, b1 = slope_intercept(xs1, ys1)
    reg_line = [(m1 * x) + b1 for x in xs1]
    plt.scatter(xs1, ys1, color = "green")
    plt.plot(xs1, reg_line)
    plt.ylabel("Kidnapping & Abduction Crime Rate")
    plt.xlabel("Year")
    plt.title("Regression line for Kidnapping & Abduction")
    plt.show()

    print("Regression Intercept: " + str(m1))
    print("Regression Coefficient: " + str(b1))

    print("Year")

    for x in np.nditer(xs1):
        print(x)
        
    print("\n\nKidnapping & Abduction")

    for x in np.nditer(ys1):
        print(x)

#### prediction of next five years for "Kidnapping & Abduction" crime column:

    years = np.array([2013,2014, 2015, 2016, 2017, 2018])
    predict_crime_in_years1 = pd.DataFrame(years)
    print(predict_crime_in_years1)

    ypred_ = (m1 * years + b1)

    print("Predicted Kidnapping Rate")

    for x in np.nditer(ypred_):
        print(int(x))

     # merging the values and year with predicted values and year.

    merge_list1 = np.concatenate((xs1, years), axis=0)
    print("Years: " + str(merge_list1))


    reg_line = [(m1*x)+b1 for x in merge_list1]

    transform_list1 = np.concatenate((ys1, ypred_), axis=0)
    print("Kidnapping & Abduction: "+str(transform_list1))

    fig = plt.figure(figsize=(15,12))
    plt.scatter(merge_list1, transform_list1, color = "Green")
    plt.plot(merge_list1, reg_line)
    plt.xlim((2001, 2018))
    plt.ylabel("Kidnapping & Abduction Crime Rate")
    plt.xlabel("Years")
    plt.title("Kidnapping & abduction Crime regression Line from 2001 to 2018")
    plt.show()   

    





































