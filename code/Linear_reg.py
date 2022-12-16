import warnings
from sklearn import linear_model
from sklearn.model_selection import train_test_split
warnings.simplefilter(action="ignore", category=FutureWarning)

from source import crime_in_years_df
from Feature_Scaling import * 

if __name__ == "__main__":
    print(crime_in_years_df.head())
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

   ### ----------------- 3. Dowry Deaths Cases -----------------------------
    xs2 = crime_in_years_df.iloc[:, 0]
    ys2 = crime_in_years_df.iloc[:, 3]

    X_train2, X_test2, y_train2, y_test2 = train_test_split(xs2, ys2, test_size=0.3, random_state=0)
    print(str(X_train2) + "\n\n" + str(y_train2))

    plt.scatter(X_train2, y_train2)
    plt.ylabel("Dowery Deaths")
    plt.xlabel("Year")
    plt.title("Dowery Deaths before prediction")
    plt.show()

    # Intercept value:

    print("Intercept values:", slope_intercept(xs2, ys2))

# making linear regression line before prediction:

    m2, b2 = slope_intercept(xs2, ys2)
    reg_line = [(m2*x) + b2 for x in xs2]
    fig = plt.figure(figsize=(15,12))
    plt.scatter(xs2, ys2, color = "brown")
    plt.plot(xs2, reg_line)
    plt.ylabel("Dowery Deaths Crime Rate")
    plt.xlabel("Years")
    plt.title("Dowery death regression line before prediction")
    plt.show()

    # now. looking for intercept and coefficient cvalues:

    print("Regression intercept: " + str(m2))
    print("Regression coefficient: "+ str(b2))

    print("Year")

    for x in np.nditer(xs2):
        print(x)

    print("\n\n Dowery Deaths values")

    for x in np.nditer(ys2):
        print(x)

#### prediction of next five years for "Dowery Deaths" crime column:

    years = np.array([2013, 2014, 2015, 2016, 2017, 2018])
    predict_crime_in_years2 = pd.DataFrame(years)
    print(predict_crime_in_years2)
    ypredd__ = (m2 * years + b2)

    print("Predicted Dowery Deaths Value")

    for x in np.nditer(ypredd__):
        print(int(x)) 

 # merging the values and year with predicted values and year.

    merge_list2 = np.concatenate((xs2, years), axis=0)
    print("Years: " + str(merge_list2))


    reg_line = [(m2*x)+b2 for x in merge_list2]

    transform_list2 = np.concatenate((ys2, ypredd__), axis=0)
    print("Dowery Deaths "+str(transform_list2))

    fig = plt.figure(figsize=(15,12))
    plt.scatter(merge_list2, transform_list2, color = "Green")
    plt.plot(merge_list2, reg_line)
    plt.xlim((2001, 2018))
    plt.ylabel("Dowery Deaths Crime Rate")
    plt.xlabel("Years")
    plt.title("Dowery Deaths Crime regression Line from 2001 to 2018")
    plt.show()       

# ----------------- 4. Assault on women with intent to outrage her modesty Cases ------------¶

    xs3 = crime_in_years_df.iloc[:, 0]
    ys3 = crime_in_years_df.iloc[:, 4]

    from sklearn.model_selection import train_test_split  
    X_train3, X_test3, y_train3, y_test3 = train_test_split(xs3, ys3, test_size=0.3, random_state=0) 
    print(str(X_train3)+"\n\n"+str(y_train3))

    plt.scatter(X_train3,y_train3)
    plt.ylabel("Assault on women with intent to outrage her modesty")
    plt.xlabel("Year")
    plt.show()

    print(slope_intercept(xs3,ys3))

    m3,b3=slope_intercept(xs3,ys3)
    reg_line=[(m3*x)+b3 for x in xs3]

    fig = plt.figure(figsize=(15,12))
    plt.scatter(xs3,ys3,color="green")
    plt.plot(xs3,reg_line)
    plt.ylabel("Assault on women with intent to outrage her modesty crime rate")
    plt.xlabel("Year")
    plt.title("before prediction a regression line for Assault her modesty Crime")
    plt.show()

    print("Regressor intercept: "+str(m3) )
    print("Regressor coefficient: "+str(b3) )

    print("Year")
    for x in np.nditer(xs3):
        print(x)
    print("\n\nAssault on women with intent to outrage her modesty Values")
    for x in np.nditer(ys3):
        print(x)

# Prediction of next years for "Assault on women with intent to outrage her modesty" column:¶

    years = np.array([2013, 2014, 2015, 2016, 2017, 2018])
    predict_crime_in_years3 = pd.DataFrame(years)
    print(predict_crime_in_years3)

    print("Predicrted values for Assault on women with intent to outrage her modesty")

    ypredd3 = (m3 * years + b3)
    for x in np.nditer(ypredd3):
        print(int(x))

  # merging the values and year with predicted values and year.

    merge_list3 = np.concatenate((xs3, years), axis=0)
    print("Years: " + str(merge_list3))

    reg_line = [(m3*x)+b3 for x in merge_list3]

    transform_list3 = np.concatenate((ys3, ypredd3), axis=0)
    print("Assault on women with intent to outrage her modesty "+str(transform_list3))

    fig = plt.figure(figsize=(15,12))
    plt.scatter(merge_list3, transform_list3, color = "Green")
    plt.plot(merge_list3, reg_line)
    plt.xlim((2001, 2018))
    plt.ylabel("Assault on women with intent to outrage her modesty Crime Rate")
    plt.xlabel("Years")
    plt.title("Prediucted Assault to outrage her modesty regression Line from 2001 to 2018")
    plt.show()  

 # ----------------- -5. Insult to modesty of Women Cases ----------------------------¶

    xs4 = crime_in_years_df.iloc[:, 0]
    ys4 = crime_in_years_df.iloc[:, 5]

    X_train4, X_test4, y_train4, y_test4 = train_test_split(xs4, ys4, test_size=0.3, random_state=0) 
    print(str(X_train4)+"\n\n"+str(y_train4))

    plt.scatter(X_train4,y_train4)
    plt.ylabel("Insult to modesty of Women ")
    plt.xlabel("Year")
    plt.show()   

    print(slope_intercept(xs4, ys4))

    m4,b4=slope_intercept(xs4,ys4)
    reg_line=[(m4*x)+b4 for x in xs4]

    fig = plt.figure(figsize=(15,12))
    plt.scatter(xs4,ys4,color="brown")
    plt.plot(xs4,reg_line)
    plt.ylabel("Insult to modesty of Women crime rate")
    plt.xlabel("Year")
    plt.title("Making a regression line")
    plt.show()

    print("Regressor intercept: "+str(m4) )
    print("Regressor coefficient: "+str(b4) )

    print("Year")
    for x in np.nditer(xs4):
        print(x)
    print("\n\nInsult to modesty of Women crime values")
    for x in np.nditer(ys4):
        print(x)

#### Prediction of next years for "Insult to modesty of Women" column:
        
    years = np.array([2013,2014,2015,2016,2017,2018])
    predict_crime_in_years4=pd.DataFrame(years)
    print(predict_crime_in_years4)

    print("Predicted values of Insult to modesty of women")
    ypredd4=(m4 * years + b4)
    for x in np.nditer(ypredd4):
        print(int(x)) 

# merging the values and year with predicted values and year.

    merge_list4 = np.concatenate((xs4, years), axis=0)
    print("Years: " + str(merge_list4))


    reg_line = [(m4 * x) + b3 for x in merge_list4]

    transform_list4 = np.concatenate((ys4, ypredd4), axis=0)
    print("Insult to modesty of women: "+str(transform_list4))

    fig = plt.figure(figsize=(15,12))
    plt.scatter(merge_list4, transform_list4, color = "brown")
    plt.plot(merge_list4, reg_line)
    plt.xlim((2001, 2018))
    plt.ylabel("Insult to modesty of women Crime Rate")
    plt.xlabel("Years")
    plt.title("Predicted Insult to modesty of women regression Line from 2001 to 2018")
    plt.show()   

#  ---- 6. Cruelty by Husband or his Relatives --------

    xs5 = crime_in_years_df.iloc[:,0]
    ys5 = crime_in_years_df.iloc[:,6]

    X_train5, X_test5, y_train5, y_test5 = train_test_split(xs5, ys5, test_size=0.3, random_state=0) 
    print(str(X_train5)+"\n\n"+str(y_train5))

    plt.scatter(xs5,ys5)
    plt.ylabel("Cruelty by Husband or his Relatives ")
    plt.xlabel("Year")
    plt.show()   

    print(slope_intercept(xs5,ys5))

    m5,b5=slope_intercept(xs5,ys5)
    reg_line=[(m5*x)+b5 for x in xs5]

    fig = plt.figure(figsize=(12,10))
    plt.scatter(xs5,ys5,color="green")
    plt.plot(xs5,reg_line)
    plt.ylabel("Cruelty by Husband or his Relatives crime rate")
    plt.xlabel("Year")
    plt.title("before prediction a regression line of cruelty by Husband/relatives") 
    plt.show()

    print("Regressor intercept: "+str(m5) )
    print("Regressor coefficient: "+str(b5) )

    print("Year")
    for x in np.nditer(xs5):
        print(x)
    print("\n\n Cruelty by Husband or his Relatives crime values")
    for x in np.nditer(ys5):
        print(x)   

 #### Prediction of next years for "Cruelty by Husband or his Relatives" column:

    years = np.array([2013,2014,2015,2016,2017,2018])
    predict_crime_in_years5 = pd.DataFrame(years)
    print(predict_crime_in_years5)
    ypredd5 = (m5 * years + b5)

    print("Predicted Values of Cruelty by Husband/relatives")
    for x in np.nditer(ypredd5):
        print(int(x)) 

    # merging the values and year with predicted values and year.

    merge_list5 = np.concatenate((xs5, years), axis=0)
    print("Years: " + str(merge_list5))


    reg_line = [(m5 * x) + b5 for x in merge_list5]

    transform_list5 = np.concatenate((ys5, ypredd5), axis=0)
    print("Cruelty by Husband/relatives: "+str(transform_list5))

    fig = plt.figure(figsize=(15,12))
    plt.scatter(merge_list5, transform_list5, color = "brown")
    plt.plot(merge_list5, reg_line)
    plt.xlim((2001, 2018))
    plt.ylabel("Cruelty by Husband/relaties Crime Rate")
    plt.xlabel("Years")
    plt.title("Predicted values of cruelty by husband/relatives regression Line from 2001 to 2018")
    plt.show()

## ---------------- 7. Importation of Girls------------

    xs6 = crime_in_years_df.iloc[:,0]
    ys6 = crime_in_years_df.iloc[:,7]

    X_train6, X_test6, y_train6, y_test6 = train_test_split(xs5, ys5, test_size=0.3, random_state=0) 
    print(str(X_train6)+"\n\n"+str(y_train6))

    plt.scatter(X_train6,y_train6)
    plt.ylabel("Importation of Girls ")
    plt.xlabel("Year")
    plt.show()

    slope_intercept(xs6,ys6)

    m6,b6 = slope_intercept(xs6,ys6)
    reg_line=[(m6 * x) + b6 for x in xs6]
    plt.scatter(xs6, ys6, color="green")
    plt.plot(xs6 ,reg_line)
    plt.ylabel("Importation of Girls crime rate")
    plt.xlabel("Year")
    plt.title("before prediction regression line for Imporetation of Girls")

    print("Regressor intercept: "+str(m6) )
    print("Regressor coefficient: "+str(b6) )

    print("Year")
    for x in np.nditer(xs6):
        print(x)
    print("\n\nValues of Importation of Girls")
    for x in np.nditer(ys6):
        print(x) 

#### Prediction of next years for "Importation of Girls" column:
     
    years = np.array([2013,2014,2015,2016,2017,2018])
    predict_crime_in_years6 = pd.DataFrame(years)
    print(predict_crime_in_years6)

    print("Predicted vlaues of Imporatation of Girls:")

    ypredd6 = (m6 * years + b6) * -1
    for x in np.nditer(ypredd6):
        print(int(x))

# merging the values and year with predicted values and year.

    merge_list6 = np.concatenate((xs6, years), axis=0)
    print("Years: " + str(merge_list6))


    reg_line = [(m6 * x) + b6 for x in merge_list6]

    transform_list6 = np.concatenate((ys5, ypredd6), axis=0)
    print("Importation of Girls: "+str(transform_list6))

    fig = plt.figure(figsize=(15,12))
    plt.scatter(merge_list6, transform_list6, color = "brown")
    plt.plot(merge_list6, reg_line)
    plt.xlim((2001, 2018))
    plt.ylabel("Importation of girls Crime Rate")
    plt.xlabel("Years")
    plt.title("Predicted values of Imporatation of Girls regression Line from 2001 to 2018")
    plt.show()

    ### Accuracy:

    accuracy = regressor.score(X1_test, y1_test)
    accuracy = round(accuracy, 2)
    print("Accuaracy: " + str(accuracy * 100) + '%')







