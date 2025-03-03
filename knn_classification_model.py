def prediction(Class_Name, urban_test, new_instance): 
    #Import the different packages
    from sklearn.model_selection import train_test_split
    from sklearn import preprocessing
    from sklearn.neighbors import KNeighborsClassifier

    dfX = urban_test.drop (columns = [Class_Name])
    sy = urban_test[Class_Name]

    #Part a: Dividing the data into the training set and test set
    dfX_train, dfX_test, sy_train, sy_test = train_test_split(dfX,sy)

    dfX_train, dfX_test, sy_train, sy_test;

    #Preprocessing encoder
    le = preprocessing.LabelEncoder()
  

    le = le.fit(sy_train)

    #Transformed training sets
    y_train = le.transform(sy_train)
    sy_train, y_train

    dfX_train.to_numpy()
    #using the MinMaxScaler from the classifiers
    nl = preprocessing.MinMaxScaler()
    nl = nl.fit(dfX_train.to_numpy())

    X_train = nl.transform(dfX_train.to_numpy())

    #Part c: Build the machine learning model using knn-classifier, we used a k-value of 15

    knn = KNeighborsClassifier(n_neighbors=15)
    knn.fit(X_train, y_train)
    y_test = le.transform(sy_test.to_numpy())
    X_test = nl.transform(dfX_test.to_numpy())
    knn.predict(X_test)
    knn.score(X_test, y_test)

    #Will then predict the new_instance class using the knn transformation
    if new_instance is not None:
        # Ensure the new instance has the same structure as the training data
        new_instance_scaled = nl.transform(new_instance)  # Scale the new instance
        new_instance_class = knn.predict(new_instance_scaled)  # Predict the class of the new instance
        new_instance_class = le.inverse_transform(new_instance_class)  # Convert numeric label back to original class
        return new_instance_class # Return the predicted class of the new instance
def train_classifier(csv_name):
    import pandas as pd

    # Read in the data from the csv into a DataFrame object
    urban_test = pd.read_csv(csv_name)
    #Map Issue Name to Numerical Equivalent that can be used later on for the classification system
    dict_ammenties = {"Amenities Nearby (eg.parks)": 0, "Pot holes": 1, "Icy Roads": 2, "Recreational Amenities": 3, "Drainage": 4, "Theft": 5, 
                    "Animal Control (eg. dogs peeing on lawns) ": 6, "Garbage Disposal": 7, "Homeless Nearby": 8, "Hailstorms": 9, "Bus stops / Transportation": 10,
                    "Construction": 11,
                    }
    #update the rows in the dictionary
    urban_test.replace(dict_ammenties, inplace=True)

    #We want data sets for the Following: Issue, Household, Minors, Age, Durations
    issue = pd.DataFrame()
    household = pd.DataFrame()
    minors = pd.DataFrame()
    age = pd.DataFrame()
    durations = pd.DataFrame()

    #stores the different data frame objects that we want to classify
    dataframe_list = [] 

    #Training the knn classifer with randomized data that has an attached class so that it can better predict
    issue['Issue'] = urban_test.iloc[:, 0]
    issue['Priority'] = urban_test.iloc[:, 1]
    #Add to the dataframe list
    dataframe_list.append(issue)
    household['Members'] = urban_test.iloc[:, 2]
    household['family_size'] = urban_test.iloc[:, 3]
    #Add to the dataFrame list
    dataframe_list.append(household)
    minors['minor_number'] = urban_test.iloc[:, 4]
    minors['Dependents'] = urban_test.iloc[:, 5]
    #Add dataFrame list
    dataframe_list.append(minors)
    age['Age'] = urban_test.iloc[:, 6]
    age['age_range'] = urban_test.iloc[:, 7]
    #Add to dataFrame list 
    dataframe_list.append(age)
    durations['length'] = urban_test.iloc[:, 8]
    durations['issue_duration'] = urban_test.iloc[:, 9]
    dataframe_list.append(durations)

    #Classification Names that will be used are stored in a list and will index correspond to the columns of the dataframe objects
    class_names = ['Priority', 'family_size', 'Dependents', 'age_range', 'issue_duration']
    return class_names, dataframe_list

if __name__ == "__main__":
    class_names, dataframe_list = train_classifier("raw_data.csv")
    #Using the final model to predict the class of a new instance

    #Training knn for different predictors and then passing in new examples
    import numpy as np
    import pandas as pd

    #read sample user input
    sample_df = pd.read_csv("sample_data.csv")
    dict_ammenties = {"Amenities Nearby (eg.parks)": 0, "Pot holes": 1, "Icy Roads": 2, "Recreational Amenities": 3, "Drainage": 4, "Theft": 5, 
                    "Animal Control (eg. dogs peeing on lawns) ": 6, "Garbage Disposal": 7, "Homeless Nearby": 8, "Hailstorms": 9, "Bus stops / Transportation": 10,
                    "Construction": 11,
                    }
    sample_df.replace(dict_ammenties, inplace=True)
    #get a list of column names from the data
    col_name = sample_df.columns.to_list()
    #remove timestamp from consideration
    col_name = col_name[1:]

    #Create a list that will hold dataFrames for the fields that require classification
    new_tmp_df_list_predicted_classes = []
    #Go through class names list since that is the number of fields that we require calssifying
    for i in range(len(class_names)):
        data_list = sample_df[col_name[i]]  #the values extracted for the column name corresponding to the data we want to classify
        new_data_val = pd.DataFrame() #New DataFrame Object
        new_data_val[col_name[i]] = data_list  #Creating the first column for the dataframe
        new_data_val[class_names[i]] = None #Initially setting classification fields to None
        for j in range(len(data_list)): #iterate through all sample user input values
            new_example_rawdata = np.array([ [data_list[j]] ]) #convert to an array so it can be passed in as an instance for classification
            val = prediction(class_names[i], dataframe_list[i], new_example_rawdata) #call prediction function

            new_data_val.loc[j] = [data_list[j], str(val[0])] #Add predicited class returned from classifier to the DataFrame
        
        new_tmp_df_list_predicted_classes.append(new_data_val) #Add these dataFrames to the list created

    #Create final categorized DataFrame
    categorized = pd.DataFrame()
    #Iterate through range of the prediction classified issues and add them to the cateogroized dataFrame
    for i in range(len(new_tmp_df_list_predicted_classes)):
        #Grab column name list for the DataFrame of interest
        col_list = new_tmp_df_list_predicted_classes[i].columns.to_list()
        #Set the corresponding column name in the categorized DataFrame to be the values of the column in the DataFrame
        categorized[col_list[0]] = new_tmp_df_list_predicted_classes[i][col_list[0]]
        categorized[col_list[1]] = new_tmp_df_list_predicted_classes[i][col_list[1]]

    #Append remaining coloumns that did not need to be classified
    for i in range(5, len(col_name)):
        categorized[col_name[i]] = sample_df[col_name[i]]

    #Revert the issues back to their names not numbers
    dict_ammenties_reverse = {0: "Amenities Nearby (eg.parks)", 1: "Pot holes", 2: "Icy Roads", 3: "Recreational Amenities", 4: "Drainage", 5: "Theft", 
                    6: "Animal Control (eg. dogs peeing on lawns) ", 7: "Garbage Disposal", 8: "Homeless Nearby", 9: "Hailstorms", 10: "Bus stops / Transportation",
                    11: "Construction",
                    }
    #update the rows in the dictionary
    categorized["Issue"] = categorized["Issue"].replace(dict_ammenties_reverse)
    #Write into CSV File
    categorized.to_csv("categorized_data.csv")


