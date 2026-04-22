import pandas as pd
from sklearn.preprocessing import OneHotEncoder
def preprocessing(df : pd.DataFrame) -> pd.DataFrame :
    df.drop_duplicates(inplace=True)
    for col in ['Years at Company', 'Monthly Income']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[~((df[col] < lower_bound) | (df[col] > upper_bound))]
    
    #Binary encoding
    yesNo = {'Yes': 1, 'No': 0}
    for col in ['Overtime', 'Remote Work', 'Leadership Opportunities', 'Innovation Opportunities']:
        df[col] = df[col].map(yesNo)

    gender = {'Male' : 1, 'Female' : 0}
    df['Gender'] = df['Gender'].map(gender)

    attrition = {'Stayed' : 0, 'Left' : 1}
    df['Attrition'] = df['Attrition'].map(attrition)

    #Ordinal encoding
    #`Work-Life Balance`, `Job Satisfaction`, `Performance Rating`, `Education Level`, `Job Level`, `Company Size`, `Company Reputation`, `Employee Recognition`
    workLife = {'Poor': 1, 'Fair': 2, 'Good': 3, 'Excellent': 4}
    df['Work-Life Balance'] = df['Work-Life Balance'].map(workLife)

    jobSatisfaction = {'High': 3,'Very High': 4,'Medium': 2,'Low':1}
    df['Job Satisfaction'] = df['Job Satisfaction'].map(jobSatisfaction)

    perf_rating = {'Average': 3, 'High': 4, 'Below Average': 2, 'Low': 1}
    df['Performance Rating'] = df['Performance Rating'].map(perf_rating)

    education_order = {'High School': 1, 'Associate Degree': 2, 'Bachelor’s Degree': 3, 'Master’s Degree': 4, 'PhD': 5}
    df['Education Level'] = df['Education Level'].map(education_order)

    job_level_order = {'Entry': 1, 'Mid': 2, 'Senior': 3}
    df['Job Level'] = df['Job Level'].map(job_level_order)

    company_size_order = {'Small': 1, 'Medium': 2, 'Large': 3}
    df['Company Size'] = df['Company Size'].map(company_size_order)

    reputation_order = {'Poor': 1, 'Fair': 2, 'Good': 3, 'Excellent': 4}
    df['Company Reputation'] = df['Company Reputation'].map(reputation_order)

    recognition_order = {'Low': 1, 'Medium': 2, 'High': 3, 'Very High': 4}
    df['Employee Recognition'] = df['Employee Recognition'].map(recognition_order)

    #Nominal encoding
    df = df.drop_duplicates().reset_index(drop=True)
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_data = encoder.fit_transform(df[['Job Role', 'Marital Status']])
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(['Job Role', 'Marital Status']))
    df = pd.concat([df.drop(['Job Role', 'Marital Status'], axis=1), encoded_df], axis=1)
    
    
    return df
