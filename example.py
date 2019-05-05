import pandas as pd
import os
import core as pcm


root = "/home/ydon868/Desktop/PCA4MIX"#root path for input csv file
filename = "Iris.csv"
#Load csv file into Pandas Dataframe
input_data = pd.read_csv(os.path.join(root, filename))

#Select columns to be processed
mydata = input_data.iloc[:,0:5]

#Create PCA4MIX object
mypcm = pcm.PCA4MIX()

#Split dataframe to quantitative and qualitative dataframes
data_numeric, data_category = mypcm.splitdata(mydata)

#Call pcamix function
eigDF, U, F, Sqloadings, ReContriPct, CoefMatrix = mypcm.pcamix(data_numeric, data_category)

#Save to csv
ReContriPct.to_csv("ReContriPct.csv", sep="\t",encoding="utf-8")



