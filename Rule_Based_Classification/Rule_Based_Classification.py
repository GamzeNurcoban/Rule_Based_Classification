


################################### RULE BASED CLASSIFICATION  ########################################
# A game company wants to create level-based new customer definitions (personas) by using some
# features ( Country, Source, Age, Sex) of its customers, and to create segments according to these new customer
# definitions and to estimate how much profit can be generated from  the new customers according to these segments.

# In this study, how to do rule-based classification and customer-based revenue calculation
# have been discussed step by step.

########################## Importing Libraries ##########################


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 20)

############
import pandas as pd
df=pd.read_csv("WEEK_2/Kural_Tabanli_Siniflandirma_/Kural_Tabanli_Siniflandirma/Kural_Tabanli_Siniflandirma/persona.csv")
df.head()
df.tail()
df.shape
df.info()
df.columns
df.index
df.describe().T
df.isnull().values.any()
df.isnull().sum()


#############  Görev 1 ##################

# Soru 1: persona.csv dosyasını okutunuz ve veri seti ile ilgili genel bilgileri gösteriniz.

def load_dataset(dataframe):
    return pd.read_csv("Datasets/" + dataframe + ".csv")


df = load_dataset("persona")
df.head()


######## Describing The Data ########

def check_df(dataframe, head=5):
    """
    This Function returns:
        - shape : The dimension of dataframe.
        - size : Number of elements in the dataframe.
        - type : The data type of each variable.
        - Column Names : The column labels of the DataFrame.
        - Head : The first "n" rows of the DataFrame.
        - Tail : The last "n" rows of the DataFrame.
        - Null Values : Checking if any "NA" Value is into DataFrame
        - quantile : The Basics Statistics
    Parameters
    ----------
    dataframe : dataframe
        Dataframe where the dataset is kept.
    head : int, optional
        The function which is used to get the first "n" rows.
    Returns
    -------
    Examples
    ------
        import pandas as pd
        df = pd.read_csv("titanic.csv")
        print(check_df(df,10))
    """
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Size #####################")
    print(dataframe.size)
    print("##################### Type #####################")
    print(dataframe.dtypes)
    print("############### Column Names ####################")
    print(dataframe.columns)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("################## Null Values ##################")
    print(dataframe.isnull().values.any())
    print("################## Quantiles ####################")
    print(dataframe.quantile(q=[0, 0.25, 0.50, 0.75, 1]))

check_df(df)

###### Selection of Categorical and Numerical Variables ########

def grab_col_names(dataframe, cat_th=5, car_th=20):
    """
    This function to perform the selection of numeric and categorical variables in the data set in a parametric way.
    Note: Variables with numeric data type but with categorical properties are included in categorical variables.
    Parameters
    ----------
    dataframe: dataframe
        The data set in which Variable types need to be parsed
    cat_th: int, optional
        The threshold value for number of distinct observations in numerical variables with categorical properties.
        cat_th is used to specify that if number of distinct observations in numerical variable is less than
        cat_th, this variables can be categorized as a categorical variable.
    car_th: int, optional
        The threshold value for categorical variables with  a wide range of cardinality.
        If the number of distinct observations in a categorical variables is greater than car_th, this
        variable can be categorized as a categorical variable.
    Returns
    -------
        cat_cols: list
            List of categorical variables.
        num_cols: list
            List of numerical variables.
        cat_but_car: list
            List of categorical variables with  a wide range of cardinality.
    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))
    Notes
    ------
        Sum of elements in lists the cat_cols,num_cols  and  cat_but_car give the total number of variables in dataframe.
    """

    # cat cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].dtypes != "O" and
                   dataframe[col].nunique() < cat_th]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].dtypes == "O" and
                   dataframe[col].nunique() > car_th]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    return {"Categorical_Data": cat_cols,
            "Numerical_Data": num_cols,
            "Categorical_But_Cardinal_Data": cat_but_car}


grab_col_names(df)

####### General Exploration for Categorical Data ########

def cat_summary(dataframe, plot=False):
    cat_cols = grab_col_names(dataframe)["Categorical_Data"]
    for col_name in cat_cols:
        print("############## Unique Observations of Categorical Data ###############")
        print("The unique number of " + col_name + ": " + str(dataframe[col_name].nunique()))

        print("############## Frequency of Categorical Data ########################")
        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio": dataframe[col_name].value_counts() / len(dataframe)}))
        if plot == True:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show()


cat_summary(df, plot=True)


######### General Exploration for Numerical Data ###########

def num_summary(dataframe, plot=False):
    numerical_col = ['PRICE', 'AGE']  ##grab_col_names(dataframe)["Numerical_Data"]
    quantiles = [0.25, 0.50, 0.75, 1]
    for col_name in numerical_col:
        print("########## Summary Statistics of " + col_name + " ############")
        print(dataframe[col_name].describe(quantiles).T)

        if plot:
            sns.histplot(data=dataframe, x=col_name)
            plt.xlabel(col_name)
            plt.title("The distribution of " + col_name)
            plt.grid(True)
            plt.show(block=True)


num_summary(df, plot=True)

######################################################
# Soru 2: Kaç unique SOURCE vardır? Frekansları nedir?

df["SOURCE"].nunique()
len(df["SOURCE"].unique())

######################################################
# Soru 3: Kaç unique PRICE vardır?

df[["PRICE"]].nunique()

######################################################
# Soru 4: Hangi PRICE'dan kaçar tane satış gerçekleşmiş?

df[["PRICE"]].value_counts()

######################################################
# Soru 5: Hangi ülkeden kaçar tane satış olmuş?

df["COUNTRY"].value_counts(ascending=False, normalize=True)

# Oransal olarak seçim yapmak istersek;
# df['COUNTRY'].value_counts(ascending=False, normalize=True).loc[lambda x : x > 0.20]
# df[['COUNTRY']].value_counts(ascending=False, normalize=True).where(lambda x: x > 0.20 ).dropna()

######################################################
#  Soru 6: Ülkelere göre satışlardan toplam ne kadar kazanılmış?

df.groupby("COUNTRY").agg({"PRICE": "sum"})

######################################################
# Soru 7: SOURCE türlerine göre satış sayıları nedir?

df.groupby("SOURCE").agg({"PRICE": "count"}) # farklı bir değişkene göre de count alınabilir

######################################################
# Soru 8: Ülkelere göre PRICE ortalamaları nedir?

df.groupby("COUNTRY").agg({"PRICE": "mean"})

######################################################
# Soru 9: SOURCE'lara göre PRICE ortalamaları nedir?

df.groupby("SOURCE").agg({"PRICE": "mean"})

######################################################
# Soru 10: COUNTRY-SOURCE kırılımında PRICE ortalamaları nedir?

df.groupby(["COUNTRY", "SOURCE"]).agg({"PRICE": "mean"})

df.pivot_table(index="COUNTRY", columns="SOURCE", values="PRICE", aggfunc="mean")

#####################################################################################
# Görev 2: COUNTRY, SOURCE, SEX, AGE kırılımında ortalama kazançlar nedir?

# df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE": "mean"})
df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE": "mean"})

#####################################################################################
# Görev 2: Çıktıyı PRICE’a göre sıralayınız.
df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE": "mean"}).sort_values("PRICE", ascending=False)

# Önceki sorudaki çıktıyı daha iyi görebilmek için sort_values metodunu azalan olacak şekilde PRICE’a göre uygulayınız.
# Çıktıyı agg_df olarak kaydediniz.

df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE": "mean"}).sort_values("PRICE", ascending=False)

agg_df = df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE": "mean"}).sort_values("PRICE", ascending=False)
agg_df.head()

####################################################################################
# Görev 4: Indekste yer alan isimleri değişken ismine çeviriniz.

agg_df.reset_index()
agg_df.reset_index(inplace=True)
agg_df.head()

####################################################################################
#  Görev 5: Age değişkenini kategorik değişkene çeviriniz ve agg_df’e ekleyiniz.

 #AGE değişkeninin nerelerden bölüneceğini belirtelim:
bins = [0, 18, 23, 30, 40, agg_df["AGE"].max()]

# Bölünen noktalara karşılık isimlendirmelerin ne olacağını ifade edelim:
mylabels = ['0_18', '19_23', '24_30', '31_40', '41_' + str(agg_df["AGE"].max())]

# age'i bölelim:
agg_df["age_cat"] = pd.cut(agg_df["AGE"], bins, labels=mylabels)
agg_df.head()
###################################################################################
my_bins = [df["AGE"].min() - 1, 18, 23, 35, 45, df["AGE"].max()]
my_labels = [str(df["AGE"].min()) + '_18', '19_23', '24_35', '36_45', '46_' + str(df["AGE"].max())]

agg_df["AGE_CAT"] = pd.cut(agg_df["AGE"], bins=my_bins, labels=my_labels)

agg_df.groupby("AGE_CAT").agg({"AGE": ["min", "max", "count"]})
agg_df.head()

# CONCAT işleminde AGE olmayacağı için dataframe'den silelim ve sıralamayı uygun hale getirelim:
agg_df = agg_df[["COUNTRY", "SOURCE", "SEX", "AGE_CAT", "PRICE"]]
agg_df.head()

# ["_".join(row) for row in  agg_df.values[:, :4]]

agg_df["CUSTOMERS_LEVEL_BASED"] = pd.DataFrame(["_".join(row).upper() for row in agg_df.values[:, 0:4]])
agg_df.head()

# 2.yöntem -> agg_df[['COUNTRY', 'SOURCE', 'SEX', 'AGE_CAT']].agg(lambda x: '_'.join(x.values), axis=1).apply(lambda x: x.upper())

####################################################################################
# Görev 7: Yeni müşterileri (personaları) segmentlere ayırınız.

df_persona = agg_df.groupby('CUSTOMERS_LEVEL_BASED').agg({'PRICE': "mean"})
df_persona = df_persona.sort_values("PRICE", ascending=False)
df_persona.head()

# Segments
segment_labels = ["D", "C", "B", "A"]

df_persona["SEGMENT"] = pd.qcut(df_persona["PRICE"], 4, labels=segment_labels)

df_persona.head()
df_persona.reset_index(inplace=True)
df_segment = df_persona.groupby('SEGMENT').mean("PRICE").reset_index().sort_values("SEGMENT", ascending=False)

#
# # Demonstrating segments as bars on a chart, where the length of each bar varies based on the value of the customer profile
# # Set the width and height of the figure
# plt.figure(figsize=(10,6))
# # Add title
# plt.title("Segments Distibutions of Users")
# sns.barplot(x=df_segment["SEGMENT"].unique(), y=df_segment['PRICE'])
# # Add label for vertical axis
# plt.ylabel("Average Price of Segments")
# plt.show()


######################## Prediction ########################

def AGE_CAT(age):
    if age <= 18:
        AGE_CAT = "15_18"
        return AGE_CAT
    elif (age > 18 and age <= 23):
        AGE_CAT = "19_23"
        return AGE_CAT
    elif (age > 23 and age <= 35):
        AGE_CAT = "24_35"
        return AGE_CAT
    elif (age > 35 and age <= 45):
        AGE_CAT = "36_45"
        return AGE_CAT
    elif (age > 45 and age <= 66):
        AGE_CAT = "46_66"
        return AGE_CAT

df_persona.head()

def ruled_based_classification():
    COUNTRY = input("Enter a country name (USA/EUR/BRA/DEU/TUR/FRA):")
    SOURCE = input("Enter the operating system of phone (IOS/ANROID):")
    SEX = input("Enter the gender (FEMALE/MALE):")
    AGE = int(input("Enter the age:"))
    AGE_SEG = AGE_CAT(AGE)
    new_user = COUNTRY.upper() + '_' + SOURCE.upper() + '_' + SEX.upper() + '_' + AGE_SEG
    print(new_user)
    if df_persona[df_persona["CUSTOMERS_LEVEL_BASED"] ==new_user].shape[0]>0:
        print("Segment:" + df_persona[df_persona["CUSTOMERS_LEVEL_BASED"] == new_user].loc[:, "SEGMENT"].values[0])
        print("Price:" + str(df_persona[df_persona["CUSTOMERS_LEVEL_BASED"] == new_user].loc[:, "PRICE"].values[0]))
    else:
        print("Unknown User! Try Again!")


ruled_based_classification()

new_user = "EUR_IOS_MALE_36_45"
agg_df[agg_df["CUSTOMERS_LEVEL_BASED"] == new_user]
new_user = "FRA_ANDROID_MALE_15_18"



# Kullanıcıyı dataframe'den seçmek için:

# df_persona[df_persona["CUSTOMERS_LEVEL_BASED"] == "FRA_ANDROID_MALE_15_18"]["SEGMENT"]
# type(df_persona[df_persona["CUSTOMERS_LEVEL_BASED"] == "FRA_ANDROID_MALE_15_18"].loc[:, "SEGMENT"].values)
# df_persona[df_persona["CUSTOMERS_LEVEL_BASED"] == "FRA_ANDROID_MALE_15_18"].loc[:, "SEGMENT"].values[0]


# 33 yaşında ANDROID kullanan bir Türk kadını hangi segmente aittir ve ortalama ne kadar gelir kazandırması beklenir?
new_user = "TUR_ANDROID_FEMALE_24_35"
df_persona[df_persona["CUSTOMERS_LEVEL_BASED"] == new_user]["SEGMENT"].values[0]
df_persona[df_persona["CUSTOMERS_LEVEL_BASED"] == new_user]["PRICE"].values[0]

# 35 yaşında IOS kullanan bir Fransız kadını hangi segmente aittir ve ortalama ne kadar gelir kazandırması beklenir
new_user = "FRA_ANDROID_FEMALE_24_35"
df_persona[df_persona["CUSTOMERS_LEVEL_BASED"] == new_user]["SEGMENT"].values[0]
df_persona[df_persona["CUSTOMERS_LEVEL_BASED"] == new_user]["PRICE"].values[0]
