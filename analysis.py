import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import sys
import glob
import seaborn as sns; sns.set()
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split

import scipy as sp
from sklearn.preprocessing import LabelEncoder

import xgboost as xgb

pd.set_option('display.max_rows', None)

def parse_args() -> tuple:
    parser = argparse.ArgumentParser()
    parser.add_argument("IN_DIR", help="Input dir", type=str)

    args = parser.parse_args()
    return (args.IN_DIR)

def down_sample(df, rand) -> pd.DataFrame:
    dual_len = len(df[df["FAC_PENALTY"] == 1])
    df_dropsample = df[df["FAC_PENALTY"] == 0].sample(n=(len(df) - (dual_len * 2)), random_state=rand)
    df = df.drop(index=df_dropsample.index)

    return df

def chi_squared(df) -> list:
    chi_list = []
    for c in df.columns.values:
        crossed_df = pd.crosstab(df[c], df["FAC_PENALTY"])
        x2, p, dof, expected = sp.stats.chi2_contingency(crossed_df)
        min_d = min(crossed_df.shape) - 1
        n = len(df[c])
        if min_d == 0 or n == 0:
            v = 0
        else:
            v = np.sqrt(x2/(min_d*n))
        chi_list.append([c, x2, p, dof, expected, v])

    return chi_list

def spearmanr(df) -> list:
    spear_list = []
    for c in df.columns.values:
        res = sp.stats.spearmanr(df[c], df["FAC_PENALTY"])
        stat = res.statistic
        p = res.pvalue
        spear_list.append([c, stat, p])

    return spear_list

def kendall(df) -> list:
    kendall_list = []
    for c in df.columns.values:
        corr, p = sp.stats.kendalltau(df[c], df["FAC_PENALTY"])
        kendall_list.append([c, corr, p])

    return kendall_list


def xgboost(df) -> list:
    # explanatory variable
    df_ex = df.drop(["FAC_PENALTY"], axis=1)

    # response variable
    df_res = df.filter(items=["FAC_PENALTY"], axis=1)

    # delete NaN columns
    df_ex = df_ex.dropna(how="any", axis=1)

    print("df_ex length: ", len(df_ex))
    print("df_res length: ", len(df_res))
    print("FAC_PENALTY=1 length: ", len(df[df["FAC_PENALTY"] == 1]))
    
    # split data
    X_train, X_test, y_train, y_test = train_test_split(df_ex, df_res, test_size=0.3, random_state=1)

    # xgboost
    model = xgb.XGBClassifier()
    model.fit(X_train, y_train.values.ravel())

    # predict
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # result
    labels = X_train.columns
    importances = model.feature_importances_

    label_list = []
    importance_list = []
    for i in labels:
        label_list.append(i)
    for i in importances:
        importance_list.append(float(i))

    with open("feature_importances.txt", "w") as f:
        for li in zip(label_list, importance_list):
            f.write(str(li[0]))
            f.write("\t")
            f.write(str(li[1]))
            f.write("\n")

    print(classification_report(y_test, y_test_pred))

    return [y_train, y_train_pred, y_test, y_test_pred]


def label_encoding(df, label:str):
    le = LabelEncoder()
    le = le.fit(df[label])
    df[label] = le.transform(df[label])
    
    return df

def main() -> None:
    (in_dir) = parse_args()

    csv_list = glob.glob(in_dir + "/*.csv")

    df_concat = pd.DataFrame()
    for l in csv_list:
        d = pd.read_csv(l)
        df_concat = pd.concat([df_concat, d])
    df = df_concat

    print("data row: " + str(df.shape[0]))
    print("data col: " + str(df.shape[1]))
    
    df["FAC_PENALTY_COUNT"] = df["FAC_PENALTY_COUNT"].mask((df["FAC_PENALTY_COUNT"] == "-"), 0)
    df = df.astype({"FAC_PENALTY_COUNT": "int64"})

    df = df.replace('-', np.nan)

    df = df.replace('Y', 1)
    df = df.replace('N', 0)
    df = df.replace('Yes', 1)
    df = df.replace('No', 0)


    df["FAC_PENALTY"] = 0
    df["FAC_PENALTY"] = df["FAC_PENALTY"].mask((df["FAC_PENALTY_COUNT"] > 0), 1)

    df["CAA"] = df["EPA Programs"].str.contains("CAA")
    df["CWA"] = df["EPA Programs"].str.contains("CWA")
    df["RCRA"] = df["EPA Programs"].str.contains("RCRA")
    df = df * 1

    df["CAA"] = df["CAA"].replace(np.nan, 0)
    df["CWA"] = df["CWA"].replace(np.nan, 0)
    df["RCRA"] = df["RCRA"].replace(np.nan, 0)

    related_var_list = [
        "Facility",
        "ECHO Facility Report",
        "FAC_DERIVED_TRIBES",
        "EJSCREEN Report",
        "Federal Agency",
        "Count",
        "FAC_COUNTY",
        "FAC_FIPS_CODE",
        "FAC_INDIAN_CNTRY_FLG",
        "FAC_COLLECTION_METHOD",
        "FAC_DERIVED_HUC",
        "FAC_DERIVED_WBD",
        "FAC_DATE_LAST_INSPECTION",
        "FAC_DATE_LAST_INFORMAL_ACTION",
        "FAC_DATE_LAST_FORMAL_ACTION",
        "FAC_DATE_LAST_PENALTY",
        "AIR_IDS",
        "CAA_NAICS",
        "TRI_REPORTER",
        "FAC_ACTIVE_FLAG",
        "SDWA_SNC_FLAG",
        "Latitude",
        "Longitude",
        "EPA Programs",
        "FAC_TOTAL_PENALTIES",
        "FAC_PENALTY_COUNT",
        "FAC_LAST_PENALTY_AMT",
        "FAC_FORMAL_ACTION_COUNT",
        "FAC_INSPECTION_COUNT",
        "FAC_INFORMAL_COUNT",
        "FAC_DERIVED_CB2010",
        "FAC_DERIVED_CD113",
        "City",
    ]

    df = df.drop(related_var_list, axis=1)
    df = df.reset_index(drop=True)
    
    del_thresh = int(len(df) * 0.5)
    df = df.dropna(thresh=del_thresh, axis=1)
    df = df.reset_index(drop=True)

    df = df.dropna(how='any')

    print(df.isnull().sum())
    print("data row: " + str(df.shape[0]))
    print("data col: " + str(df.shape[1]))

    df = df.astype({"Region": "int64",
                    "FAC_PERCENT_MINORITY": "float64",
                    "FAC_POP_DEN": "float64",
                    "FAC_QTRS_WITH_NC": "int64",
                    "FAC_PROGRAMS_WITH_SNC": "int64",
                    "FAC_COMPLIANCE_STATUS": "str",
                    "CAA": "int64",
                    "CWA": "int64",
                    "RCRA": "int64",
                    })
    print(df.dtypes)

    df["FAC_POP_DEN"] = pd.cut(df["FAC_POP_DEN"], [0, 100, 500, 1000, 3000, 5000, 7000, 9000, 11000, 20000, 30000, 40000, 50000, 60000], labels=[1,2,3,4,5,6,7,8,9,10,11,12,13], ordered=False).cat.codes
    df["FAC_PERCENT_MINORITY"] = pd.cut(df["FAC_PERCENT_MINORITY"], [0,10,20,30,40,50,60,70,80,90,100], labels=[1,2,3,4,5,6,7,8,9,10], ordered=False).cat.codes
    df = df.reset_index(drop=True)


    var_list = [
        "State",
        "Status",
        "Industry",
        "FAC_SNC_FLG",
        "AIR_FLAG",
        "NPDES_FLAG",
        "SDWIS_FLAG",
        "RCRA_FLAG",
        "TRI_FLAG",
        "GHG_FLAG",
        "FAC_COMPLIANCE_STATUS",
    ]

    # label encoding
    for v in var_list:
        print(v)
        df = label_encoding(df, v)
    

    fi = [
        "FAC_PENALTY",
        "FAC_QTRS_WITH_NC",
        "TRI_FLAG",
        "AIR_FLAG",
        "GHG_FLAG",
        "Status",
        "NPDES_FLAG",
        "RCRA_FLAG",
        "State",
        "Federal Facility",
        "Region",
    ]
    chi = [
        "FAC_PENALTY",
        "FAC_QTRS_WITH_NC",
        "FAC_COMPLIANCE_STATUS",
        "TRI_FLAG",
        "GHG_FLAG",
        "FAC_PROGRAMS_WITH_SNC",
        "FAC_SNC_FLG",
        "State",
        "CWA",
        "NPDES_FLAG",
        "Industry",
    ]
    spearkendall = [
        "FAC_PENALTY",
        "FAC_QTRS_WITH_NC",
        "TRI_FLAG",
        "GHG_FLAG",
        "FAC_COMPLIANCE_STATUS",
        "FAC_PROGRAMS_WITH_SNC",
        "FAC_SNC_FLG",
        "CWA",
        "NPDES_FLAG",
        "SDWIS_FLAG",
        "RCRA",
    ]
    # df = df.filter(items=spearkendall)


    # chi_list = chi_squared(df)
    # spear_list = spearmanr(df)
    # kendall_list = kendall(df)

    # with open("chi_list.txt", "w") as f:
    #     for chi in chi_list:
    #         f.write('\t'.join([str(i) for i in chi[:4]]))
    #         f.write("\n")

    # with open("spear_list.txt", "w") as f:
    #     for s in spear_list:
    #         f.write('\t'.join([str(i) for i in s[:3]]))
    #         f.write("\n")

    # with open("kendall_list.txt", "w") as f:
    #     for s in kendall_list:
    #         f.write('\t'.join([str(i) for i in s[:3]]))
    #         f.write("\n")

    # sys.exit()


    rand_list = [
        213, 455, 510, 603, 700, 703, 710, 782, 899, 1387, 1705, 1845, 1880, 1973, 2043, 2061, 2286, 2333, 2369, 2498,
        # 2537, 2558, 2572, 2601, 2622, 2642, 2766, 2852, 2890, 2980, 3069, 3080, 3112, 3127, 3197, 3274, 3325, 3397, 3431, 3471,
        # 3490, 3512, 3580, 4160, 4211, 4366, 4491, 4562, 4850, 4866, 5071, 5203, 5372, 5573, 5613, 5744, 5803, 5867, 5917, 6149,
        # 6371, 6455, 6465, 6515, 6659, 6682, 7066, 7278, 7324, 7354, 7523, 7550, 7642, 7746, 7851, 7953, 7976, 8065, 8123, 8159,
        # 8174, 8207, 8229, 8263, 8331, 8349, 8375, 8408, 8427, 8451, 8648, 8952, 8988, 9119, 9166, 9245, 9318, 9459, 9697, 9833,
    ]
    
    with open("xgb_output.txt", "w") as f:

        for i in rand_list:

            df_downsample = down_sample(df, i)
            
            print("down_sample")
            print(df_downsample.shape[0])
            print(df_downsample.shape[1])

            [y_train, y_train_pred, y_test, y_test_pred] = xgboost(df_downsample)

            # result
            recall_train = recall_score(y_train, y_train_pred)
            recall_test = recall_score(y_test, y_test_pred)
            specificity_train = specificity_score(y_train, y_train_pred)
            specificity_test = specificity_score(y_test, y_test_pred)
            precision_train = precision_score(y_train, y_train_pred)
            precision_test = precision_score(y_test, y_test_pred)
            accuracy_train = accuracy_score(y_train, y_train_pred)
            accuracy_test = accuracy_score(y_test, y_test_pred)
            f1_train = f1_score(y_train, y_train_pred)
            f1_test = f1_score(y_test, y_test_pred)
            auc_train = roc_auc_score(y_train, y_train_pred)
            auc_test = roc_auc_score(y_test, y_test_pred)
            
            # write
            out_list = [
                i,
                recall_train,
                specificity_train,
                precision_train,
                accuracy_train,
                f1_train,
                auc_train,
                recall_test,
                specificity_test,
                precision_test,
                accuracy_test,
                f1_test,
                auc_test,
            ]

            f.write('\t'.join([str(i) for i in out_list]))
            f.write("\n")

    sys.exit()


def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).flatten()
    if tn == 0: return 0
    return tn / (tn + fp)


if __name__ == "__main__":
    main()
