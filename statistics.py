import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import sys
import glob
import seaborn as sns; sns.set()
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


    spear_list = spearmanr(df)
    kendall_list = kendall(df)

    with open("spear_list.txt", "w") as f:
        for s in spear_list:
            f.write('\t'.join([str(i) for i in s[:3]]))
            f.write("\n")

    with open("kendall_list.txt", "w") as f:
        for s in kendall_list:
            f.write('\t'.join([str(i) for i in s[:3]]))
            f.write("\n")


if __name__ == "__main__":
    main()
