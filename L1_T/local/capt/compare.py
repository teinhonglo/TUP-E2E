import pandas as pd
from decimal import *
import argparse
parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--testans_filename",
                     default="test_ans.txt",
                     type=str)

parser.add_argument("--recog_TF_filename",
                     default="L2_test_TF.txt",
                     type=str)

parser.add_argument("--TR_filename",
                     default="TR_utt.txt",
                     type=str)

args = parser.parse_args()

testans_filename=args.testans_filename
recog_TF_filename=args.recog_TF_filename
TR_utt_filename = args.TR_filename

data_ans = pd.read_csv(testans_filename, sep=' ',names = ["UID","phone", "TF"])
data_predict = pd.read_csv(recog_TF_filename, sep=' ',names = ["UID", "TF"])
TA=0
FA=0
FR=0
TR=0
FA_dict = {}
FR_dict = {}
TR_utt_phone = {}
TR_dict = {}

for i in range(len(data_predict)):
    uid=data_predict["UID"][i]
    ans = (data_ans.loc[data_ans['UID'] == uid]["TF"]).tolist()
    ans_phone = (data_ans.loc[data_ans['UID'] == uid]["phone"]).tolist()
    nans_phone = []
    nans_phone =  ans_phone[0].split(",")
    print(nans_phone)
    predict = (data_predict.loc[data_predict['UID'] == uid]["TF"]).tolist()
    nans = ans[0].split(",")
    npredict = predict[0].split(",")
    print(uid)
    print(nans)
    
    nnpredict = list(filter(lambda a: a != "*", npredict))
    print(nnpredict)
    for k in range(len(nans)):
        if (len(nnpredict)>=len(nans)):
            if nans[k] == "T" and nnpredict[k]=="T":
                TA+=1
            elif nans[k] == "T" and nnpredict[k]=="F":
                FR+=1
                FR_dict[nans_phone[k]] = FR_dict.setdefault(nans_phone[k],0) + 1
            elif nans[k] == "F" and nnpredict[k]=="F":
                TR+=1
                if(uid not in TR_dict):
                    TR_dict[uid] = []
                    TR_dict[uid].append(k)
                else:
                    TR_dict[uid].append(k)
            elif nans[k] == "F" and nnpredict[k]=="T":
                FA+=1
                FA_dict[nans_phone[k]] = FA_dict.setdefault(nans_phone[k],0) + 1

    '''
    if (len(predict)>len(ans)):
        for i in range(len(ans)+1,len(predict)):
    '''



print("TA",TA)
print("FA",FA)
print("TR",TR)
print("FR",FR)
recallC = Decimal(TA/(TA+FR))
percisionC = Decimal(TA/(TA+FA))
F1C=Decimal(2*recallC*percisionC)/Decimal(recallC+percisionC)
recallM = Decimal(TR/(TR+FA))
percisionM = Decimal(TR/(TR+FR))
F1M=Decimal(2*recallM*percisionM)/Decimal(recallM+percisionM)
print("recallC:",recallC)
print("percisionC:",percisionC)
print("F1C:",F1C)
print("recallM:",recallM)
print("percisionM:",percisionM)
print("F1M:",F1M)
print("Detection acc :",Decimal(TA+TR)/Decimal(TA+FA+TR+FR))
print("False Accept Rate:",Decimal(FA)/Decimal(FA+TR))
print("False Reject Rate:",Decimal(FR)/Decimal(FR+TA))

a = sorted((value,key) for (key,value) in FA_dict.items())
b = sorted((value,key) for (key,value) in FR_dict.items())
print(a)
print(b)

with open(TR_utt_filename, 'w') as f:
    for utt in TR_dict:
        f.write(utt+" ")
        for i in range(len(TR_dict[utt])):
            if(i < len(TR_dict[utt])-1):
                f.write(str(TR_dict[utt][i])+" ")
            else:
                f.write(str(TR_dict[utt][i])+"\n")  
    
