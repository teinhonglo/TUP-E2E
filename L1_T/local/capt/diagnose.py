#this code is for dealing recognize results to get which phone has error
import json
import re
import numpy as np
import numpy as np
import tabulate as tb 
import argparse
parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--transcript_filename",
                     default="trans_origin.txt",
                     type=str)

parser.add_argument("--recog_filename",
                     default="data.json",
                     type=str)

parser.add_argument("--TR_filename",
                     default="TR_utt.txt",
                     type=str)

args = parser.parse_args()

TR_filename=args.TR_filename
transcript_filename=args.transcript_filename
recog_filename=args.recog_filename
print(TR_filename)

def wagner_fischer(word_1, word_2):
    n = len(word_1) + 1  # counting empty string 
    m = len(word_2) + 1  # counting empty string

    # initialize D matrix
    D = np.zeros(shape=(n, m), dtype=np.int)
    D[:,0] = range(n)
    D[0,:] = range(m)

    # B is the backtrack matrix. At each index, it contains a triple
    # of booleans, used as flags. if B(i,j) = (1, 1, 0) for example,
    # the distance computed in D(i,j) came from a deletion or a
    # substitution. This is used to compute backtracking later.
    B = np.zeros(shape=(n, m), dtype=[("del", 'b'), 
                      ("sub", 'b'),
                      ("ins", 'b')])
    B[1:,0] = (1, 0, 0) 
    B[0,1:] = (0, 0, 1)

    for i, l_1 in enumerate(word_1, start=1):
        for j, l_2 in enumerate(word_2, start=1):
            deletion = D[i-1,j] + 1
            insertion = D[i, j-1] + 1
            substitution = D[i-1,j-1] + (0 if l_1==l_2 else 2)

            mo = np.min([deletion, insertion, substitution])

            B[i,j] = (deletion==mo, substitution==mo, insertion==mo)
            D[i,j] = mo
    return D, B

def naive_backtrace(B_matrix):
    i, j = B_matrix.shape[0]-1, B_matrix.shape[1]-1
    backtrace_idxs = [(i, j)]

    while (i, j) != (0, 0):
        if B_matrix[i,j][1]:
            i, j = i-1, j-1
        elif B_matrix[i,j][0]:
            i, j = i-1, j
        elif B_matrix[i,j][2]:
            i, j = i, j-1
        backtrace_idxs.append((i,j))

    return backtrace_idxs

def align(word_1, word_2, bt):

    aligned_word_1 = []
    aligned_word_2 = []
    operations = []

    backtrace = bt[::-1]  # make it a forward trace
    for k in range(len(backtrace) - 1): 
        i_0, j_0 = backtrace[k]
        i_1, j_1 = backtrace[k+1]
        w_1_letter = None
        w_2_letter = None
        op = None

        if i_1 > i_0 and j_1 > j_0:  # either substitution or no-op
            if word_1[i_0] == word_2[j_0]:  # no-op, same symbol
                w_1_letter = word_1[i_0]
                w_2_letter = word_2[j_0]
                op = "T"
            else:  # cost increased: substitution
                w_1_letter = word_1[i_0]
                w_2_letter = word_2[j_0]
                op = "F"
        elif i_0 == i_1:  # insertion
                w_1_letter = " "
                w_2_letter = word_2[j_0]
                op = "*"
        else: #  j_0 == j_1,  deletion
            w_1_letter = word_1[i_0]
            w_2_letter = " "
            op = "F"

        aligned_word_1.append(w_1_letter)
        aligned_word_2.append(w_2_letter)
        operations.append(op)

    return aligned_word_1, aligned_word_2, operations

# =================== MAIN ===================

each_utt_dict={}
TR_dict={}

with open(TR_filename) as lines:
    for line in lines:
        nline = line.split(' ', 1)
        TR_dict[nline[0]] = nline[1].split()

with open(transcript_filename) as lines:
    for line in lines:
        nline = line.split(' ', 1)
        each_utt_dict[nline[0]] = nline[1]
count=0
ini_TD=0
ini_FD=0
fin_TD=0
fin_FD=0
tone_TD=0
tone_FD=0

def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

def split_final(p_diag,r_diag):
    pre_diag = re.sub('{|}','',p_diag)
    real_diag = re.sub('{|}','',r_diag)
    num_start=0
    for p in range(len(pre_diag)):
        if(pre_diag[p].isdigit() == True):
            num_start=p
            break
    pre_fin_phone = pre_diag[:num_start]
    pre_tone = pre_diag[num_start:]
    num_start2=0
    for p in range(len(real_diag)):
        if(real_diag[p].isdigit() == True):
            num_start2=p
            break
    real_fin_phone = real_diag[:num_start2]
    real_tone = real_diag[num_start2:]
    return(pre_fin_phone,real_fin_phone,pre_tone,real_tone)    

with open(recog_filename) as handle:
    dictdump = json.loads(handle.read())
    for key in TR_dict:
        pre_ans=[]
        print(str(key)+" ",end="")
        for j in range(len(dictdump["utts"][key]["output"])-4):
            str1=re.sub('<eos>','',dictdump["utts"][key]["output"][j]["rec_token"]).split()
            nstr1=list(filter(('<unk>').__ne__, str1))
            #each_score_list = dictdump["utts"][key]["output"][j]["each_score"]
            all_pos = []
            rec_token_list = dictdump["utts"][key]["output"][j]["rec_token"].split()
            for i in range(len(rec_token_list)):
                if(rec_token_list[i]=="<eos>" or rec_token_list[i]=="<unk>"):
                    all_pos.append(i)
            str2=each_utt_dict[key].split()           
            D, B = wagner_fischer(str2, nstr1)
            bt = naive_backtrace(B)
            alignment_table = align(str2, nstr1, bt)
            print("\nAlignment:")
            print(tb.tabulate(alignment_table, tablefmt="orgtbl"))
            print("\ndiag:")   
            TR_list = TR_dict[key]
            str_real_diag=re.sub('}{','} {',dictdump["utts"][key]["output"][j]["text"]).split()
            print(str_real_diag)
            for kk in range(len(TR_list)):
                if(alignment_table[1][int(TR_list[kk])]==" " and kk<len(str_real_diag)):
                    no_diag ="{unk}"
                    print("pre_diag:",pre_diag)
                    print("act_diag:",str_real_diag[int(TR_list[kk])])
                    if(hasNumbers(str_real_diag[int(TR_list[kk])])==True and str_real_diag[int(TR_list[kk])]=="{0}"):
                        ini_TD+=1
                    elif(hasNumbers(str_real_diag[int(TR_list[kk])])==False):
                        ini_FD+=1
                    if(hasNumbers(str_real_diag[int(TR_list[kk])])==True and str_real_diag[int(TR_list[kk])]=="{00}"):
                        pre_diag = re.sub('{|}','',no_diag)
                        real_diag = re.sub('{|}','',str_real_diag[int(TR_list[kk])-1])
                        pre_fin_phone,real_fin_phone,pre_tone,real_tone = split_final(pre_diag,real_diag)
                        #print("pre_fin:",pre_fin_phone)
                        #print("real_fin:",real_fin_phone)
                        #print("pre_tone:",pre_tone)
                        #print("real_tone:",real_tone)
                        fin_TD+=1
                        tone_TD+=1
                    elif(hasNumbers(str_real_diag[int(TR_list[kk])])==True and str_real_diag[int(TR_list[kk])]!="{00}"):
                        pre_diag = re.sub('{|}','',no_diag)
                        real_diag = re.sub('{|}','',str_real_diag[int(TR_list[kk])])
                        pre_fin_phone,real_fin_phone,pre_tone,real_tone = split_final(pre_diag,real_diag)
                        #print("pre_fin:",pre_fin_phone)
                        #print("real_fin:",real_fin_phone)
                        #print("pre_tone:",pre_tone)
                        #print("real_tone:",real_tone)
                        fin_TD+=1
                        tone_TD+=1
                else:
                    print("pre_diag:",alignment_table[1][int(TR_list[kk])])
                    if(int(TR_list[kk])>len(str_real_diag)-1):
                        print("act_diag:",str_real_diag[int(TR_list[kk])-1])
                        if(hasNumbers(str_real_diag[int(TR_list[kk])-1])==False):
                            if(alignment_table[1][int(TR_list[kk])]==str_real_diag[int(TR_list[kk])-1]):
                                ini_TD+=1
                            else:
                                ini_FD+=1  
                        else:
                            pre_diag = re.sub('{|}','',alignment_table[1][int(TR_list[kk])])
                            real_diag = re.sub('{|}','',str_real_diag[int(TR_list[kk])-1])
                            pre_fin_phone,real_fin_phone,pre_tone,real_tone = split_final(pre_diag,real_diag)
                            #print("pre_fin:",pre_fin_phone)
                            #print("real_fin:",real_fin_phone)
                            #print("pre_tone:",pre_tone)
                            #print("real_tone:",real_tone)
                            if(pre_fin_phone==real_fin_phone):
                                fin_TD+=1
                            else:
                                fin_FD+=1
                            if(pre_tone==real_tone):
                                tone_TD+=1
                            else:
                                tone_FD+=1 
                    else:
                        print("act_diag:",str_real_diag[int(TR_list[kk])])
                        if(hasNumbers(str_real_diag[int(TR_list[kk])])==False):
                            if(alignment_table[1][int(TR_list[kk])]==str_real_diag[int(TR_list[kk])]):
                                ini_TD+=1
                            else:
                                ini_FD+=1
                        else:
                            pre_diag = re.sub('{|}','',alignment_table[1][int(TR_list[kk])])
                            real_diag = re.sub('{|}','',str_real_diag[int(TR_list[kk])])
                            pre_fin_phone,real_fin_phone,pre_tone,real_tone = split_final(pre_diag,real_diag)
                            #print("pre_fin:",pre_fin_phone)
                            #print("real_fin:",real_fin_phone)
                            #print("pre_tone:",pre_tone)
                            #print("real_tone:",real_tone)
                            if(pre_fin_phone==real_fin_phone):
                                fin_TD+=1
                            else:
                                fin_FD+=1
                            if(pre_tone==real_tone):
                                tone_TD+=1
                            else:
                                tone_FD+=1 
                count+=1
            for i in range(len(alignment_table[2])):
                if j == 0:
                    if (alignment_table[0][i]!= "<unk>" and alignment_table[1][i]!= "<unk>"):
                        pre_ans.append(alignment_table[2][i])
                else:
                    if (alignment_table[0][i]!= "<unk>" and alignment_table[1][i]!= "<unk>") and alignment_table[2][i] == "T":
                        if(i > len(pre_ans)-1):
                            pre_ans.append("T")
                        else:
                            pre_ans[i] = "T"
                            
        for  k in range(len(pre_ans)):
            if k < len(pre_ans)-1:
                print(pre_ans[k],end=",")
            else:
                print(pre_ans[k])
                pre_ans.clear()

print(count)
print("ini_TD:",ini_TD/(ini_TD+ini_FD))
print("fin_TD:",fin_TD/(fin_TD+fin_FD))
print("tone_TD:",tone_TD/(tone_TD+tone_FD))
