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

args = parser.parse_args()

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
                '''
                if(w_1_letter=="{x}" or w_1_letter=="{sh}"):
                    if(each_score_dict[w_2_letter][0]>=-1):
                        each_score_dict[w_2_letter].pop(0)
                        op = "T"
                    else:
                        each_score_dict[w_2_letter].pop(0)
                        op = "F"
                else:
                    each_score_dict[w_2_letter].pop(0)
                    op = "T"
                '''
            else:  # cost increased: substitution
                w_1_letter = word_1[i_0]
                w_2_letter = word_2[j_0]
                op = "F"
                '''
                if(w_1_letter=="{t}" or w_1_letter=="{l}" or w_1_letter=="{n}" or w_1_letter=="{j}" or w_1_letter=="{z}" or w_1_letter=="{d}"):
                    if(each_score_dict[w_2_letter][0]>=-0.5):
                        each_score_dict[w_2_letter].pop(0)
                        op = "F"
                    else:
                        each_score_dict[w_2_letter].pop(0)
                        op = "T"
                else:
                    each_score_dict[w_2_letter].pop(0)
                    op = "F"
                '''
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




each_utt_dict={} 
transcript_filename = args.transcript_filename
recog_filename = args.recog_filename

with open(transcript_filename) as lines:
    for line in lines:
        nline = line.split(' ', 1)
        each_utt_dict[nline[0]] = nline[1]

with open(recog_filename) as handle:
    dictdump = json.loads(handle.read())
    for key in dictdump["utts"]:
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
            #neach_score_list = np.delete(each_score_list, all_pos).tolist()
            #each_score_dict = {}
            #for i in range(len(neach_score_list)):
            #    if(nstr1[i] not in each_score_dict):
            #        a=[]
            #        each_score_dict[nstr1[i]] = a
            #        each_score_dict[nstr1[i]].append(neach_score_list[i])
            #    else:
            #        each_score_dict[nstr1[i]].append(neach_score_list[i])

            #nstr11=list(filter(('<blank>').__ne__, nstr1))
            #str2=re.sub('<space>','<unk>',dictdump["utts"][key]["output"][j]["token"]).split()
            #print(nstr1)
            str2=each_utt_dict[key].split()
            #print(str2)
            #print("nstr1:",nstr1) 
            #print("str2:",str2)           
            D, B = wagner_fischer(str2, nstr1)
            bt = naive_backtrace(B)
            alignment_table = align(str2, nstr1, bt)
            #print("\nAlignment:")
            #print(tb.tabulate(alignment_table, tablefmt="orgtbl"))
            for i in range(len(alignment_table[2])):
                if j == 0:
                    if (alignment_table[0][i]!= "<unk>" and alignment_table[1][i]!= "<unk>"):
                        pre_ans.append(alignment_table[2][i])
                        #print(alignment_table[2][i],end=",")
                    #elif i == len(alignment_table[2])-1 :
                        #print(alignment_table[2][i])
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


'''
        if len(str1) == len(str2):

            for i in range(len(str1)):
                if i != len(str1)-1:
                    if str1[i] == str2[i]:
                        print("T",end=",")
                    else:
                        print("F",end=",")  
                else:
                    if str1[i] == str2[i]:
                        print("T")
                    else:
                        print("F")
        elif len(str1) < len(str2):
            str2.remove("<unk>")
            for i in range(len(str2)):
                if i != len(str2)-1:
                    if str2[i] in str1:
                        print("T",end=",")
                        str1.remove(str2[i])
                    else:
                        print("F",end=",")  
                else:
                    if str2[i] in str1:
                        print("T")
                        str1.remove(str2[i])
                    else:
                        print("F") 
        else:
            str1.remove("<unk>")
            str2.remove("<unk>")
            print(str1)
            print(str2)
'''
