global nop, non
from pandas import DataFrame
from pandas import Series
from keras.utils import np_utils
import os
import numpy as np
import pandas as pd


def readPeptide(pepfile, lr):
    data = []
    lr = 30 - lr
    with open(pepfile, 'r') as f:
        for line in f:
            # print(line.rstrip().split('\t')[0])
            if lr == 0:
                data.append(line.rstrip().split('\t')[0])
            else:
                data.append(line.rstrip().split('\t')[0][lr:-lr])
    return data


def find_all_index(arr, item):
    return [i for i, a in enumerate(arr) if a == item]


def command_pssm(input_file, output_file, pssm_file, DBname):
    import subprocess
    cmd1 = r'D:/GPS-uber/BLAST/blast-BLAST_VERSION+/bin/psiblast -query ' + input_file + ' -db ' + DBname + ' -num_iterations 3 -num_threads 12 -out ' + output_file + ' -out_ascii_pssm ' + pssm_file
    # wt.write(time.strftime( ISOTIMEFORMAT, time.localtime( time.time() ) )+':pssm11'+input_file+'\n')
    subprocess.check_call(cmd1, shell=True)
    # print(cmd1)


def GetPSSM(ProSeq, OutDir, PSSMDir, DBname):
    from Bio import SeqIO
    records = list(SeqIO.parse(ProSeq, "fasta"))

    global aa1
    aa1 = {}
    # aa=open('../../pssm/aa.txt','a')
    for i, item in enumerate(records):
        if not os.path.exists(r'ZID/'):
            os.makedirs(r'ZID/')
        # print(records[i].seq,records[i].seq in ppf,ppf.__len__())
        if records[i].seq in ppf:
            seq1 = records[i].seq
            # print(seq1)
            if ppf[seq1].split('\t')[1] == 'nan': continue
            input1 = ppf[seq1].split('\t')[1]
            input2 = ppf[seq1].split('\t')[2]
            if (not os.path.exists(input1)) or (not os.path.exists(input2)): continue
            out1 = PSSMDir + r'/' + records[i].id.split('|')[1] + '.pssm'
            out2 = PSSMDir[:-14] + r'/asa_result2/' + records[i].id.split('|')[1] + '.txt'
            import shutil
            shutil.copy(input1, out1)
            shutil.copy(input2, out2)
            # print(seq1,records[i].seq)
            aa1[str(seq1)] = input1
        else:
            input_file = r'ZID/' + records[i].id.split('|')[1] + '.fa'
            SeqIO.write(records[i], input_file, 'fasta')

            output_file = OutDir + r'/' + records[i].id.split('|')[1] + '.out'
            pssm_file = PSSMDir + r'/' + records[i].id.split('|')[1] + '.pssm'

            if not os.path.exists(pssm_file):
                command_pssm(input_file, output_file, pssm_file, DBname)
            aa1[str(records[i].seq)] = pssm_file


def StandardpPSSM(OldPSMMdir):
    listfile = os.listdir(OldPSMMdir)
    if listfile.__len__() == 0:
        import shutil
        shutil.copy('../1.pssm', OldPSMMdir)
    listfile = os.listdir(OldPSMMdir)
    FinPSSM, num1 = [], []
    for i, eachfile in enumerate(listfile):
        num1.append(int(eachfile.split('.')[0]))
    num = sorted(num1)
    for i, nn in enumerate(num):
        eachfile = str(nn) + '.pssm'
        with open(OldPSMMdir + '/' + eachfile, 'r') as inputpssm:
            count = 0
            Dirdata = []
            for line in inputpssm:
                count += 1
                if count <= 3:
                    continue
                if line.count('\n') == len(line):
                    break
                temp = line.strip().split()[2:22]
                Dirdata.append(temp)
            DirdataR = np.array(Dirdata)
            DirPSSM = np.reshape(DirdataR, (1, DirdataR.shape[0] * DirdataR.shape[1])).tolist()
            FinPSSM.append(DirPSSM[0])

    return FinPSSM, num


def psStandardpPSSM(OldPSMMdir):
    listfile = os.listdir(OldPSMMdir)
    if listfile.__len__() == 0:
        import shutil
        shutil.copy('../1.txt', OldPSMMdir)
    listfile = os.listdir(OldPSMMdir)
    FinPSSM, FinPSSM0, FinPSSM1, num1 = [], [], [], []
    for i, eachfile in enumerate(listfile):
        # print(eachfile)
        # num1.append(int(eachfile.split('.')[0]))
        num1.append(eachfile.split('.')[0])
    num = sorted(num1)
    namesout = []
    # global namesout
    for i, nn in enumerate(num):
        eachfile = str(nn) + '.txt'
        # print(eachfile)
        with open(OldPSMMdir + '/' + eachfile, 'r') as inputpssm:
            count = 0
            Dirdata, Dirdata0, Dirdata1, Dirdata2, Dirdata3 = [], [], [], [], []
            Dirdata4, Dirdata5, Dirdata6, Dirdata7, Dirdata8 = [], [], [], [], []
            for line in inputpssm:
                count += 1
                if count <= 1:
                    continue
                if len(line) < 2:
                    break
                # print(line)
                temp = float(line.strip().split('\t')[3])
                Dirdata.append(temp)
                temp0 = float(line.strip().split('\t')[8])
                Dirdata0.append(temp0)
                temp1 = float(line.strip().split('\t')[9])
                Dirdata1.append(temp1)
                temp2 = float(line.strip().split('\t')[10])
                Dirdata2.append(temp2)
                temp3 = float(line.strip().split('\t')[4])
                Dirdata4.append(temp3)
                temp4 = float(line.strip().split('\t')[5])
                Dirdata5.append(temp4)
                temp5 = float(line.strip().split('\t')[6])
                Dirdata6.append(temp5)
                temp6 = float(line.strip().split('\t')[7])
                Dirdata7.append(temp6)

            Dirdata3 = Dirdata0 + Dirdata1 + Dirdata2
            Dirdata8 = Dirdata4 + Dirdata5 + Dirdata6 + Dirdata7
            # if Dirdata == [] or len(Dirdata) != 61:
            # print(len(Dirdata))
            # continue
            FinPSSM.append(Dirdata)
            FinPSSM0.append(Dirdata3)
            FinPSSM1.append(Dirdata8)
            namesout.append(nn)
            # print(len(Dirdata),len(Dirdata3),len(Dirdata8))#61 183 244
    # print(FinPSSM[0],FinPSSM[1],FinPSSM[2],FinPSSM[3])
    # print(FinPSSM0[0], FinPSSM0[1], FinPSSM0[2], FinPSSM0[3])
    # print(FinPSSM1[0], FinPSSM1[1], FinPSSM1[2], FinPSSM1[3])
    # FinPSSM=np.array(FinPSSM).reshape(len(num),len(Dirdata))
    # FinPSSM0 = np.array(FinPSSM0).reshape(len(num), len(Dirdata3))
    # FinPSSM1 = np.array(FinPSSM1).reshape(len(num), len(Dirdata8))
    # print(np.array(FinPSSM).shape,np.array(FinPSSM0).shape,np.array(FinPSSM1).shape)

    return FinPSSM, FinPSSM0, FinPSSM1, num, namesout
    # FinPSSM:asa FinPSSM0:二级结构 FinPSSM1：四个角度
    # secondary structure type, ASA, φ, ψ, θ, τ, and probabilities as coil (C), sheet (E), and helix (H)


def readweight(weight_file):
    weight = None
    with open(weight_file, 'r') as f:
        for i, line in enumerate(f):
            if i == 2 - 1:
                weight = np.array([float(x) for x in line.rstrip().split('\t')])
    return weight


def read_pssm(pssm_file):
    # this function reads the pssm file given as input, and returns a LEN x 20 matrix of pssm values.
    # index of 'ACDE..' in 'ARNDCQEGHILKMFPSTWYV'(blast order)
    idx_res = (0, 4, 3, 6, 13, 7, 8, 9, 11, 10, 12, 2, 14, 5, 1, 15, 16, 19, 17, 18)
    # open the two files, read in their data and then close them
    # declare the empty dictionary with each of the entries
    aa = []
    pssm = []
    # iterate over the pssm file and get the needed information out
    with open(pssm_file) as inputpssm:
        count = 0
        for line in inputpssm:

            count += 1
            if count <= 3:
                continue
            if line.count('\n') == len(line):
                break
            temp = line.strip().split()[2:22]
            aa_temp = line.strip().split()[1]
            aa.append(aa_temp)
            pssm_temp = [-float(i) for i in temp]
            pssm.append([pssm_temp[k] for k in idx_res])
    return aa, pssm


def get_phys7(aa):
    # this function takes a path to a pssm file and finds the pssm + phys 7 input
    # to the NN in the required order - with the required window size (8).
    # define the dictionary with the phys properties for each AA
    phys_dic = {'A': [-0.350, -0.680, -0.677, -0.171, -0.170, 0.900, -0.476],
                'C': [-0.140, -0.329, -0.359, 0.508, -0.114, -0.652, 0.476],
                'D': [-0.213, -0.417, -0.281, -0.767, -0.900, -0.155, -0.635],
                'E': [-0.230, -0.241, -0.058, -0.696, -0.868, 0.900, -0.582],
                'F': [0.363, 0.373, 0.412, 0.646, -0.272, 0.155, 0.318],
                'G': [-0.900, -0.900, -0.900, -0.342, -0.179, -0.900, -0.900],
                'H': [0.384, 0.110, 0.138, -0.271, 0.195, -0.031, -0.106],
                'I': [0.900, -0.066, -0.009, 0.652, -0.186, 0.155, 0.688],
                'K': [-0.088, 0.066, 0.163, -0.889, 0.727, 0.279, -0.265],
                'L': [0.213, -0.066, -0.009, 0.596, -0.186, 0.714, -0.053],
                'M': [0.110, 0.066, 0.087, 0.337, -0.262, 0.652, -0.001],
                'N': [-0.213, -0.329, -0.243, -0.674, -0.075, -0.403, -0.529],
                'P': [0.247, -0.900, -0.294, 0.055, -0.010, -0.900, 0.106],
                'Q': [-0.230, -0.110, -0.020, -0.464, -0.276, 0.528, -0.371],
                'R': [0.105, 0.373, 0.466, -0.900, 0.900, 0.528, -0.371],
                'S': [-0.337, -0.637, -0.544, -0.364, -0.265, -0.466, -0.212],
                'T': [0.402, -0.417, -0.321, -0.199, -0.288, -0.403, 0.212],
                'V': [0.677, -0.285, -0.232, 0.331, -0.191, -0.031, 0.900],
                'W': [0.479, 0.900, 0.900, 0.900, -0.209, 0.279, 0.529],
                'Y': [0.363, 0.417, 0.541, 0.188, -0.274, -0.155, 0.476]}
    # set the phys7 data.
    phys = [phys_dic.get(i, phys_dic['A']) for i in aa]
    return phys


def window(feat, winsize=8):
    # apply the windowing to the input feature
    feat = np.array(feat)
    output = np.concatenate([np.vstack([feat[0]] * winsize), feat])
    output = np.concatenate([output, np.vstack([feat[-1]] * winsize)])
    output = [np.ndarray.flatten(output[i:i + 2 * winsize + 1]).T for i in range(0, feat.shape[0])]
    return output


def window_data(*feature_types):
    n = len(feature_types[0])
    features = np.empty([n, 0])
    for feature_type in feature_types:
        test = np.array(window(feature_type))
        features = np.concatenate((features, test), axis=1)
    return features


def sigmoid(input):
    # apply the sigmoid function
    output = 1 / (1 + np.exp(-input))
    return (output)


def nn_feedforward(nn, input):
    input = np.matrix(input)
    # find the number of layers in the NN so that we know how much to iterate over
    num_layers = nn['n'][0][0][0][0]
    # num_input is the number of input AAs, not the dimentionality of the features
    num_input = input.shape[0]
    x = input
    # for each layer up to the final
    for i in range(1, num_layers - 1):
        # get the bais and weights out of the nn
        W = nn['W'][0][0][0][i - 1].T
        temp_size = x.shape[0]
        b = np.ones((temp_size, 1))
        x = np.concatenate((b, x), axis=1)
        # find the output of this layer (the input to the next)
        xw = np.dot(x, W)
        x = sigmoid(xw)
    # for the final layer.
    # note that this layer is done serpately, this is so that if the output nonlinearity
    # is not sigmoid, it can be calculated seperately.
    W = nn['W'][0][0][0][-1].T
    b = np.ones((x.shape[0], 1))
    x = np.concatenate((b, x), axis=1)
    output = np.dot(x, W)  # x*W
    pred = sigmoid(output)
    return pred


dict_ASA0 = dict(zip("ACDEFGHIKLMNPQRSTVWY",
                     (115, 135, 150, 190, 210, 75, 195, 175, 200, 170,
                      185, 160, 145, 180, 225, 115, 140, 155, 255, 230)))


def run_iter(dict_nn, input_feature0, aa, ofile):
    SS_order = ('C' 'E' 'H')
    list1 = ('SS', 'ASA', 'TTPP')
    list_res1 = []
    for x in list1:
        nn = dict_nn[x]
        norm_max = nn['high'][0][0][0]
        norm_min = nn['low'][0][0][0]
        input_feature1 = (input_feature0 - np.tile(norm_min, (input_feature0.shape[0], 1))) / np.tile(
            (norm_max - norm_min), (input_feature0.shape[0], 1))
        r1 = nn_feedforward(nn, input_feature1)
        list_res1.append(r1)
    pred_ss_1, pred_asa_1, pred_ttpp_1 = list_res1
    SS_1 = [SS_order[i.tolist()[0][0]] for i in np.argmax(pred_ss_1, 1)]
    pred_ttpp_1_denorm = (pred_ttpp_1 - 0.5) * 2
    theta = np.degrees(np.arctan2(pred_ttpp_1_denorm[:, 0], pred_ttpp_1_denorm[:, 2]))
    tau = np.degrees(np.arctan2(pred_ttpp_1_denorm[:, 1], pred_ttpp_1_denorm[:, 3]))
    phi = np.degrees(np.arctan2(pred_ttpp_1_denorm[:, 4], pred_ttpp_1_denorm[:, 6]))
    psi = np.degrees(np.arctan2(pred_ttpp_1_denorm[:, 5], pred_ttpp_1_denorm[:, 7]))
    if ofile == 'NULL':
        return SS_1, pred_ss_1, pred_asa_1, theta, tau, phi, psi
    fp = open(ofile, 'w')
    fp.write('#\tAA\tSS\tASA\tPhi\tPsi\tTheta(i-1=>i+1)\tTau(i-2=>i+1)\tP(C)\tP(E)\tP(H)')
    fp.write('\n')
    for ind, x in enumerate(aa):
        asa = pred_asa_1[ind] * dict_ASA0.get(x, dict_ASA0['A'])

        fp.write(('%i\t%c\t%c\t%5.1f' + '\t%6.1f' * 4 + '\t%.3f' * 3) % (
            ind + 1, x, SS_1[ind], asa, phi[ind], psi[ind], theta[ind], tau[ind], pred_ss_1[ind, 0], pred_ss_1[ind, 1],
            pred_ss_1[ind, 2]))
        fp.write('\n')
    fp.close()
    return SS_1, pred_ss_1, pred_asa_1, theta, tau, phi, psi


def main(list_params, pssm_file, outfile, out_suffix):
    basenm = os.path.basename(pssm_file)
    if basenm.endswith('.pssm'):
        basenm = basenm[:-5]
    elif basenm.endswith('.mat'):
        basenm = basenm[:-4]

    outfile0 = '%s%s.%s' % (outfile, basenm, out_suffix)  #

    aa, pssm = read_pssm(pssm_file)
    pred1(list_params, aa, pssm, outfile0)


def load_NN(nn_filename):
    return np.load(nn_filename, mmap_mode=None, allow_pickle=True, fix_imports=True,
                   encoding="latin1")  # load in the NN mat file.


def pred1(list_params, aa, pssm, outfile0):
    list_nn = list_params
    phys = get_phys7(aa)
    input_feature = window_data(pssm, phys)
    ## DO FIRST PREDICTIONS
    for it1 in (1, 2, 3):
        ofile = outfile0
        if it1 < 3: ofile = 'NULL'
        dict_nn = list_nn[it1 - 1]
        res1 = run_iter(dict_nn, input_feature, aa, ofile)
        if it1 == 3: break
        ## feature after 1st iteration
        SS_1, pred_ss_1, pred_asa_1, theta, tau, phi, psi = res1
        tt_input = np.sin(np.concatenate((np.radians(theta), np.radians(tau)), axis=1)) / 2 + 0.5
        tt_input = np.concatenate(
            (tt_input, np.cos(np.concatenate((np.radians(theta), np.radians(tau)), axis=1)) / 2 + 0.5), axis=1)
        pp_input = np.sin(np.concatenate((np.radians(phi), np.radians(psi)), axis=1)) / 2 + 0.5
        pp_input = np.concatenate(
            (pp_input, np.cos(np.concatenate((np.radians(phi), np.radians(psi)), axis=1)) / 2 + 0.5), axis=1)
        ttpp_input = np.concatenate((tt_input, pp_input), axis=1)
        input_feature = window_data(pssm, phys, pred_ss_1, pred_asa_1, ttpp_input)
    return


def Splitpp(f):
    id = []
    seq = []
    pos = []
    site = []
    # pot = []
    st = ''
    i = 0
    # j = 0
    # k = 0
    with open('../../' + f, 'r') as fout:
        for line in fout:
            line = line.rstrip()
            i = i + 1
            if line.startswith('>') or '|' in line:
                if i != 1:
                    st = st.upper()
                    st = st.replace('U', '*')
                    seq.append(st)
                    st = ''
                if '|' in line:
                    id.append(line.split('|')[1])
                elif ' ' in line:
                    id.append(line.split('>')[1].split(' ')[0])
                elif '\t' in line:
                    id.append(line.split('>')[1].split('\t')[0])
                else:
                    id.append(line.split('>')[1])
            else:
                st = st + line
    st = st.upper()
    st = st.replace('U', '*')
    seq.append(st)
    for i, k in enumerate(seq):
        po = find_all_index(k, 'K')
        pos.append(po)
        sit = []
        for j, p in enumerate(po):
            if p >= 30 and (len(k) - p - 1) < 30:
                s = k[p - 30:]
                for i in range(30 - (len(k) - p - 1)):
                    s = s + "*"
            elif p < 30 and (len(k) - p - 1) < 30:
                s = k[:]
                for i in range(30 - (len(k) - p - 1)):
                    s = s + "*"
                for i in range(30 - (p)):
                    s = "*" + s
            elif p < 30 and (len(k) - p - 1) >= 30:
                s = k[:p + 31]
                for i in range(30 - (p)):
                    s = "*" + s
            else:
                s = k[p - 30:p + 31]
            sit.append(s)
        site.append(sit)
    list0, list30, list20, list10, pid, ps = [], [], [], [], [], []
    dic = f.split('.')[0][9:]
    if not os.path.exists('123/' + dic):
        os.makedirs('123/' + dic)
    os.chdir('123/' + dic)
    if not os.path.exists('asa_result'):
        os.makedirs('asa_result')
    if not os.path.exists('ZFilepathPSSM'):
        os.makedirs('ZFilepathPSSM')
    if not os.path.exists('ZFilepathresult'):
        os.makedirs('ZFilepathresult')
    with open('peptide.txt', 'w') as fout:
        for i, d in enumerate(id):
            for j, s in enumerate(site[i]):
                list30.append(s)
                list20.append(s[10:-10])
                list10.append(s[20:-20])
                pid.append(d)
                ps.append(pos[i][j] + 1)
                fout.write(s + '\t' + d + '\t' + str(pos[i][j] + 1) + '\n')
    list0.append(list30)
    list0.append(list20)
    list0.append(list10)
    os.chdir('../../')
    return list0, pid, ps, dic


def acf(list30):  # acl,aal
    # import pandas as pd
    # wt.write(time.strftime( ISOTIMEFORMAT, time.localtime( time.time() ) )+':acf00'+'\n')
    global acl, aal, acn, aan, acsl, aasl
    pos = DataFrame(list30)
    all_ = pos
    # with open('../aaindex1.txt', 'r') as fout:
    #    line=fout.readline()
    #    line=line.rstrip()[5:]
    #    aalist=line.split('\t')
    ll, j = 0, 0
    na = {}  # AAindex字典：特征->氨基酸对应值
    vvl = []  # top10特征名字列表,用于在na字典中搜索
    with open('top10.txt', 'r') as fout:
        line = fout.readline()
        line = line.rstrip()[5:]
        aalist = line.split('\t')  # aa名称列表
        for line in fout:
            ll = ll + 1
            line = line.rstrip()
            na[line.split('\t')[0]] = line.split('\t')[1:]
            vvl.append(line.split('\t')[0])

    # wt.write(time.strftime( ISOTIMEFORMAT, time.localtime( time.time() ) )+':acf100')
    # with open('../aaindex1.txt', 'r') as fout:
    #    line=fout.readline()
    #    line=line.rstrip()[5:]
    #    aalist=line.split('\t')
    #    for line in fout:
    #        j=j+1
    #        line=line.rstrip()
    #        na[j]=line.split('\t')[1:]
    # vvl1.sort(reverse=True)
    # vvl = []
    # con=range(11,31)
    # ww=open('top10.txt','w')
    # ww.write('name\t'+'\t'.join(aalist)+'\n')
    # for co in (range(30)):
    #     if co + 1 in con:
    #         llo = 0
    #     else:
    #         vvl.append(vvl1[co])
    #         ww.write(str(vvlist[vvl1[co]])+'\t'+'\t'.join(na[vvlist[vvl1[co]]])+'\n')

    def doc2num1(ss):
        sss, AAindex_Encode = [], []
        ss = ss.replace('*', '0')
        ss = ss.replace('X', '0')
        ss = ss.replace('B', '0')
        ss = ss.replace('U', '0')
        for k in vvl:
            s = list(ss)
            for i, ii in enumerate(aalist):
                s = [(na[k][i]) if x == ii else x for x in s]
            s = [float(x) for x in s]
            sss = sss + s
        return sss

    def doc2num(ss):  # ss=all_[0]=usp序列
        # ss=ss[20:-20]
        AAindex_Encode = []
        ss = ss.replace('*', '0')
        ss = ss.replace('X', '0')
        ss = ss.replace('B', '0')
        ss = ss.replace('U', '0')
        for k in vvl:
            s = list(ss)  # 将ss中的序列从dataframe转化为list存在s中
            for i, ii in enumerate(aalist):  # enumerate枚举，i是索引，ii是值：此处，i是aa名称的索引，ii是aa名称
                s = [(na[k][i]) if x == ii else x for x in s]  # k:vvl（特征） + i:aalist（aa） 一一对应
            s = [float(x) for x in s]
            AAindex_Encode.append(s)
        ACF_Encode = np.zeros((np.array(AAindex_Encode).shape[0], np.array(AAindex_Encode).shape[1]))
        for i, seq in enumerate(AAindex_Encode):
            for k, kv in enumerate(ss):
                sumValue = 0
                for j in range(0, len(seq) - k):
                    singleValue = seq[j] * seq[j + k]
                    sumValue = sumValue + singleValue
                ACF_Encode[i][k] = round(sumValue / (len(seq) - k), 2)
        ACF_Encode = ACF_Encode.flatten().tolist()
        return ACF_Encode

    # wt.write(time.strftime( ISOTIMEFORMAT, time.localtime( time.time() ) )+':acf11'+'\n')
    all_['doc2num'] = all_[0].apply(lambda ss: doc2num(ss))  # 将存在all_[0]中的序列以ss的名字导入doc2num，每次取一条序列
    # wt.write(time.strftime( ISOTIMEFORMAT, time.localtime( time.time() ) )+':acf12'+'\n')
    all_['doc2num1'] = all_[0].apply(lambda ss: doc2num1(ss))  # 将存在all_[0]中的序列以ss的名字导入doc2num1，每次取一条序列
    # wt.write(time.strftime( ISOTIMEFORMAT, time.localtime( time.time() ) )+':acf13'+'\n')
    # for i in list(all_['doc2num']):
    #     print(len(i))
    xy = np.array(list(all_['doc2num']), dtype=np.float64)
    x2 = xy.tolist()
    x = np.array(list(all_['doc2num1']), dtype=np.float64)
    x3 = x.tolist()

    # acl,aal=acf(list0[0])
    acl, aal = x2, x3
    # aal = x3
    # acn=acl
    # if  spes==9:aan=[a[100:-100] for a in aal]
    # elif spes==0 or s==8:aan=[a[200:-200] for a in aal]
    # else:aan=aal
    # acsl=ls(acl,'ACF')
    # aasl=ls(aal,'AAindex')
    return acl, aal  # acf,AAindex
    # return x3


def pssm1(list10, dic, protname):
    print('pssm1')
    global ppf
    ppf = {}
    # pp=open('./pssm/pssm.txt','r')
    # for l in pp:
    #     l=l.rstrip().split('\t')
    #     ppf[l[0]]=l[1]+'\t'+l[2]+'\t'+l[3]
    pnlist = list10
    # sp2 = ['all', 'Homo_sapiens', 'Mus_musculus', 'Saccharomyces_cerevisiae', 'Oryza_sativa', 'Rattus_norvegicus',
    #        'Escherichia_coli', 'Solanum_lycopersicum',
    #        'Bacillus_subtilis', 'Brachypodium_distachyon', 'Corynebacterium_glutamicum', 'Mycobacterium_tuberculosis',
    #        'Toxoplasma_gondii', 'Vibrio_parahaemolyticus']
    # sp2=['all']
    # for i in sp2:
    #     if i==sp2[spe]:continue
    #     if not os.path.exists('./pssm/'+i+'ps.txt'):continue
    #     pp = open('./pssm/'+i+'ps.txt', 'r')
    #     for l in pp:
    #         l = l.rstrip().split('\t')
    #         ppf[l[0]] = l[1] + '\t' + './pssm/'+l[2] + '\t' +'./pssm/'+ l[3]
    # for i in sp2:
    #     if i == sp2[spe]: continue
    #     if not os.path.exists('./pssm/aa' + i + '.txt'):continue
    #     pp = open('./pssm/aa' + i + '.txt', 'r')
    #     for l in pp:
    #         if not l.count('\t')==3:continue
    #         l = l.rstrip().split('\t')
    #         ppf[l[0]] = l[1] + '\t' + l[2] + '\t' + l[3]
    with open(dic + '/Protein.txt', 'w') as fout:
        for it, iter in enumerate(pnlist):
            fout.write('>sp|' + str(protname[it]) + '|' + '\n')
            fout.write(iter + "\n")
    ProSeq = dic + "/Protein.txt"
    OutDir = dic + '/ZFilepathresult'
    PSSMDir = dic + '/ZFilepathPSSM'
    DBname = dic + '/ZDB/DB'
    GetPSSM(ProSeq, OutDir, PSSMDir, DBname)
    # time.strftime( ISOTIMEFORMAT, time.localtime( time.time() ) )+':pssm1'+'\n')


#
# def pssm2(list10,dic,strat):
#     global ppf
#     ppf={}
#     pp=open('../../pssm/pssm.txt','r')
#     for l in pp:
#         l=l.rstrip().split('\t')
#         ppf[l[0]]=l[1]+'\t'+l[2]+'\t'+l[3]
#     pnlist = list10
#     with open(dic+'/Protein1.txt','w') as fout:
#             for it,iter in enumerate(pnlist):
#                 fout.write('>sp|'+str(it+strat)+'|'+'\n')
#                 fout.write(iter+"\n")
#     ProSeq = dic+"/Protein1.txt"
#     OutDir = dic+'/ZFilepathresult'
#     PSSMDir=dic+'/ZFilepathPSSM'
#     DBname = '../ZDB/DB'
#     GetPSSM(ProSeq,OutDir,PSSMDir,DBname)
# def pssm3(list10,dic,strat):
#     global ppf
#     ppf={}
#     pp=open('../../pssm/pssm.txt','r')
#     for l in pp:
#         l=l.rstrip().split('\t')
#         ppf[l[0]]=l[1]+'\t'+l[2]+'\t'+l[3]
#     pnlist = list10
#     with open(dic+'/Protein2.txt','w') as fout:
#             for it,iter in enumerate(pnlist):
#                 fout.write('>sp|'+str(it+strat)+'|'+'\n')
#                 fout.write(iter+"\n")
#     ProSeq = dic+"/Protein2.txt"
#     OutDir = dic+'/ZFilepathresult'
#     PSSMDir=dic+'/ZFilepathPSSM'
#     DBname = '../ZDB/DB'
#     GetPSSM(ProSeq,OutDir,PSSMDir,DBname)
def ss1(list10, dic):
    global sn, asal, ssl, asan, ssn, asasl, sssl
    OldPSMMdir = dic + '/ZFilepathPSSM'
    origin_outfile = dic + '/asa_result'
    detial_pssm = OldPSMMdir + '/'
    outfile = origin_outfile + '/'
    if not os.path.exists(outfile):
        os.makedirs(outfile)
    next_listfile = os.listdir(detial_pssm)
    # next_listfile.sort(key=lambda x: int(x[0:-5]))
    for i, pssm_file in enumerate(next_listfile):
        if os.path.exists(outfile + pssm_file.split('.')[0] + '.txt'): continue
        nndir = r'E:/Omics/SS/'
        dict1_nn = load_NN(nndir + 'pp1.npz')
        dict2_nn = load_NN(nndir + 'pp2.npz')
        dict3_nn = load_NN(nndir + 'pp3.npz')
        list_nn = (dict1_nn, dict2_nn, dict3_nn)
        main(list_nn, detial_pssm + '/' + pssm_file, outfile, r'txt')
    sn, asal, ssl, btal, namesout = ss(dic)
    # print(asal)
    # asan, ssn = asal, ssl
    # asasl =ls(asal, 'ASA')
    # sssl =ls(ssl, 'SS')
    return asal, ssl, btal, namesout  # ASA, SS, BTA


# def pssm(dic,hg):
#     global pn, pssl,psn
#     OldPSMMdir = dic+'/ZFilepathPSSM'#输出
#     FinPSSM, num =StandardpPSSM(OldPSMMdir)
#     pn, psl =num, FinPSSM
#     psn = psl
#     # pssl = ls(psl, 'PSSM')
#     #return num, FinPSSM

def ss(dic):
    OldPSMMdir = dic + '/asa_result'  # 输出
    FinPSSM, FinPSSM1, FinPSSM2, num, namesout = psStandardpPSSM(OldPSMMdir)
    xy = np.array(list(FinPSSM), dtype=np.float64)
    x2 = xy.tolist()
    xy1 = np.array(list(FinPSSM1), dtype=np.float64)
    x3 = xy1.tolist()
    xy2 = np.array(list(FinPSSM2), dtype=np.float64)
    x4 = xy2.tolist()
    return num, x2, x3, x4, namesout


# 氨基酸对组成 0、1、2、3
def kmors(list10, km, m='l'):
    # global kmn
    pos = DataFrame(list10)
    all_ = pos
    aalist = []
    if m == 'd':
        ablist = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    else:
        ablist = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'X',
                  'Y']
    for aa in ablist:
        for bb in ablist:
            aalist.append(aa + bb)

    def doc2num(s):
        ssss = []
        for k in range(km):
            alist = []
            for i, a in enumerate(s):
                if i + k + 1 < len(s):
                    alist.append(a + s[i + k + 1])
                else:
                    continue
            ss = [float(alist.count(i)) for i in aalist]
            ssss = ssss + ss[:]
        return list(ssss)

    all_['doc2num'] = all_[0].apply(lambda s: doc2num(s))
    x = np.array(list(all_['doc2num']), dtype=np.int64)
    x2 = x.tolist()
    return x2


# 氨基酸组成
def aac1(list10):
    pos = DataFrame(list10)
    all_ = pos
    aalist = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    ll = len(list10[0])

    def doc2num(s):
        s = [float(s.count(i) / ll) for i in aalist]
        s = s[:]
        return list(s)

    all_['doc2num'] = all_[0].apply(lambda s: doc2num(s))

    x = np.array(list(all_['doc2num']), dtype=np.float64)
    x2 = x.tolist()
    return x2


# list中肽段转为2进制 d控制是否序列中有U
def be1(list10, d):
    # import pandas as pd

    all_ = DataFrame(list10)
    abc = Series(range(0, 21),
                 index=['K', 'L', 'A', 'E', 'V', 'G', 'S', 'D', 'I', 'T', 'R', '*', 'P', 'Q', 'N', 'F', 'Y', 'M', 'H',
                        'C', 'W'])
    if d == 0: abc = Series(range(0, 22),
                            index=['K', 'L', 'A', 'E', 'V', 'G', 'S', 'D', 'I', 'T', 'R', '*', 'P', 'Q', 'N', 'F', 'Y',
                                   'M', 'H', 'C', 'W', 'U'])
    abc[:] = range(len(abc))
    word_set = set(abc.index)

    def doc2num(s):
        s = s.replace('X', '*')
        s = s.replace('B', '*')
        s = [i for i in s if i in word_set]
        return list(abc[s])

    all_['doc2num'] = all_[0].apply(lambda s: doc2num(s))
    for i, lew in enumerate(all_['doc2num']):
        if len(lew) != 21:
            print(all_[0][i])
    x = np.array(list(all_['doc2num']), dtype=np.int64)
    gen_matrix = lambda z: (np_utils.to_categorical(z, len(abc)).flatten())

    def data_generator(data, batch_size):
        batches = [range(batch_size * i, min(len(data), batch_size * (i + 1))) for i in
                   range(int(len(data) / batch_size + 1))]
        while True:
            for i in batches:
                xx = np.array(list(map(gen_matrix, data[i])))
            return (xx)

    x = data_generator(x[:], len(x) + 1)
    bin = x.tolist()
    return bin


# list:[k-mer]
def gps(list):
    global gpn
    for i, ii in enumerate(list):
        ii = ii.replace('U', '*')
        list[i] = ii

    # from keras.models import load_model

    def generateMMData(querylist, plist, pls_weight, mm_weight, loo=True, positive=False):
        gp = GpsPredictor(plist, pls_weight, mm_weight)

        d = []

        for query_peptide in querylist:
            d.append(gp.generateMMdata(query_peptide, loo).tolist())
        return d

    mm_weight = readweight('BLOSUM62R.txt')  # 1th is intercept

    ll = len(list[0])
    gpn = generateMMData(list, list, np.repeat(1, ll), mm_weight, loo=False, positive=False)
    # plist = readPeptide('./Protein/0.peptide',int(ll/2))

    # gpn = generateMMData(list, plist,np.repeat(1, ll), mm_weight, loo=False, positive=False)# for p in nlist]
    return gpn


class GpsPredictor(object):
    def __init__(self, plist, pls_weight, mm_weight):
        '''
        initial GPS predictor using positive training set, pls_weight vector and mm_weight vector
        :param plist: (list) positive peptides list
        :param pls_weight:  (list) pls_weight vector
        :param mm_weight:   (list) mm_weight vector
        '''
        self.alist = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y',
                      'V', 'B', 'Z', 'X', '*']
        self.plist = plist
        self.pls_weight = np.array(pls_weight).flatten()
        self.mm_weight = np.array(mm_weight).flatten()

        self.__count_matrix = self._plist_index()
        self.__mm_matrix, self.__mm_intercept = self._mmweight2matrix()

    def predict(self, query_peptide, loo=False):
        '''
        return the gps score for the query peptide
        :param query_peptide: (str) query peptide
        :param loo: (bool) if true, count_matrix will minus 1 according to the amino acid in each position in query peptide
        :return: gps score
        '''
        count_clone = self.__count_matrix * len(self.plist)
        matrix = np.zeros_like(self.__count_matrix)
        for i, a in enumerate(query_peptide):
            if a not in self.alist: a = 'C'
            if loo: count_clone[i, self.alist.index(a)] -= 1
            matrix[i, :] = self.__mm_matrix[self.alist.index(a), :]
        rm_num = 1 if loo else 0
        pls_count_matrix = (count_clone.T * self.pls_weight).T / (len(self.plist) - rm_num)
        return np.sum(matrix * pls_count_matrix) + self.__mm_intercept

    def generatePLSdata(self, query_peptide, loo=False):
        '''
        generate the pls vector of query peptide
        :param query_peptide: (str) query peptide
        :param loo: (bool) if true, the count_matrix will minus 1 according to the amino acid in each position in query peptide
        :return: (np.ndarray) the vector of feature for each position
        '''
        count_clone = self.__count_matrix * len(self.plist)
        matrix = np.zeros_like(count_clone)
        for i, a in enumerate(query_peptide):
            if a not in self.alist: a = 'C'
            if loo:
                count_clone[i, self.alist.index(a)] -= 1

            matrix[i, :] = self.__mm_matrix[self.alist.index(a), :]
        rm_num = 1 if loo else 0
        count_clone = (count_clone.T * self.pls_weight).T
        return np.sum(matrix * count_clone / (len(self.plist) - rm_num), 1)

    def generateMMdata(self, query_peptide, loo=False):
        count_clone = self.__count_matrix * len(self.plist)

        indicator_matrix = np.zeros_like(count_clone)
        for i, a in enumerate(query_peptide):
            if a not in self.alist: a = 'C'
            if loo: count_clone[i, self.alist.index(a)] -= 1
            indicator_matrix[i, self.alist.index(a)] = 1

        rm_num = 1 if loo else 0

        count_clone /= (len(self.plist) - rm_num)

        pls_count_matrix = (count_clone.T * self.pls_weight).T

        m = np.dot(indicator_matrix.T, pls_count_matrix) * self.__mm_matrix

        m += m.T

        np.fill_diagonal(m, np.diag(m) / float(2))

        iu1 = np.triu_indices(m.shape[0])

        return m[iu1]

    def getcutoff(self, randompeplist, sp=[0.98, 0.95, 0.85]):
        '''
        return cutoffs using 10000 random peptides as negative
        :param randompeplist: (list) random generated peptides
        :param sp: (float list) sp to be used for cutoff setting
        :return: (float list) cutoffs, same lens with sp
        '''
        rand_scores = sorted([self.predict(p) for p in randompeplist])
        cutoffs = np.zeros(len(sp))
        for i, s in enumerate(sp):
            index = np.floor(len(rand_scores) * s).astype(int)
            cutoffs[i] = rand_scores[index]
        return cutoffs

    def _plist_index(self):
        '''
        return the amino acid frequency on each position, row: position, column: self.alist, 61 x 24
        :return: count matrix
        '''
        n, m = len(self.plist[0]), len(self.alist)
        count_matrix = np.zeros((n, m))
        for i in range(n):
            for p in self.plist:
                count_matrix[i][self.alist.index(p[i])] += 1
        return count_matrix / float(len(self.plist))

    def _mmweight2matrix(self):
        '''
        convert matrix weight vector to similarity matrix, 24 x 24, index order is self.alist
        :return:
        '''
        aalist = self.getaalist()
        mm_matrix = np.zeros((len(self.alist), len(self.alist)))
        for n, d in enumerate(aalist):
            value = self.mm_weight[n + 1]  # mm weight contain intercept
            i, j = self.alist.index(d[0]), self.alist.index(d[1])
            mm_matrix[i, j] = value
            mm_matrix[j, i] = value
        return mm_matrix, self.mm_weight[0]

    def getaalist(self):
        '''return aa-aa list
        AA: 0
        AR: 1
        '''
        aa = [self.alist[i] + self.alist[j] for i in range(len(self.alist)) for j in range(i, len(self.alist))]
        return aa


def check():
    print('-------checking-------')
    global namelist, ACF, AAindex, binary, gpn, CKSAAPs, PseAAC, ASA, SS, BTA
    i = 0
    while i < len(namelist):
        print(namelist[i])
        pssmfile = r'./ZFilepathPSSM/' + str(namelist[i]) + r'.pssm'
        if not os.path.exists(pssmfile):
            del namelist[i]
            del ACF[i]
            del AAindex[i]
            del binary[i]
            del gpn[i]
            del CKSAAPs[i]
            del PseAAC[i]
            del ASA[i]
            del SS[i]
            del BTA[i]
            i -= 1
        i += 1


def storage10features(storehouse, namelist, acf, aaindex, obc, gps, cksaap, pseaac, asa, ss, bta,
                      pssmframe, label):  # all of them are lists
    if label == 0:
        storehouse = storehouse + '10features_for_negative_data/'
    elif label == 1:
        storehouse = storehouse + '10features/'
    else:
        storehouse = storehouse + '10features_for_experiment1/'
    if not os.path.exists(storehouse):
        os.makedirs(storehouse)
    with open(storehouse + 'peptides.txt', mode='w') as f1:
        for name in namelist:
            if name in pepdic.keys():
                f1.write(name + '\n')

    def storefeature(name, featurename, feature):
        if not os.path.exists(storehouse + featurename):
            os.makedirs(storehouse + featurename)
        with open(storehouse + featurename + '/' + name + '.' + str.lower(featurename), mode='w') as file:
            for fea in feature:
                file.write(str(fea) + '\t')
            file.write('\n')

    for iter, item in enumerate(namelist):  # iter[index] item[pepname]
        if item in pepdic.keys():
            storefeature(item, 'ACF', acf[iter])
            storefeature(item, 'AAINDEX', aaindex[iter])
            storefeature(item, 'OBC', obc[iter])
            storefeature(item, 'GPS', gps[iter])
            storefeature(item, 'CKSAAP', cksaap[iter])
            storefeature(item, 'PSEAAC', pseaac[iter])
            storefeature(item, 'ASA', asa[iter])
            storefeature(item, 'SS', ss[iter])
            storefeature(item, 'BTA', bta[iter])
            storefeature(item, 'PSSM', pssmframe[iter])


def get10features(fileplace):
    with open(fileplace + '/negative_peps.txt', mode='r') as f1:
        pepnames = f1.read().rstrip().split('\n')

    def getfeatures(fileplace, featuretype, namelist):
        feature = []
        for name in namelist:
            with open(fileplace + featuretype + '/' + name + '.' + str.lower(featuretype), mode='r') as file:
                fea = file.read().rstrip().split('\t')
                feature.append(fea)
        return feature

    ACF = getfeatures(fileplace, 'ACF', pepnames)
    AAINDEX = getfeatures(fileplace, 'AAINDEX', pepnames)
    ASA = getfeatures(fileplace, 'ASA', pepnames)
    BTA = getfeatures(fileplace, 'BTA', pepnames)
    CKSAAP = getfeatures(fileplace, 'CKSAAP', pepnames)
    GPS = getfeatures(fileplace, 'GPS', pepnames)
    OBC = getfeatures(fileplace, 'OBC', pepnames)
    PSEAAC = getfeatures(fileplace, 'PSEAAC', pepnames)
    PSSM = getfeatures(fileplace, 'PSSM', pepnames)
    SS = getfeatures(fileplace, 'SS', pepnames)

    return pepnames, ACF, AAINDEX, ASA, BTA, CKSAAP, GPS, OBC, PSEAAC, PSSM, SS


'''main program
#pepdic:{ID:peptide}
namelist=list(pepdic.keys())
peplist=list(pepdic.values())

ACF,AAindex=acf(peplist)

AAindex = acf(peplist)
d=1
binary=be1(peplist,d)
gpn=gps(peplist)
km=1
CKSAAPs=kmors(peplist,km,m='l')

PseAAC=aac1(peplist)
pssm1(peplist,outdic ,namelist)
ASA, SS, BTA, namesout = ss1(peplist, outdic)
bas_feat={}
for basname in pepdic.keys():
    pssm=open(outdic+"/ZFilepathPSSM/%s.pssm"%(basname),'r').readlines()[3:-6]
    bas_feat[basname] = []
    for line in pssm:
        if line.startswith(' '):
            dat=[i for i in line.strip().split(' ')[2:] if i!='']
            dat=dat[:20]
            bas_feat[basname]+=dat

pssmframe=pd.DataFrame.from_dict(bas_feat,orient='index')
pssmframe.columns=['pssm~'+str(i) for i in pssmframe.columns]
pssmframe=np.array(pssmframe.values).astype(int)
'''
'''
text = ['K6', 'K11', 'K27', 'K29', 'K33', 'K48', 'K63', 'M1']
for f in text:
    with open(f'./transfer/%s/%s.txt' % (f, f), encoding='utf-8') as file:
        sites = file.readlines()
        pepdic = {}
        for site in sites:
            site = site.rstrip().split('\t', 1)
            pepdic[site[0]] = site[1]
        outdic = '.'
        # pepdic:{ID:peptide}
        namelist = list(pepdic.keys())
        peplist = list(pepdic.values())

        ACF, AAindex = acf(peplist)  # ACF,AAindex

        d = 0  # 存在U
        binary = be1(peplist, d)  # OBC
        gpn = gps(peplist)  # GPS
        km = 1
        CKSAAPs = kmors(peplist, km, m='l')  # CKSAAPs

        PseAAC = aac1(peplist)  # PseAAC
        pssm1(peplist, outdic, namelist)
        ASA, SS, BTA, namesout = ss1(peplist, outdic)  # ASA,SS,BTA
        bas_feat = {}

        print(len(ASA), '\t', len(SS), '\t', len(BTA))

        check()

        for basname in namelist:
            pssm = open(outdic + "/ZFilepathPSSM2/%s.pssm" % basname, 'r').readlines()[3:65]
            bas_feat[basname] = []
            for line in pssm:
                if line.startswith(' '):
                    dat = [i for i in line.strip().split(' ')[2:] if i != '']
                    dat = dat[:20]
                    bas_feat[basname] += dat

        pssmframe = pd.DataFrame.from_dict(bas_feat, orient='index')
        pssmframe.columns = ['pssm~' + str(i) for i in pssmframe.columns]
        pssmframe = np.array(pssmframe.values).astype(int)

        storage10features(f'./transfer/%s/' % f, namelist, ACF, AAindex, binary, gpn, CKSAAPs, PseAAC, ASA,
                          SS, BTA, pssmframe, 1)

        del namelist, ACF, AAindex, binary, gpn, CKSAAPs, PseAAC, ASA, SS, BTA, pssmframe, peplist, pepdic, pssm

        print('————————————now negative data————————————')

        with open(f'./transfer/%s/negative_%s.txt' % (f, f), encoding='utf-8') as file:
            sites = file.readlines()
            pepdic = {}
            for site in sites:
                site = site.rstrip().split('\t', 1)
                pepdic[site[0]] = site[1]
            outdic = '.'
            # pepdic: {ID: peptide}
            namelist = list(pepdic.keys())
            peplist = list(pepdic.values())

            ACF, AAindex = acf(peplist)  # ACF,AAindex

            d = 0  # 存在U
            binary = be1(peplist, d)  # OBC
            gpn = gps(peplist)  # GPS
            km = 1
            CKSAAPs = kmors(peplist, km, m='l')  # CKSAAPs

            PseAAC = aac1(peplist)  # PseAAC
            pssm1(peplist, outdic, namelist)
            ASA, SS, BTA, namesout = ss1(peplist, outdic)  # ASA,SS,BTA
            bas_feat = {}

            print(len(ASA), '\t', len(SS), '\t', len(BTA))

            check()

            for basname in namelist:
                pssm = open(outdic + "/ZFilepathPSSM2/%s.pssm" % basname, 'r').readlines()[3:65]
                bas_feat[basname] = []
                for line in pssm:
                    if line.startswith(' '):
                        dat = [i for i in line.strip().split(' ')[2:] if i != '']
                        dat = dat[:20]
                        bas_feat[basname] += dat

            pssmframe = pd.DataFrame.from_dict(bas_feat, orient='index')
            pssmframe.columns = ['pssm~' + str(i) for i in pssmframe.columns]
            pssmframe = np.array(pssmframe.values).astype(int)

            storage10features(f'./transfer/%s/' % f, namelist, ACF, AAindex, binary, gpn, CKSAAPs, PseAAC,
                              ASA, SS, BTA, pssmframe, 0)
'''
#pepdic:{ID:peptide}
outdic = 'E:/QD065LPSc/Ksuc'
txt = open('E:/QD065LPSc/Ksuc/pos_Ksuc.txt', mode='r')
txt = txt.readlines()
pepdic = {}
for line in txt:
    line = line.rstrip().split('\t')
    if len(line[1]) != 21:
        continue
    pepdic[line[0]] = line[1]
namelist=list(pepdic.keys())
peplist=list(pepdic.values())

ACF, AAindex=acf(peplist)

d=1
binary=be1(peplist,d)
gpn=gps(peplist)
km=1
CKSAAPs=kmors(peplist,km,m='l')

PseAAC=aac1(peplist)
pssm1(peplist,outdic ,namelist)

ASA, SS, BTA, namesout = ss1(peplist, outdic)
bas_feat={}
del_list = []
for basname in pepdic.keys():
    if not os.path.exists(outdic+"/ZFilepathPSSM/%s.pssm"%(basname)):
        del_list.append(basname)
for del_name in del_list:
    del pepdic[del_name]
for basname in pepdic.keys():
    pssm=open(outdic+"/ZFilepathPSSM/%s.pssm"%(basname),'r').readlines()[3:24]
    bas_feat[basname] = []
    for line in pssm:
        if line.startswith(' '):
            dat=[i for i in line.strip().split(' ')[2:] if i!='']
            dat=dat[:20]
            bas_feat[basname]+=dat

pssmframe=pd.DataFrame.from_dict(bas_feat, orient='index')
pssmframe.columns=['pssm~'+str(i) for i in pssmframe.columns]
pssmframe=np.array(pssmframe.values).astype(int)

storage10features('E:/QD065LPSc/Ksuc/', namelist, ACF, AAindex, binary, gpn, CKSAAPs, PseAAC, ASA, SS, BTA, pssmframe,
                  1)
