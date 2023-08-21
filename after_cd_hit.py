from _collections import defaultdict
import random


def get_usp10(seq, location):
    left = location - 1
    right = len(seq) - location
    if left >= 10 and right > 10:
        usp = seq[location - 10: location + 11]
    elif left < 10 and right > 10:
        usp = seq[0:location + 11].rjust(21, '*')
    elif left >= 10 and right <= 10:
        usp = seq[location - 10: -1].ljust(21, '*')
    else:
        usp = seq[0:location + 1].rjust(11, '*') + seq[location + 1:-1].ljust(10, '*')
    return usp


# '''
fasta = open('E:/Omics/Result/Lac_sites_for_cd_hit.fasta', mode='r')
fasta = fasta.read().split('>')[1:]
peplist = []
for fast in fasta:
    pepname = fast.split('\n')[0]
    peplist.append(pepname)
txt = open('E:/QD065LPSc/Result/pre_training_sites.txt', mode='r')
txt = txt.readlines()
Site_dict = defaultdict(list)
for line in txt:
    line = line.rstrip().split('\t')
    pep = line[0]
    site = int(line[1])
    if pep not in Site_dict.keys():
        Site_dict[pep].append(site)
    elif site not in Site_dict[pep]:
        Site_dict[pep].append(site)
    else:
        continue

Seq_dict = {}
txt = open('E:/QD065LPSc/Result/pre_training_sequences.txt', mode='r').readlines()
for line in txt:
    line = line.rstrip().split('\t')
    pep = line[0]
    seq = line[1]
    Seq_dict[pep] = seq

neg_Site_dict = defaultdict(list)
for pepname in Seq_dict.keys():
    for i, aa in enumerate(Seq_dict[pepname]):
        site = i + 1
        if aa == 'K' and site not in Site_dict[pepname]:
            if pepname not in neg_Site_dict.keys():
                neg_Site_dict[pepname].append(site)
            elif site not in neg_Site_dict[pepname]:
                neg_Site_dict[pepname].append(site)
#'''
pos_usp10 = {}
for pepname in Site_dict.keys():
    for site in Site_dict[pepname]:
        pep_site = pepname + '_' + str(site)
        usp10 = get_usp10(Seq_dict[pepname], site - 1)
        pos_usp10[pep_site] = usp10
pos_len = len(pos_usp10.keys())
pos_set = open('E:/QD065LPSc/Ksuc/pos_Ksuc.txt', mode='w')
for ps in pos_usp10.keys():
    pos_set.write(ps + '\t' + pos_usp10[ps] + '\n')
pos_set.close()
#'''

'''
neg_usp10 = {}
for pepname in neg_Site_dict.keys():
    for site in neg_Site_dict[pepname]:
        pep_site = pepname + '_' + str(site)
        usp10 = get_usp10(Seq_dict[pepname], site - 1)
        neg_usp10[pep_site] = usp10
neg_sample_list = random.sample(neg_usp10.keys(), pos_len)
neg_set = open('E:/QD065LPSc/Ksuc/neg_Ksuc.txt', mode='w')
for ns in neg_sample_list:
    neg_set.write(ns + '\t' + neg_usp10[ns] + '\n')
neg_set.close()
#'''
'''
Seq_dict = {}
Site_dict = defaultdict(list)
all_sites = open('E:/Omics/Result/crawler_get99.txt')
all_sites = all_sites.readlines()
for line in all_sites:
    line = line.rstrip().split('\t')
    pep = line[0]
    sequence = line[1]
    site = int(line[2].split('K')[1])
    if pep not in Seq_dict.keys():
        Seq_dict[pep] = sequence
    if pep not in Site_dict.keys():
        Site_dict[pep].append(site)
    elif site not in Site_dict[pep]:
        Site_dict[pep].append(site)
    else:
        continue
usp10_dict = {}
for pepname in Site_dict.keys():
    for site in Site_dict[pepname]:
        pep_site = pepname + '_' + str(site)
        usp10 = get_usp10(Seq_dict[pepname], site - 1)
        usp10_dict[pep_site] = usp10
file = open('E:/Omics/Kla/transfer/experiment_sites.txt', mode='w')
for s in usp10_dict.keys():
    file.write(s + '\t' + usp10_dict[s] + '\n')
file.close()
'''
