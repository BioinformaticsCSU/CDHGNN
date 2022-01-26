import numpy as np

def get_trids():
    nucle_com = []
    chars = ['A', 'C', 'G', 'U']
    base=4

    if k_mer == 4:
        end = len(chars) ** 4
        for i in range(0, end):
            n = i
            ch0 = chars[n % base]
            n = n / base
            ch1 = chars[int(n % base)]
            n = n / base
            ch2 = chars[int(n % base)]
            n = n / base
            ch3 = chars[int(n % base)]
            nucle_com.append(ch0 + ch1 + ch2 + ch3)
    elif k_mer == 5:
        end = base ** 5
        for i in range(0, end):
            n = i
            ch0 = chars[n % base]
            n = n / base
            ch1 = chars[int(n % base)]
            n = n / base
            ch2 = chars[int(n % base)]
            n = n / base
            ch3 = chars[int(n % base)]
            n = n / base
            ch4 = chars[int(n % base)]
            nucle_com.append(ch0 + ch1 + ch2 + ch3 + ch4)

    elif k_mer == 6:
        end=base**6
        for i in range(0,end):
            n=i
            ch0=chars[n%base]
            n=n/base
            ch1=chars[int(n%base)]
            n=n/base
            ch2=chars[int(n%base)]
            n=n/base
            ch3=chars[int(n%base)]
            n=n/base
            ch4=chars[int(n%base)]
            n=n/base
            ch5=chars[int(n%base)]
            nucle_com.append(ch0 + ch1 + ch2 + ch3 + ch4 + ch5)
    return  nucle_com

### encoding sequence
def get_nucleotide_composition(tris, seq):
    seq_len = len(seq)
    tri_feature = []
    k = len(tris[0])
    #tmp_fea = [0] * len(tris)
    for x in range(len(seq) + 1- k):    # k determines the stride of slidding window
        kmer = seq[x:x+k]
        if kmer in tris:
            ind = tris.index(kmer)  # from 0
            tri_feature.append(str(ind))
        else:
            tri_feature.append(-1)
    #tri_feature = [float(val)/seq_len for val in tmp_fea]
        #pdb.set_trace()
    return np.asarray(tri_feature)

def read_fasta_file(fasta_file):
    seq_dict = {}
    with open(fasta_file, 'r') as file:
        for line in file.readlines():
            circ_index = line.split('\t')[0].strip()
            circ_fasta = line.split('\t')[1].strip()
            seq_dict[circ_index] = circ_fasta
    return seq_dict
