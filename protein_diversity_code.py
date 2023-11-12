# -*- coding: utf-8 -*-

## Input: one or more amino acid fasta files
## (see data_list, below)

from Bio import SeqIO, AlignIO
import math
import pandas as pd
import trace
import glob
import numpy as np
AAs = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']

#take in a string of letters for a given site
#return Shannon information for that site

def shannon_info(site, AAs=AAs):

    temp_info = 0

    for aa in AAs:

        if aa in site:

            temp_info += site.count(aa)/len(site) * math.log2(site.count(aa)/len(site))

    return abs(temp_info)

#Von Neumann Entropy (density matrix) - used for Shannon Entropy calculations

def vn_entropy(column,sub_matrix,matrix_label):

    column_diag = np.zeros((sub_matrix.shape[0],sub_matrix.shape[0]))

    for aa in matrix_label:
        column_diag[matrix_label.index(aa),matrix_label.index(aa)] = column.count(aa) / len(column)

    omega = column_diag * sub_matrix

    entropy = - omega * vlog_20(omega)

    return entropy.to_numpy().trace()

LG_aa_order = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']
B50_aa_order = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V','B','J','Z','X','*']

#choose substitution matrix:
#original LG
#LG = pd.read_csv("LG.csv",header=0, index_col=0)
#LG with bottom half (exclusive of the diagnal) proportions sum to 1
#Don't use this, only inflates values
#substitution_matrix = pd.read_csv("./protein_diversity/LG-1.csv",header=0, index_col=0)

B50 = pd.read_table("BLOSUM50.tab",header=None,index_col=None,names=B50_aa_order)
B50.set_index(pd.Series(B50_aa_order),inplace=True)

max_freq = np.array([[0.05,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0,0.05,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0,0,0.05,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0.05,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0.05,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0.05,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0.05,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0.05,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0.05,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0,0.05,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0,0,0.05,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0,0,0,0.05,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0,0,0,0,0.05,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0,0,0,0,0,0.05,0,0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.05,0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.05,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.05,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.05,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.05,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.05,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])

#VNE value at Shannon Entropy Max: all amino acids at frequency 0.05
omega = B50 * max_freq
ent = -(omega * np.log2(omega))

#Pairwise similarity maximum for two allele locus (C,E)
(1/B50.loc['C','E']) * 0.5

#VNE maximum at two allele locus (C,W)
max_vne = np.array([[0.090909,0],[0,0.07878788]])
omega = max_vne * np.array([[0.5,0],[0,0.5]])
ent = -(omega * np.log2(omega))

data_list = glob.glob("./*.fasta")

#calculate the diversity measures
for currentfile in data_list:
    data = AlignIO.read(currentfile,'fasta')
    print("{} measures:".format(currentfile))


    num_samples = len(data)
    ungapped_sites = 0

    shannon = 0

    #get diversity meaasure(s) for ungapped sites
    for i in range(len(data[0,:])):


        column = data[:,i]
        if '-' not in column:

            shannon += shannon_info(column)


    print("shannon-entropy: {:.4}".format(shannon/(i+1)))
