#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Surface states along the [001] direction
"""

import numpy as np
import matplotlib.pyplot as plt 
import scipy.io as sio
from mpl_toolkits.mplot3d import Axes3D

# Definition and Replacement
pi=np.pi
cos=np.cos
sin=np.sin
linspace=np.linspace
zeros=np.zeros
exp=np.exp

#**********************************************************************************
path=np.load('./input/path.npy')
path[:,2]=0 

nk=1500   # number of the kpoints for the two given high-symmetry point
n_uc=40  # number of the unit_cell along [001] direction

n_label=len(path[:,2])
k_list=zeros(((n_label-1)*nk,2))
for i in range(n_label-1):
	k_list[i*nk:(i+1)*nk,0]=linspace(path[i,0],path[i+1,0],nk)
	k_list[i*nk:(i+1)*nk,1]=linspace(path[i,1],path[i+1,1],nk)

eigvalue=np.zeros((2*n_uc,nk*(n_label-1)))

C1=1
C2=2
C3=1
C4=0.833
CGA=1
CGB=CGA
CA=0
CB=0
LA=1
LB=LA

for j in range((n_label-1)*nk):

	ka=k_list[j,0]
	kb=k_list[j,1]

	H=np.zeros((n_uc*2,n_uc*2),dtype=np.complex)
    
	h_AA=CGA+C1+C2+C3+C4+2*CA
	h_BB=CGB+C1+C2+C3+C4+2*CB
	h_AB=-C1-C2*exp(-1j*(kb-ka))-C3*exp(1j*ka)
	H1=[[h_AA, h_AB], [np.conj(h_AB), h_BB]]
	H2=[[-CA,0],[-C4,-CB]]

	for i in range(0,n_uc):
		H[2*i:2*i+2,2*i:2*i+2]=H1
	for i in range(0,n_uc-1):
		H[2*i:2*i+2,2*i+2:2*i+4]=H2
		H[2*i+2:2*i+4,2*i:2*i+2]=np.conj(H2).T
	if j==2000:
		temp=H
		# print(temp)
		print(np.linalg.eig(temp))

	D,V=np.linalg.eig(H)
	eigvalue[:,j]=D.real

x=linspace(0,1,nk*(n_label-1))
for i in range(0,2*n_uc):
	plt.scatter(x,eigvalue[i,:],c=[0.5,0.5,0.5],s=0.01)

plt.ylim([4,8])
plt.show()