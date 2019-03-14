#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Function General two band
"""
"""
Function List: 
		(1) gene_k_list(generate k point)
		(2) 
"""


import fun_H

import numpy as np

### Definition and replacement
linspace=np.linspace
pi=np.pi
zeros=np.zeros
cos=np.cos
sin=np.sin

#>>>>>>>>>> fun_1: generate the k point along the given path
def gene_k_list(K, nk):

	nK=len(K[:,0])
	DK=np.diff(K,axis=0)
	K_norm=np.linalg.norm(DK,axis=1,keepdims=True)

	nk_seg=np.round(nk*(K_norm/np.sum(K_norm)))
	nk_lab=np.cumsum(nk_seg)
	nk_lab=np.insert(nk_lab,0,0)
	nk=np.sum(nk_seg)
	nk_seg=nk_seg.astype(np.int16)
	nk_lab=nk_lab.astype(np.int16)
	nk=nk.astype(np.int16)

	k_list=zeros((nk, 3))
	for i in range(0,nK-1):
		kx_list=linspace(K[i,0], K[i+1,0],nk_seg[i])
		ky_list=linspace(K[i,1], K[i+1,1],nk_seg[i])
		kz_list=linspace(K[i,2], K[i+1,2],nk_seg[i])
		k_list[nk_lab[i]:nk_lab[i+1],:]=np.transpose([kx_list, ky_list,kz_list])

	return  k_list, nk_lab, nk
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def gene_band(K_path, nk):

	k_list, k_label, nk = gene_k_list(K_path, nk)
	LEN = k_label[-1]
	E_list=zeros((LEN, 2))
	for i in range(0,LEN-1):
		ka=k_list[i,0];	kb=k_list[i, 1];  kc=k_list[i, 2]
		H=fun_H.Hamil_PBC(ka, kb, kc)
		D, V=np.linalg.eig(H)
		E_list[i, :]=D.real
	return E_list, k_list, k_label, nk

def gene_slab(n_uc,nk,K_path):
	
	K_path[:,2]=0
	k_list, nk_label,nk=gene_k_list(K_path, nk)
	eigvalue=zeros((2*n_uc,nk))
	for i in range(0,nk):
		ka=k_list[i,0]; kb=k_list[i,1]
		# print(ka,kb)
		H=fun_H.Hamil_slab(n_uc,ka,kb)
		D,V=np.linalg.eig(H)
		eigvalue[:,i]=D.real

	return eigvalue,nk_label,nk
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def gene_BerryPhase(K_path,nk):
	
	k_list, nk_label, nk = gene_k_list(K_path,nk)
	nkc=40
	kc_list=linspace(0,2*pi,nkc)
	# print(kc_list[-1])
	v_11=zeros((nk,nkc),dtype=complex)
	v_12=zeros((nk,nkc),dtype=complex)

	for i in range(0,nk):
		ka=k_list[i,0]; kb=k_list[i,1]
		for j in range(0,nkc):
			kc=kc_list[j]
			H=fun_H.Hamil_PBC(ka,kb,kc)
			D,V=np.linalg.eig(H)
			v_11[i,j]=V[0,0]; v_12[i,j]=V[1,0]
			if D[0]>D[1]:
				v_11[i,j]=V[0,1]; v_12[i,j]=V[1,1]
		if i==149:
			print(H)
			print(V)
			print(D)
	berryPhase=zeros((nk,1))
	for i in range(0,nk):

		v1_temp=v_11[i,:]
		v2_temp=v_12[i,:]
		v1=v1_temp[1:nkc]
		v1=np.append(v1,v1_temp[0])
		v2=v2_temp[1:nkc]
		v2=np.append(v2,v2_temp[0])
		F=np.conj(v1_temp)*v1+np.conj(v2_temp)*v2
		
		BP=np.sum(np.log(F))
		berryPhase[i]=np.mod(BP.imag,2*pi)
		
		if berryPhase[i]>2*pi-0.05:
			berryPhase[i]=0
		if berryPhase[i]<0.05:
			berryPhase[i]=0
		# print(F)
	return berryPhase, nk_label, nk
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def gene_ChernNum(kx_list,nk0):

	ky=linspace(-pi,pi,nk0)
	kz=linspace(-pi,pi,nk0)
	k_list=zeros((nk0**2,2))

	for i in range(0,nk0):
		k_list[(i-1)*nk0+1:i*nk0,:]=[ky[i]*np.ones[nk0,1], kz]

	nkk=len(k_list[:,0])
	BCu=zeros((nkk,1))
	nk=len(kx_list)
	ChernNum=zeros((nk,1))

	for j in range(0,nk):
		for i in range(0,nkk):
			M=fun_H.Hamil_Chern(K_3D)
			BCu[i]=-1/(4*pi)*np.linalg.det(M)/(d1**2+d2**2+d3**2)**(3/2)
		ChernNum[j]=(2*pi/nk0)**2*np.sum(BCu)

	thres=0.02
	ChernNum[ChernNum>1+thres]=1;
	ChernNum[ChernNum<-1-thres]=-1

	return ChernNum
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def gene_surf_Green(omega, K):
	
	C1=1
	C2=0.5
	C3=1
	C4=0.833
	CAA=0.2
	CBB=0.01
	CGA=0.77
	CGB=1

	t1=-C1
	t2=-C2
	t3=-C3
	t4=-C4
	tA=-CAA
	tB=-CBB

	a=np.transpose([1,0,0])
	b=np.transpose([0,1,0])
	c=np.transpose([0,0,1])

	H0_AB=t1+t2*exp(-1j*np.dot(K,b-a))+t3*exp(-1j*np.dot(K,a))
	H0_AA = C1+C2+C3+C4+CGA+2*CAA
	H0_BB = C1+C2+C3+C4+CGB+2*CBB
	H00=[[H0_AA, H0_AB], [np.conj(H0_AB).T, H0_BB]]

	H1_AB=t4*exp(-1j*K*(c-a))
	H01=[[tA,H1_AB], [np.conj(H1_AB).T, tB]]
	H10=np.conj(H01).T

	a=H01
	ve=H00
	b=H10
	ves=H00

	thres=1e-10
	eta=1e-12

	N=100

	for i in range(0,N-1):
		temp=(omega*np.eye(2)-ve+1j*eta)**(-1)
		ve=ve+np.dot(b,np.dot(a,temp))+np.dot(a,np.dot(b,temp))
		ves=ves+np.dot(a,np.dot(b,temp))
		a=np.dot(a,np.dot(a,temp))
		b=np.dot(b,np.dot(b,temp))

		if np.sum(np.sum(abs(a)))<thres and np.sum(np.sum(abs(b)))<thres:
			break

		if i==N:
			disp(i)

	return ves
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>