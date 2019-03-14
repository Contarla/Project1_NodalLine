#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Hamiltonian with PBC condition
"""
import numpy as np

### Definition and Replacement
cos=np.cos
pi=np.pi
sin=np.sin
exp=np.exp

def Hamil_PBC(ka,kb,kc):

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

	H=np.zeros((2,2),dtype=complex)
	
	dx=-C1-C2*cos(kb-ka)-C3*cos(ka)-C4*cos(kc-ka)
	dy=C2*sin(kb-ka)-C3*sin(ka)+C4*sin(kc-ka)
	dz=(CGA-CGB)/2+(CA-CB)*(1-cos(kc))
	d0=C1+C2+C3+C4+(CGA+CGB)/2+(CA+CB)*(1-cos(kc))
	H=[[d0+dz,dx-1j*dy], [dx+1j*dy, d0-dz]]

	return H


def Hamil_slab(n_uc,ka,kb):

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

	H=np.zeros((n_uc*2,n_uc*2),dtype=complex)

	h_AA=CGA+C1+C2+C3+C4+2*CA
	h_BB=CGB+C1+C2+C3+C4+2*CB
	h_AB=-C1-C2*exp(1j*(kb-ka))-C3*exp(-1j*ka)
	H1=[[h_AA, h_AB], [np.conj(h_AB), h_BB]]
	H2=[[-CA,0],[-C4,-CB]]
	# print(h_AB)

	for i in range(0,n_uc):
		H[2*i:2*i+2,2*i:2*i+2]=H1
		# H[2*i,2*i]=H1
		# H[2*i+1,2*i+1]=H1
		# print(H1)
	for i in range(0,n_uc-1):
		H[2*i:2*i+2,2*i+2:2*i+4]=H2
		H[2*i+2:2*i+4,2*i:2*i+2]=np.conj(H2).T

	return H

