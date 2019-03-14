#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------------------
Name:      NodalLine.py
Purpose:  To calculate and plot the nodal point, slab band, and Berry phase
                of a two band tight-binding model
Author:    Kaifa Luo (luokaifa96@gmail.com)

 Created:   10-2017
 Licence:   Free
 Version:   0.1  Transforming the procedure into a callable one in order
                         to call them from a higher level script.
---------------------------------------------------------------------------------------------
"""

###### Part 1. NodalLine
###### Part 2. Band
###### Part 3. Slab
###### Part 4. Berry phase

import fun_band # The basic function for the band plotting
import fun_H     # The Slab Hamiltionian and Bulk Hamiltonian

import numpy as np
import matplotlib.pyplot as plt 
import scipy.io as sio
from mpl_toolkits.mplot3d import Axes3D

# Definition and Replacement
pi=np.pi
cos=np.cos
sin=np.sin
linspace=np.linspace

#****************************MAIN PROGRAM******************************************

fig = plt.figure()   #Main figure
data=sio.loadmat('./input/node.mat'); path=data['PATH']; node=data['NL']   # load data


##>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# #=>>>>>>>>>>>>>>>>>>> Part I: NodalLine

NL= fig.add_subplot(221, projection='3d')

#Nodal line and its projection on the xy, yz and xz plane
cl=[0.5, 0.5, 0.5]; sz=0.1
NL.scatter(node[:,0],node[:,1],(-pi)*np.ones((1,len(node[:,0]))), s=sz,c=cl)
NL.scatter(node[:,0], (2*pi)*np.ones((1,len(node[:,0]))),node[:,2],s=sz,c=cl)
NL.scatter((-pi)*np.ones((1,len(node[:,0]))),node[:,1],node[:,2], s=sz,c=cl)
NL.scatter(node[:,0],node[:,1],node[:,2],s=sz,c='r')

# plot the xy plane
k_range=np.array([[-pi,pi],[0,2*pi],[-pi,pi]])
nk=800
kx=linspace(k_range[0,0],k_range[0,1],nk)
ky=linspace(k_range[1,0],k_range[1,1],nk)
kz=linspace(k_range[2,0],k_range[2,1],nk)
x, y =np.meshgrid(kx, ky)
z = x*0
NL.plot_surface(x,y,z,cmap=plt.cm.coolwarm,alpha=0.45)

# plot the axis left
shift=0.1
NL.plot([pi,pi],[0,0],[-pi-shift,pi],'--',c=cl)
NL.plot([-pi-shift,pi],[0,0],[pi,pi],'--',c=cl)
NL.plot([pi,pi],[0,2*pi+shift],[pi,pi],'--',c=cl)

 # axis properties
NL.set_xlim(-pi, pi)
NL.set_ylim(0, 2*pi)
NL.set_zlim(-pi,pi)
NL.set_xticks([-pi, 0, pi])
NL.set_yticks([0, pi, 2*pi])
NL.set_zticks([-pi, 0, pi])
NL.set_xticklabels(['-$\pi$', '0', '$\pi$'])
NL.set_yticklabels(['0', '$\pi$', '2$\pi$'])
NL.set_zticklabels(['-$\pi$', '0', '$\pi$'])


#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# #==>>>>>>>>>>>>>>>>>>> Part II: Band structure of the given path

Band= fig.add_subplot(222)

cl=[0.5, 0.5, 0.5]; sz=0.1
nk=1500

#-----------------------------------------------------------------------------------
#>>>>>>>>>>>>>>>function:gene_band of the fun_band
# Kpath
# nK=len(path[:,0])
# DK=np.diff(path,axis=0)
# K_norm=np.linalg.norm(DK,axis=1,keepdims=True)

# nk_seg=np.round(nk*(K_norm/np.sum(K_norm)))
# nk_lab=np.cumsum(nk_seg)
# nk_lab=np.insert(nk_lab,0,0)
# nk=np.sum(nk_seg)
# nk_seg=nk_seg.astype(np.int16)
# nk_lab=nk_lab.astype(np.int16)
# nk=nk.astype(np.int16)
# # print(nk_seg,nk_lab,nk)

# k_list=np.zeros((nk, 3),dtype=complex)
# k=0
# for i in range(0,nK-1):
# 	kx_list=linspace(path[i,0], path[i+1,0],nk_seg[i])
# 	ky_list=linspace(path[i,1], path[i+1,1],nk_seg[i])
# 	kz_list=linspace(path[i,2], path[i+1,2],nk_seg[i])
# 	k=k+len(kx_list)
# 	k_list[nk_lab[i]:nk_lab[i+1],:]=np.transpose([kx_list, ky_list,kz_list])
# 	# print(nk_seg[i],nk_lab[i],nk_lab[i+1],path[i,:],path[i+1,:])
# # print(nK,nk_seg[nK-2],k)
# # print(path)
# # print(k_list[1070:1500,:])

# #####
# LEN = nk_lab[-1]
# E_list=np.zeros((LEN, 2))
# for i in range(LEN):
# 	ka=k_list[i,0];	kb=k_list[i, 1];  kc=k_list[i, 2]
# 	H=fun_H.Hamil_PBC(ka, kb, kc)
# 	D,V=np.linalg.eig(H)
# 	print(D)
# 	E_list[i, :]=D.real
#-------------------------------------------------------------------------------------

E_list, k_list, nk_label, nk = fun_band.gene_band(path, nk)

### Band Plotting
x=linspace(0,nk-1,nk)
for i in range(2):
	Band.scatter(x,E_list[:,i],s=sz,c='r')

### Band postsetting
for i in range(1,len(nk_label)-1):
	Band.plot([nk_label[i],nk_label[i]],[0,13],'--',c=cl)

font = {'family' : 'Times New Roman','weight' : 'normal','size':15,}
Band.set_xlim([0,1500])
Band.set_ylim([0,13])
Band.set_xticks(nk_label)
Band.set_xticklabels(['$\Gamma$', 'M', 'A','Y','B','M','$\Gamma$'],font)
Band.set_ylabel('$\omega^{-2}$',font)
Band.set_title('Band structure',font)


#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#==>>>>>>>>>>>>>>>>>>> Part III: Surface states of the given path

Surf= fig.add_subplot(223)
nk=1500
n_uc=40

# path[:,2]=0
# k_list, nk_label,nk=fun_band.gene_k_list(path, nk)
# eigvalue=np.zeros((2*n_uc,nk),dtype=complex)
# for i in range(0,nk):
# 	ka=k_list[i,0]; kb=k_list[i,1]
# 	H=Hamil.Hamil_slab(n_uc,ka,kb)
# 	# if i==30:
# 	# 	print(path)
# 	# 	print(H)
# 	# 	print(H[2*i+2:2*i+4,2*i:2*i+2])
	
# 	D,V=np.linalg.eig(H)
# 	eigvalue[:,i]=D

eigvalue,nk_label,nk=fun_band.gene_slab(n_uc, nk, path)
# eg_rg=[0.388*nk,0.612*nk]

x=linspace(0,nk-1,nk)
for i in range(0,2*n_uc):
	Surf.scatter(x,eigvalue[i,:],c=[0.5,0.5,0.5],s=0.01)
	# if i==n_uc or i==n_uc+1:
	# 	Surf.plt(eg_rg.eigvalue[i,eg_rg],'r')


### Band postsetting
for i in range(1,len(nk_label)-1):
	Surf.plot([nk_label[i],nk_label[i]],[0,13],'--',c=[0.5,0.5,0.5])

font = {'family' : 'Times New Roman','weight' : 'normal','size':15,}
Surf.set_xlim([0,1500])
Surf.set_ylim([4,8])
Surf.set_xticks(nk_label)
Surf.set_xticklabels(['$\Gamma$', 'M', 'A','Y','B','M','$\Gamma$'],font)
Surf.set_ylabel('$\omega^{-2}$',font)
Surf.set_title('Surface States',font)

#==>>>>>>>>>>>>>>>>>>> Part IV: BerryPhase

BP= fig.add_subplot(224)

nk=500
path[:,2]=0
BerryPhase, nk_label, nk=fun_band.gene_BerryPhase(path,nk)
# np.save('./output/BerryPhase.npy',BerryPhase)
# np.load('./output/BerryPhase.npy')
print(BerryPhase)

x=linspace(0,nk,nk)
print(len(x))
BP.scatter(x,BerryPhase,c='r',s=15)

for i in range(1,len(nk_label)-1):
	BP.plot([nk_label[i],nk_label[i]],[0,2*pi],'--',c=[0.5,0.5,0.5])

font = {'family' : 'Times New Roman','weight' : 'normal','size':15}
font1 = {'family' : 'Times New Roman','weight' : 'normal','size':10}
BP.set_xlim([0,nk])
BP.set_ylim([0,2*pi])
BP.set_yticks([0,pi,2*pi])
BP.set_yticklabels(['0','$\pi$','$2\pi$'])
BP.set_xticks(nk_label)
BP.set_xticklabels(['$\Gamma$', 'M', 'A','Y','B','M','$\Gamma$'],font)
BP.set_ylabel('$Berry Phase$',font1)


##=============================================
## Show the whole picture
# plt.savefig('./output/nodalline.png')
plt.show()
#==============================================

#*********************************************************************