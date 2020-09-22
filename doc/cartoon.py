#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from sys import exit

def genbasicfig(timestep=False, icetext='', bedtext=''):
    x = np.linspace(0.0,10.0,1001)
    b = 0.07*(x-3.0)**2 + 0.2*np.sin(2.0*x)
    plt.plot(x, b, '--k', lw=3.0)

    h0 = 3.0
    L = 3.0

    if timestep:
        hm = np.maximum(0.0, 1.1*h0*(-0.2 + np.sqrt(np.maximum(0.0,1.0 - (x-4.5)**2/L**2))) )
        sm = b + hm
        plt.plot(x, sm, 'k:', lw=3.0)

    h = np.maximum(0.0, h0*(-0.2 + np.sqrt(np.maximum(0.0,1.0 - (x-5)**2/L**2))) )
    s = b + h
    plt.plot(x, s, 'k', lw=3.0)

    if timestep:
        plt.arrow(x[750],s[750]+1.0,x[700]-x[750],-0.55,lw=1.0,head_width=0.1,color='k',
                  length_includes_head=True)
        #plt.text(x[750],s[750]+1.0,r'$u_n$',fontsize=24.0,color='k')
        plt.arrow(x[260],sm[260]+1.0,x[300]-x[260],-0.45,lw=1.0,head_width=0.1,color='k',
                  length_includes_head=True)
        plt.text(x[150],sm[250]+1.0,r'$u_{n-1}$',fontsize=24.0,color='k')
    else:
        #plt.arrow(x[600],b[600],0.0,h[600],lw=1.0,head_width=0.1,color='k',
        #          length_includes_head=True)
        #plt.arrow(x[600],s[600],0.0,-h[600],lw=1.0,head_width=0.1,color='k',
        #          length_includes_head=True)
        #plt.text(x[600]-0.4,b[600]+0.4*h[600],r'$u$',fontsize=24.0,color='k')
        #plt.arrow(x[450],b[600]+0.3*h[600],-1.0,0.0,
        #          lw=2.0,head_width=0.2,color='k',length_includes_head=True)
        #plt.text(x[400],b[600]+0.45*h[600],r'$\mathbf{q}$',fontsize=24.0,color='k')
        plt.text(x[600]-1.0,b[600]+0.4*h[600],icetext,fontsize=20.0,color='k')
        plt.text(x[900]-0.4,b[900]-1.0,bedtext,fontsize=20.0,color='k')

    plt.axis([0.0,10.0,-0.5,4.5])
    plt.axis('off')
    return x, s, b

def drawclimate(x,s,mycolor,climatetext=''):
    #plt.text(x[0]-0.5,s.max()+0.1,r'$f$',fontsize=24.0)
    plt.text(x[0]-0.2,s.max()-0.7,climatetext,fontsize=20.0)
    for j in range(10):
        xarr = x[50+100*j]
        if j>0:
            magarr = 0.6*np.sin(np.pi/2 + 0.6*xarr)
        else:
            magarr = 0.05
        plt.arrow(xarr,s.max()+0.2,0.0,magarr,lw=1.5,head_width=0.1,color=mycolor)

figdebug = False
def figsave(name):
    print('saving %s ...' % name)
    if figdebug:
        plt.show()  # debug
    else:
        plt.savefig(name,bbox_inches='tight')

# basic figure:  u, q, f
plt.figure(figsize=(10,4))
x, s, _ = genbasicfig(timestep=False,icetext='ice',bedtext='bedrock')
drawclimate(x,s,'k',climatetext='surface mass balance')
figsave('cartoon.pdf')

## time step figure:  u_n-1, u_n
#plt.figure(figsize=(10,4))
#genbasicfig(timestep=True)
#figsave('cartoon-dt.pdf')

