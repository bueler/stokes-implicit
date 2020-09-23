#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from sys import exit

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"

fsize = 20.0

def genbasicfig(icetext='', bedtext=''):
    x = np.linspace(0.0,10.0,1001)
    b = 0.07*(x-3.0)**2 + 0.2*np.sin(2.0*x)
    plt.plot(x, b, '--k', lw=3.0)

    h0 = 3.0
    L = 3.0

    h = np.maximum(0.0, h0*(-0.2 + np.sqrt(np.maximum(0.0,1.0 - (x-5)**2/L**2))) )
    s = b + h
    plt.plot(x, s, 'k', lw=3.0)

    plt.text(x[600]-1.0,b[600]+0.4*h[600],icetext,fontsize=fsize,color='k')
    plt.text(x[900]-0.4,b[900]-1.0,bedtext,fontsize=fsize,color='k')

    plt.axis([0.0,10.0,-0.5,4.5])
    plt.axis('off')
    return x, s, b

def drawclimate(x,s,mycolor,climatetext=''):
    plt.text(x[0]-0.2,s.max()-0.7,climatetext,fontsize=fsize)
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

# basic figure
plt.figure(figsize=(10,4))
x, s, b = genbasicfig(icetext='ice',bedtext='bedrock')
drawclimate(x,s,'k',climatetext='surface mass balance')
figsave('cartoon.pdf')

# domain notation figure
plt.figure(figsize=(10,5.5))
genbasicfig(icetext=r'$\Omega^t$',bedtext='')
drawclimate(x,s,'k',climatetext='')
# mark top surface
plt.annotate(r'$\overline{\partial}\Omega^t$',fontsize=fsize,
             xy=(x[300], s[300]),
             xytext=(x[300]-1.5,s[300]+0.5),
             arrowprops=dict(facecolor='black', width=0.5, headwidth=5.0, shrink=0.1))
# mark bottom surface
# BIZARRE HACK NEEDED BECAUSE \underline{} THROWS ERROR
#plt.annotate(r"$\underline{\partial}\Omega^t$",fontsize=fsize,
plt.annotate(r"$\partial\Omega^t$",fontsize=fsize,
             xy=(x[700], b[700]),
             xytext=(x[700]+1.1,b[700]-1.0),
             arrowprops=dict(facecolor='black', width=0.5, headwidth=5.0, shrink=0.1))
plt.text(x[700]+1.08,b[700]-1.02,r"$\_$",fontsize=24.0)  # HACK UNDERLINE
# show \pi \Omega^t
ypi = min(b) - 0.5
plt.plot([min(x[s>b]),max(x[s>b])],[ypi,ypi],color='k',lw=1.0)
plt.text(3.5,ypi-0.5,r'$\pi\Omega^t$',fontsize=fsize)
# show R
yR = min(b) - 1.7
plt.plot([min(x),max(x)],[yR,yR],color='k',lw=1.0)
plt.text(2.0,yR-0.5,r'$R$',fontsize=fsize)
plt.axis([0.0,10.0,yR-0.8,4.5])
figsave('domainnotation.pdf')

