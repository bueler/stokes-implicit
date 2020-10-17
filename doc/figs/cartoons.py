#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"

fsize=18.0
smallfsize=14.0
bigfsize=24.0

debug = False
def figsave(name):
    print('saving %s ...' % name)
    if debug:
        plt.show()  # debug
    else:
        plt.savefig(name,bbox_inches='tight',transparent=True)

def genbasicfig(perturb=False,reference=False):
    if perturb and reference:
        raise NotImplementedError('not set up for both perturb and reference figure')
    x = np.linspace(0.0,10.0,1001)
    # bed elevation
    b = 0.07*(x-3.0)**2 + 0.2*np.sin(2.0*x)
    plt.plot(x, b, 'k', lw=2.5)
    # current thickness for Omega^{n-1}
    h0 = 3.0
    L = 3.0
    firstshape = h0*(-0.2 + np.sqrt(np.maximum(0.0,1.0 - (x-5)**2/L**2)))
    thk = np.maximum(0.0, firstshape )
    # reference thickness for Lambda
    if reference:
        href = 0.5
        thk = np.maximum(href, thk)
        plt.plot([x[0],x[0]],[b[0],b[0]+href], 'k', lw=3.0)
        plt.plot([x[-1],x[-1]],[b[-1],b[-1]+href], 'k', lw=3.0)
    # perturbed thickness for updated domain for Omega^n
    if perturb:
        plt.plot(x, b+thk, '--k', lw=2.5)
        perturbshape = 0.2 * np.sin(1.5*x) + 0.3*(x - 5.0)
        perturbshape[x>8.0] = 0.0
        thk = np.maximum(0.0, firstshape + perturbshape )
    h = b + thk
    plt.plot(x, h, 'k', lw=3.0)
    # reset axes
    plt.axis([0.0-0.02,10.0+0.02,-0.5,4.5])
    plt.axis('off')
    return x, h, b

def drawclimate(x,h,mycolor):
    for j in range(10):
        xarr = x[50+100*j]
        if j>0:
            magarr = 0.6*np.sin(np.pi/2 + 0.6*xarr)
        else:
            magarr = 0.05
        plt.arrow(xarr,h.max()+0.2,0.0,magarr,lw=1.5,head_width=0.1,color=mycolor)

# basic figure showing context of paper
plt.figure(figsize=(10,4))
x, h, b = genbasicfig()
plt.text(x[600]-1.0,b[600]+0.4*h[600],'ice',fontsize=fsize,color='k')
plt.text(x[900]-0.4,b[900]-1.0,'bedrock',fontsize=fsize,color='k')
drawclimate(x,h,'k')
plt.text(x[0]+4.0,h.max()+0.5,'surface mass balance',fontsize=fsize)
plt.annotate('free to move',fontsize=smallfsize,
             xy=(x[300],h[300]),
             xytext=(x[300]-2.3,h[300]+0.5),
             arrowprops=dict(facecolor='black', width=0.5, headwidth=5.0, shrink=0.1))
margin = 792
plt.annotate('free to move',fontsize=smallfsize,
             xy=(x[margin],h[margin]-0.1),
             xytext=(x[margin]-1.0,h[margin]-1.5),
             arrowprops=dict(facecolor='black', width=0.5, headwidth=5.0, shrink=0.1))
figsave('context.pdf')

# domain notation figure
plt.figure(figsize=(10,5.5))
genbasicfig()
plt.text(x[600]-1.0,b[600]+0.4*h[600],r'$\Omega^t$',fontsize=fsize,color='k')
drawclimate(x,h,'k')
# mark top surface
plt.annotate(r'$\overline{\partial}\Omega^t$',fontsize=fsize,
             xy=(x[300],h[300]),
             xytext=(x[300]-1.5,h[300]+0.5),
             arrowprops=dict(facecolor='black', width=0.5, headwidth=5.0, shrink=0.1))
# mark bottom surface
# BIZARRE HACK NEEDED BECAUSE \underline{} THROWS ERROR
#plt.annotate(r"$\underline{\partial}\Omega^t$",fontsize=fsize,
plt.annotate(r"$\partial\Omega^t$",fontsize=fsize,
             xy=(x[700],b[700]),
             xytext=(x[700]+1.1,b[700]-1.0),
             arrowprops=dict(facecolor='black', width=0.5, headwidth=5.0, shrink=0.1))
plt.text(x[700]+1.08,b[700]-1.02,r"$\_$",fontsize=24.0)  # HACK UNDERLINE
# show \pi\Omega^t
ypi = min(b) - 0.5
plt.plot([min(x[h>b]),max(x[h>b])],[ypi,ypi],color='k',lw=1.0)
plt.text(3.5,ypi-0.5,r'$\pi\Omega^t$',fontsize=fsize)
# show R
yR = min(b) - 1.7
plt.plot([min(x),max(x)],[yR,yR],color='k',lw=1.0)
plt.text(2.0,yR-0.5,r'$R$',fontsize=fsize)
plt.axis([0.0,10.0,yR-0.8,4.5])
figsave('domainnotation.pdf')

# current time figure
plt.figure(figsize=(10,4))
genbasicfig()
plt.text(x[600]-1.0,b[600]+0.4*h[600],r'$\Omega^{n-1}$',fontsize=bigfsize,color='k')
figsave('currenttime.pdf')

# reference domain figure
plt.figure(figsize=(10,4))
genbasicfig(reference=True)
plt.text(x[600]-1.0,b[600]+0.4*h[600],r'$\Lambda$',fontsize=bigfsize,color='k')
Href = 0.5
Hrloc = 100
plt.arrow(x[Hrloc],b[Hrloc],0.0,Href,lw=1.5,head_width=0.1,
          length_includes_head=True,color='k')
plt.arrow(x[Hrloc],b[Hrloc]+Href,0.0,-Href,lw=1.5,head_width=0.1,
          length_includes_head=True,color='k')
plt.text(x[Hrloc],b[Hrloc]+1.2*Href,r'$H_{\mathrm{ref}}$',fontsize=bigfsize)
figsave('referencedomain.pdf')

# next time figure
plt.figure(figsize=(10,4))
genbasicfig(perturb=True)
plt.text(x[600]-1.0,b[600]+0.4*h[600],r'$\Omega^{n}$',fontsize=bigfsize,color='k')
figsave('nexttime.pdf')

