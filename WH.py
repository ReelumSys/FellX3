# -*- coding: utf-8 -*-
#import package
import numpy as np
import pandas as pd
import io
import matplotlib.pyplot as plt
#import subprocess
import sys
sys.path.append('C:/Users/Vladi/Desktop/RelX/powerxrd')
#C:/Users/Vladi/Desktop/RelX/powerxrd
import csv
from itertools import repeat
import os
import time

from numpy import*
from scipy import*
from matplotlib import pylab as plt
from scipy.optimize import fmin
from scipy.optimize import curve_fit


import contextlib


crash1 = 'crash1.csv'

global Ee2
global r

global m
global d

def myfiledel():
    #filename0 = 'myfilesize.txt'
    #if os.path.exists(filename0): os.remove(filename0)
    #filename = 'myfile.txt'
    #if os.path.exists(filename): os.remove(filename)
    filename2 = 'myfile2.txt'
    if os.path.exists(filename2): os.remove(filename2)

myfiledel()

def test_sch():
    
    plt.xlabel('2 $\\theta$')
    plt.title('backsub and Scherrer width calculation')
    #plt.show()

test_sch()




def test_allpeaks():

    
    data = xrd.Data('First.csv').importfile()
    chart = xrd.Chart(*data)

    chart.backsub(tol=1,show=False)
    #chart.SchPeak
    chart.allpeaks(tols=(0.09,0.8), verbose=False, show=True)
    #Printfunc2 = xrd.Chart.Printfunc(chart)
    #np.savetxt('FWHMfinal.xy', Printfunc2, fmt='%f', delimiter='\t')
    plt.xlabel('2 $\\theta$')
    #plt.suptitle('backsub & Automated Scherrer width calculation of all peaks*')
    #plt.show()



test_allpeaks()

def reel_Williamson_Hall():
    
    global m
    global d
    data = loadtxt ('myfile.txt', usecols=(0))
    data2 = loadtxt ('myfile2.txt', usecols=[0])
    
    print(np.radians)

    x = (data2)
    y = (data)
    wl = 0.15406
















    theta2 = np.radians(x/4)


    theta  =  np.radians(x/2)

    x = sin(theta)

    y1= np.radians(y)
    
    y = y1*cos(theta) 
    y2 = y1*cos(theta2)


    def func(x,m,c):
        return m * x + c
    
    popt,pcov= curve_fit(func,x,y)
    popt,pcov=curve_fit()
    #print(popt)
    #print(popt2)

    m = popt[0]
    c = popt[1]
    k = 0.94
    d = k*wl/c

    y_max = amax(y)
    print('y_max', y_max)



    print('crystallite size = ',d,'nm')
    print('Lattice Strain =    ',m)

    
    #d.to_csv('CrystSize.csv')

    xx = arange(0.0,0.7,0.1)
    yy = m*xx + c

    plt.xlim(0,0.7)
    plt.ylim(0,y_max+0.003)

    plt.title('Size = %.2f nm %s Strain = %.4f'%(d,'', m))


    plt.plot((x),func(x,*popt),'b-')
    plt.plot(xx,yy,'b--')

    plt.plot(x,y,'ro')
    np.savetxt('WH-realx.txt', x, fmt='%f', delimiter='')
    np.savetxt('WH-realy.txt', y, fmt='%f', delimiter='')
    datanew = pd.read_csv('WH-realx.txt', names=['sinTheta'])
    datanew2 = pd.read_csv('WH-realy.txt', names=['BetaCosTheta'])
    


    plt.xlabel(r'sin $\theta$',fontsize = 14)
    plt.ylabel(r'$\beta$ cos $\theta$',fontsize = 14)
    plt.grid(True)
    plt.savefig('WH-PLOT.png')
    #plt.show()

    y = y1*cos(theta) 
    x = sin(theta)

    Sum = (y/x)**2
    plt.close()

    return data,data2,m,d,
reel_Williamson_Hall()

def reel_Halder_Wagner():
    global Ee2
    global r
    beta = loadtxt ('myfile.txt', usecols=(0))
    theta = loadtxt ('myfile2.txt', usecols=[0])

    x10 = beta/(tan(theta)*sin(theta))

    y11 = 1000*(((beta/(tan(theta)))**2))

    Klamba = 0.15406 * (4/3)
    EePre = 16 * Klamba

    Ee = np.sqrt(Klamba/(beta*EePre))
    print(Ee)
    #print(Ee)
    Ee2 = Ee.mean()
    Ee2 = Ee2 *100
    #x12 = x10.transpose()
    #print(x12)
    x13 = pd.DataFrame(x10)
    #print(x13)

    #y13 = y11.transpose()
    #print(y13)
    y14 = pd.DataFrame(y11)
    #print(y14)


    frames = [x13,y14]

    result = pd.concat(frames, axis=1)

    #print(result)


    
    def func(x10,r,j):
        return r * x10 + j
    
    popt,pcov= curve_fit(func,x10,y11)

    r = popt[0]/1
    j = popt[1]
    
    xx = arange(0.0,7,0.1)
    yy = r*xx + j



    y_max = amax(y11)

    
    
    plt.xlim(-2,7)
    plt.ylim(-2,y_max+0.003)

    plt.title('Size = %.2f nm %s Strain = %.4f'%(Ee2,'', r))
   
    plt.plot(x13,y14, '.')
    plt.plot(xx,yy,'b--')
    
    plt.ylabel(r'1000 * ($\beta$ / tan ($\theta$))$^2$',fontsize = 14)
    plt.xlabel(r'$\beta$ (tan ($\theta$) * sin ($\theta$))',fontsize = 14)
    plt.grid(True)
    plt.savefig('HW-PLOT.png')

    #plt.show()
    plt.close()

reel_Halder_Wagner()