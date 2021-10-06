import numpy as np
import pickle

import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import datetime
import pandas as pd
import pickle
# from matplotlib.colors import ListedColormap
# # %matplotlib qt
# from mpl_toolkits.mplot3d import Axes3D
from timeit import default_timer as timer
import sys, getopt
import time
import multiprocessing as mp
import os
import pandas as pd
sys.path.insert(1, '../')


from saveLoadFunctions import saveResults, loadResults

#add all of the SDs under variants
mycolors1 = sns.color_palette("viridis", 10)[::2]
mytransparencyColors = sns.color_palette("viridis", 10)[1::2]
mygrey = sns.color_palette("Greys", 1)
mycolors2 =sns.color_palette("magma", 10)[::2]
N_WA = 7.615 * 10 ** (6)  # Washington state pop
N_FL = 21.48 * 10 ** (6)
pops = [N_WA, N_FL]

deaths_baseline = [739*np.ones(4), 1648*np.ones(4)]
[deathsBasWA, deathsWA, highRiskCurvesWA, deathsITWA, highRiskCurvesITWA] = loadResults('reopening_short_paper/WAstate_05_04_21_20percent.pickle')
[deathsBasFL, deathsFL, highRiskCurvesFL, deathsITFL, highRiskCurvesITFL] = loadResults('reopening_short_paper/FLstate_05_04_21_20percent.pickle')

results = [[deathsBasWA, deathsWA, highRiskCurvesWA, deathsITWA, highRiskCurvesITWA],
           [deathsBasFL, deathsFL, highRiskCurvesFL, deathsITFL, highRiskCurvesITFL]]
mytitles = ['No increased transmission', 'Increased transmission']
mylimits = ([0, 150])
mystates = ['wa', 'fl']
for ivals in range(2):
    print(mystates[ivals])
    N = pops[ivals]
    deaths = results[ivals][1]
    deathsIT = results[ivals][3]
    highRiskCurves = results[ivals][2]
    highRiskCurvesIT = results[ivals][4]
    allMinMax = [np.zeros((28*14,2)), np.zeros((28*14,2)), np.zeros((28*14,2)), np.zeros((28*14,2)), np.zeros((28*14,2))]


    deathsMean = np.mean(deaths, 0)
    deathsPerMillion = (deaths/N)*1e6
    deathsPerMillionMean = np.mean(deathsPerMillion, 0)

    deathsPerMillionIT = (deathsIT/N)*1e6
    deathsPerMillionITMean = np.mean(deathsPerMillionIT, 0)


    myref = 2
    myrefdeaths = deaths[:, myref]

    mydeaths = [deathsPerMillion, deathsPerMillionIT]
    mycurves = [highRiskCurves, highRiskCurvesIT]
    mymeanCurves = [np.zeros((392, 4)), np.zeros((392, 4))]
    allMinMax = [np.zeros((28*14,2)), np.zeros((28*14,2)), np.zeros((28*14,2)), np.zeros((28*14,2)), np.zeros((28*14,2))]
    allMinMaxIT= [np.zeros((28*14,2)), np.zeros((28*14,2)), np.zeros((28*14,2)), np.zeros((28*14,2)), np.zeros((28*14,2))]
    CIUpperPerMillion2 = [np.zeros(4), np.zeros(4)]
    CIlowerPerMillion2 = [np.zeros(4), np.zeros(4)]

    bars = [np.zeros((4,2)), np.zeros((4,2))]

    resMinMax = [allMinMax, allMinMaxIT]
    mydeathsMean = [np.zeros(4), np.zeros(4)]
    for kvals in range(2):
        temp = mydeaths[kvals]
        tempCurves = mycurves[kvals]
        tempMeanCurves = mymeanCurves[kvals]
        tempMinCurves = resMinMax[kvals]
        mymat = mydeathsMean[kvals]
        CIUpperPerMillion2temp = CIUpperPerMillion2[kvals]
        CIlowerPerMillion2temp = CIlowerPerMillion2[kvals]
        barsTemp = bars[kvals]
        for jvals in range(4):
            sortMat = np.sort(temp[:, jvals])
            temp2 = sortMat[24:976]
            mymat[jvals] = np.mean(temp2)
            CIUpperPerMillion2temp[jvals] = mymat[jvals] + (1.96 / np.sqrt(1000)) * mymat[jvals]
            CIlowerPerMillion2temp[jvals] = mymat[jvals] - (1.96 / np.sqrt(1000)) * mymat[jvals]
            barsTemp[jvals, :] = [sortMat[24], sortMat[976]]


    for jvals in range(4):
        tempCurves = highRiskCurves[jvals]
        tempCurvesIT = highRiskCurvesIT[jvals]
        for t in range(28 * 14):
            tempY = np.sort(tempCurves[t,:])
            tempYIT = np.sort(tempCurvesIT[t,:])
            tempY2 = tempY#[24:976]
            tempYIT2 = tempYIT#[24:976]
            mymeanCurves[0][t, jvals] = np.mean(tempY2)
            mymeanCurves[1][t, jvals] = np.mean(tempYIT2)
            allMinMax[jvals][t,:] = [tempY[24], tempY[976]]
            allMinMaxIT[jvals][t,:] = [tempYIT[24], tempYIT[976]]


    mywidth = 0.3
    ind = np.arange(4)
    Rlabels = ['30%\npre-COVID-19\ncontacts', '50%\npre-COVID-19\ncontacts',  '70%\npre-COVID-19\ncontacts',
               '100%\npre-COVID-19\ncontacts', '100%\npre-COVID-19\ncontacts,\nincreased\ntransmission']
    red_contacts = [0.3, 0.5, 0.7, 1]
    Rlabels2 = ['30%\npre-COVID-19\ncontacts', '50%\npre-COVID-19\ncontacts',  '70%\npre-COVID-19\ncontacts',
               '100%\npre-COVID-19\ncontacts']
    mylabels = ['30% pre-COVID-19 contacts', '50% pre-COVID-19 contacts', '70% pre-COVID-19 contacts',
                '100% pre-COVID-19 contacts', '100% pre-COVID-19 contacts,\nincreased transmission']

    mylabels2 = ['30% pre-COVID-19 contacts,\nincreased transmission', '50% pre-COVID-19 contacts,\nincreased transmission',
                 '70% pre-COVID-19 contacts,\nincreased transmission',
                 '100% pre-COVID-19 contacts,\nincreased transmission']


    mylabels0 = ['30%', '50%', '70%','100%']


    mylabels4 = ['30% increased transmission', '50% increased transmission',
                 '70% increased transmission',
                 '100% increased transmission']


    myFig = plt.figure(1, figsize=[12, 10])
    myFig.text(0.08, 0.92, 'A', fontsize=12, fontweight='bold')
    myFig.text(0.56, 0.92, 'B', fontsize=12, fontweight='bold')
    myFig.text(0.08, 0.45, 'C', fontsize=12, fontweight='bold')
    myFig.text(0.56, 0.45, 'D', fontsize=12, fontweight='bold')

    myFig.text(0.69, 0.95,'Increased transmission', fontsize = 12, fontweight = 'bold')
    myFig.text(0.19, 0.95, 'No increased transmission', fontsize=12, fontweight='bold')
    plt.subplots_adjust(left=0.1, right=0.97, top=0.9, hspace=0.42, wspace=0.25, bottom=0.1)


    mydic = {}
    for jvals in range(4):
        mydic[mylabels[jvals]] = mycolors1[jvals]
        # mydic[mylabels4[jvals]] = mycolors2[jvals]

    labels = list(mydic.keys())

    mybars = np.array([CIlowerPerMillion2[0], CIUpperPerMillion2[0]])
    mybarsIT = np.array([CIlowerPerMillion2[1], CIUpperPerMillion2[1]])
    plt.subplot(2, 2, ivals*2+1)
    # plt.bar(ind, deaths_baseline[ivals], width=mywidth, color=mygrey)
    plt.bar(ind, mydeathsMean[0], width=mywidth, yerr=0.5 * (bars[0][:, 1] - bars[0][:, 0]), color=mycolors1)#,bottom=deaths_baseline[ivals])
    plt.xticks(ind, Rlabels2, fontsize=10, fontweight='bold')
    plt.ylabel('Deaths over next 6 months\n(per 1 million)', fontsize=10,
               fontweight='bold')
    handles = [plt.Rectangle((0,0),1,1, color=mydic[label]) for label in labels]
    # if ivals == 0:
    plt.legend( handles, labels, loc=2, fontsize=10)
        # plt.title('No increased transmission',fontsize=12, fontweight='bold')
    plt.ylim([0,800])

    plt.subplot(2, 2, ivals*2+2)
    # plt.bar(ind, deaths_baseline[ivals], width=mywidth, color=mygrey)
    plt.bar(ind, mydeathsMean[1], yerr=0.5 * (bars[1][:, 1] - bars[1][:, 0]), width=mywidth, color=mycolors1)#, bottom=deaths_baseline[ivals])
    plt.xticks(ind, Rlabels2, fontsize=10, fontweight='bold')
    plt.ylabel('Deaths over next 6 months\n(per 1 million)', fontsize=10,
               fontweight='bold')
    handles = [plt.Rectangle((0,0),1,1, color=mydic[label]) for label in labels]
    # if ivals == 0:
    plt.legend(handles, labels, loc=2, fontsize=10)
        # plt.title('Increased transmission', fontsize=12, fontweight='bold')
    plt.ylim([0, 800])
    # myFig.savefig('figures2D/texas/florida_wa_05_04_21_20percent_deaths.eps')


    dates = ["05/04/21", "06/04/21", "07/04/21", "08/04/21", "09/04/21", "10/04/21", "11/04/21"]

    myFig = plt.figure(2, figsize=[12, 10])
    myFig.text(0.08, 0.92, 'A', fontsize=12, fontweight='bold')
    myFig.text(0.56, 0.92, 'B', fontsize=12, fontweight='bold')
    myFig.text(0.08, 0.45, 'C', fontsize=12, fontweight='bold')
    myFig.text(0.56, 0.45, 'D', fontsize=12, fontweight='bold')

    myFig.text(0.69, 0.95, 'Increased transmission', fontsize=12, fontweight='bold')
    myFig.text(0.19, 0.95, 'No increased transmission', fontsize=12, fontweight='bold')
    plt.subplots_adjust(left=0.1, right=0.97, top=0.9, hspace=0.42, wspace=0.25, bottom=0.1)
    ax0 = plt.subplot(2, 2, ivals*2+1)
    # ax0.set_rasterized(True)
    tspanPlotH = np.linspace(0, 392 / 2, 392)
    for kvals in (range(4)):
        plt.fill_between(tspanPlotH, (allMinMax[kvals][:, 0] / N) * 1e5, (allMinMax[kvals][:, 1] / N) * 1e5,
                         color=mycolors1[kvals], alpha=0.3 )
        plt.plot(tspanPlotH, (mymeanCurves[0][:, kvals] / N) * 1e5, color=mycolors1[kvals], label=mylabels[kvals], linewidth=3,
                 alpha=1)

    plt.ylabel('Prevalence of active infections\n(per 100,000)', fontsize=10,
               fontweight='bold')
    plt.xlabel('Days', fontsize=10, fontweight='bold')
    plt.ylim([(mymeanCurves[1][0, kvals] / N) * 1e5, 4200])
    plt.xlim([-0.5, tspanPlotH[-1]])
    plt.legend(loc=1, fontsize=10)
    plt.xticks([0, 31, 61, 92, 122, 153, 183], dates, fontweight='bold', rotation=45, fontsize=8)



    ax = plt.subplot(2, 2, ivals*2+2)
    ax.set_rasterized(True)
    tspanPlotH = np.linspace(0, 392 / 2, 392)
    for kvals in (range(4)):
        plt.fill_between(tspanPlotH, (allMinMaxIT[kvals][:, 0] / N) * 1e5, (allMinMaxIT[kvals][:, 1] / N) * 1e5,
                         color=mycolors1[kvals], alpha=0.3)
        plt.plot(tspanPlotH, (mymeanCurves[1][:, kvals] / N) * 1e5, color=mycolors1[kvals], label=mylabels[kvals], linewidth=3,
                 alpha=1)

    plt.ylabel('Prevalence of active infections\n(per 100,000)', fontsize=10,
               fontweight='bold')
    plt.xlabel('Days', fontsize=10, fontweight='bold')
    plt.ylim([(mymeanCurves[1][0, kvals] / N) * 1e5, 4200])
    plt.xlim([-0.5, tspanPlotH[-1]])
    plt.legend(loc=1, fontsize=10)
    plt.xticks([0, 31, 61, 92, 122, 153, 183], dates, fontweight='bold', rotation=45, fontsize=8)
    # myFig.savefig('figures2D/texas/florida_wa_05_04_21_20percent_epi_curves.pdf')
    #
    print(mydeathsMean[0])
    print(mydeathsMean[0]/mydeathsMean[0][2])
    print(bars[0])
    #
    print(mydeathsMean[1])
    print(mydeathsMean[1]/mydeathsMean[1][2])
    print(bars[1])
    # # plt.show()
    # #
    # # myFig.savefig('figures2D/texas/SimpleFigure1_deathsPerMillion_3_26_21_WithBars.pdf')
    # # #
