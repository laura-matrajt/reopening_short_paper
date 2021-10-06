import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
import seaborn as sns
from matplotlib import pyplot as plt
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

from twoDosesOptimizationFunctionsP import fromNumPeopleVaccinatedToFracVacs, fromFracVacsToNumberPeopleVaccinated2d, defineInitCond2D, pickBestSolForEachObjective2D
from twoDosesOptimizationFunctionsP import createProRataVac2DAdultsOnly, createVectorHighRiskFirst2DosesOnly, run_model3
from saveLoadFunctions import saveResults, loadResults

mycolors1 = sns.color_palette("viridis", 10)[::2]
# mycolors1 = [mycolors[1], mycolors[3], mycolors[5], mycolors[7], mycolors[10]]
mycolors2 =sns.color_palette("magma", 10)[::2]
# mycolors2 = [mycolors[10], mycolors[12], mycolors[14], mycolors[16], mycolors[11]]
# mycolors3 = [mycolors[1], mycolors[3], mycolors[5], mycolors[7], mycolors[11]]
N = 21.48 * 10 ** (6)  # florida state pop

mycolors = sns.color_palette('Paired', 5)


# load contact matrices
myfilename = '../data/consistentMatricesUS_polymodMethod01Jun2020.pickle'
myContactMats = loadResults(myfilename)
mymatAll = myContactMats['all']





# load fractions in each age and vaccine group:
myfilename = '../data/populationUS16ageGroups03Jun2020.pickle'
popUS16 = loadResults(myfilename)
popUS16fracs = popUS16[1]

myfilename = '../data/populationUSGroupsForOptimization03Jun2020.pickle'
groupInfo = loadResults(myfilename)
groupFracs = groupInfo['groupsFracs']
fracOfTotalPopulationPerVaccineGroup = groupInfo['fracOfTotalPopulationPerVaccineGroup']
[relativeFrac75_80, relativeFrac80andAbove] = groupInfo['split75andAbove']


floridaAgeDistribution = (1/100)*np.array([5.3, 5.2, 5.9, 5.7, 5.9, 12.9/2, 12.9/2, 12.2/2, 12.2/2, 12.6/2, 12.6/2, 6.9, 6.6, 11.5/2, 11.5/2, 6.9+2.4]) #taken from https://www.statista.com/statistics/912205/florida-population-share-age-group/
print(np.sum(floridaAgeDistribution))
floridaVaccinationGroups = (1/100)*np.array([5.3+ 5.2 + 5.9 +5.7,
                                             5.9 + 12.9 + 12.2 + 12.6/2,
                                             12.6/2 + 6.9+6.6,
                                             11.5,
                                             6.9 + 2.4 ])

relativeProportions = [[5.7/(5.9 +5.7), 5.9/(5.9 +5.7)], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5],
                       [6.9/(6.9 + 6.6),  6.6/(6.9 + 6.6)], [0.5, 0.5], 1, 1]

# Split the population in 16 age groups:
totalPop16 = N * floridaAgeDistribution
# Split the population in 5 vaccine groups:
totalPop5 = N * floridaVaccinationGroups

numAgeGroups = 16
numVaccineGroups = 5

# load disease severity parameters:
myfilename = '../data/disease_severity_parametersFerguson.pickle'
diseaseParams = loadResults(myfilename)
hosp_rate_16 = diseaseParams['hosp_rate_16']
icu_rate_16 = diseaseParams['icu_rate_16']

# load mortality parameters
myfilename = '../data/salje_IFR_from_hospitalized_cases.pickle'
salje_mortality_rate_16 = loadResults(myfilename)
mortality_rate_16 = salje_mortality_rate_16

# this is just 1 - ICU rate useful to compute it in advance and pass it to the ODE
oneMinusICUrate = np.ones(numAgeGroups) - icu_rate_16
# this is just 1 - Hosp rate useful to compute it in advance and pass it to the ODE
oneMinusHospRate = np.ones(numAgeGroups) - hosp_rate_16

# time horizon for the intervention:
tspan = 26 * 7  # np.linspace(0, 365, 365 * 2)

# hospital stays
gammaH = np.ones(16)
gammaH[0:10] = 1 / 3
gammaH[10:13] = 1 / 4
gammaH[13:] = 1 / 6

gammaICU = np.ones(16)
gammaICU[0:10] = 1 / 10
gammaICU[10:13] = 1 / 14
gammaICU[13:] = 1 / 12

red_sus = np.ones(16)  # assume no pre-COVID interactions in susceptibility
red_sus[0:3] = 0.56  # pre-COVID interactions in susceptibility: taken from Viner 2020
red_sus[13:16] = 2.7  # taken from Bi et al medarxiv 2021

# fraction of symptomatic people
frac_asymptomatic = 0.4 * np.ones(16)  # see notesTwoDoses.md for details
frac_asymptomatic[0:4] = 0.75  # see notes for details
frac_sym = (1 - frac_asymptomatic) * np.ones(16)  # fraction of infected that are symptomatic
oneMinusSymRate = np.ones(16) - frac_sym

#hospitalizations
sigma_base = 1 / 3.8
sigma = sigma_base * np.ones(16)
# Disease severity
R0 = 3
R0_2 = 3*1.2#(1-0.64)*3 + 0.64*(3*1.5) #taken from https://outbreak.info/location-reports?loc=USA_US-TX on 05/04/21

frac_rec = 0.3669#0.4679 #email from Ian # 0.174 #taken from https://covidestim.org/
numDosesPerWeek = 54521

# number of current infections:
# prevPer100K = 61.45 #taken from https://covidestim.org/
# currentInfections = (prevPer100K*N) /1e5
currentInfections = 10037
currentInfections = popUS16fracs*currentInfections

# VE_P1, VE_P2, VE_S1, VE_S2, VE_I1, VE_I2 = [0.44, 0.66, 0.5, 0.7, 0, 0]
VE_P1, VE_P2, VE_S1, VE_S2, VE_I1, VE_I2 = [0.51, 0.75, 0.51, 0.8, 0, 0]

#add current vaccinations #taken from http://ww11.doh.state.fl.us/comm/_partners/covid19_report_archive/vaccine/vaccine_report_latest.pdf
#consulted on may 7th vaccination data through may 6th
age16_24 =  [280225, 269426, 549651]
age25_34 =  [329138, 449138, 778276]
age35_44 =  [359023, 635731, 994754]
age45_54 =  [376770, 913909, 1290679]
age55_64 =  [412292, 1398776, 1811068]
age65_74 =  [338013, 1769881, 2107894]
age75_84 = [ 193172, 1010013, 1203185]
age85_plus =  [79673, 333991, 413664]

allVacs = np.array([[280225, 269426, 549651],
           [329138, 449138, 778276],
            [359023, 635731, 994754],
            [376770, 913909, 1290679],
            [412292, 1398776, 1811068],
            [338013, 1769881, 2107894],
            [ 193172, 1010013, 1203185],
            [79673, 333991, 413664]])


relativeProportions = [[5.7/(5.9 +5.7), 5.9/(5.9 +5.7)], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5],
                       [6.9/(6.9 + 6.6),  6.6/(6.9 + 6.6)], [0.5, 0.5], 1, 1]

peopleVac15_19 = (5.7/(5.9 +5.7))*allVacs[0]
peopleVac20_25 = 5.9/(5.9 +5.7)*allVacs[0]
peoVacWith1D = np.zeros(16)
peoFullyVac = np.zeros(16)
for ivals in range(6):
    kval = 2*ivals
    mym = np.array(relativeProportions[ivals])
    print(mym)
    temp = allVacs[ivals,0]*mym
    temp2 = allVacs[ivals,1]*mym
    peoVacWith1D[kval] = temp[0]
    peoVacWith1D[kval+3 + 1] = temp[1]
    peoFullyVac[kval+3] = temp2[0]
    peoFullyVac[kval+3 + 1] = temp2[1]
    # peoFullyVac[ivals],peoFullyVac[ ivals+1] = (temp2)


peoVacWith1D[15] = np.sum(allVacs[6:8, 0])
peoFullyVac[15] = np.sum(allVacs[6:8, 1])
print(peoVacWith1D)
print(peoFullyVac)


percentageVacWith1D = np.divide(peoVacWith1D, totalPop16)
percentageFullyVac = np.divide(peoFullyVac, totalPop16)



numDosesNeeded =  2*np.sum(totalPop16[3:]) - np.sum(peoFullyVac) #+ N - np.sum(peoVacWith1D)

peoVacWith1D_vaccineGroup = [np.sum(peoVacWith1D[0:4]), np.sum(peoVacWith1D[4:10]), np.sum(peoVacWith1D[10:13]),
                             np.sum(peoVacWith1D[13:15]), np.sum(peoVacWith1D[15])]

peoVacWith2D_vaccineGroup = [np.sum(peoFullyVac[0:4]), np.sum(peoFullyVac[4:10]), np.sum(peoFullyVac[10:13]),
                             np.sum(peoFullyVac[13:15]), np.sum(peoFullyVac[15])]


percentageVacWith1D_vaccineGroup = np.divide(peoVacWith1D_vaccineGroup, totalPop5)
percentageFullyVac_vaccineGroup = np.divide(peoVacWith2D_vaccineGroup, totalPop5)

# fracCov = 0.5#0.1 * VC  # vaccination coverage of the total population
# print(fracCov)

#round(fracCov * N)  # number of total vaccines available assuming that coverage

#calculate how many more doses we need to put in each vaccine group:
peopleNeedingTwoDosesPerVaccineGroup = np.zeros(5)
peopleNeedingSecondDosePerVaccineGroup = np.zeros(5)
for ivals in range(1,5):
    peopleNeedingSecondDosePerVaccineGroup[ivals] = peoVacWith1D_vaccineGroup[ivals]
    peopleNeedingTwoDosesPerVaccineGroup[ivals] = (totalPop5[ivals] - peoVacWith2D_vaccineGroup[ivals] - peoVacWith1D_vaccineGroup[ivals])

#for the first age group, only the people aged 16-20 are vaccinated:
peopleNeedingSecondDosePerVaccineGroup[0] = peoVacWith1D_vaccineGroup[0]
peopleNeedingTwoDosesPerVaccineGroup[0] = (totalPop16[3] - peoVacWith2D_vaccineGroup[0] - peoVacWith1D_vaccineGroup[0])


numOfDosesStillNeededPerVaccineGroup = peopleNeedingSecondDosePerVaccineGroup + 2*peopleNeedingTwoDosesPerVaccineGroup

#check:
print(peopleNeedingTwoDosesPerVaccineGroup + peoVacWith2D_vaccineGroup + peoVacWith1D_vaccineGroup)
print(totalPop5)
totalVaccineStillNeeded = np.sum(numOfDosesStillNeededPerVaccineGroup)

numVaccinesAvailable = numDosesPerWeek*(28)
newProRataVec = np.array([[0,0,0,0,0], np.divide(numOfDosesStillNeededPerVaccineGroup, np.sum(numOfDosesStillNeededPerVaccineGroup))])




red_contacts = [0.3, 0.5, 0.7, 1]



# Model parameters
deaths = np.zeros((1000, 4))
deathsIT = np.zeros((1000, 4))
deathsBas = np.zeros((1000, 4))

# load the 1000 parameters mat:
myparamsName = 'randomParamsForModelConfidenceIntervalsTexas09Mar2021.pickle'
myparamsMat = loadResults(myparamsName)

# matrices to store results:
highRiskCurves = [np.zeros((28 * 14, 1000)), np.zeros((28 * 14, 1000)), np.zeros((28 * 14, 1000)),
                  np.zeros((28 * 14, 1000))]

highRiskCurvesIT = [np.zeros((28 * 14, 1000)), np.zeros((28 * 14, 1000)), np.zeros((28 * 14, 1000)),
                  np.zeros((28 * 14, 1000))]

deathsMean = np.zeros(4)
deathsITMean = np.zeros(4)
myindex = 10
highRiskCurvesMean = np.zeros((28 * 14, 4))
highRiskCurvesITMean = np.zeros((28 * 14, 4))
for jvals in range(4):
    print(jvals)
    # SDcoeffs = SDMat[jvals]
    SDcoeffs = [1] + 3 * [red_contacts[jvals]]
    highCurves = highRiskCurves[jvals]
    highCurvesIT = highRiskCurvesIT[jvals]
    for ivals in range(1000):

        durE, durI, durP, durA, redA, redP = myparamsMat[0][ivals, :]
        # print(durE, durI, durP, durA, redA, redP)
        # fraction of symptomatic people
        gammaA = 1 / (durI + durP)  # durA  # recovery rate for asymptomatic
        gammaI = 1 / durI  # recovery rate for symptomatic infections (not hospitalized)
        gammaP = 1 / durP  # transition rate fromm pre-symptomatic to symptomatic
        gammaE = 1 / durE  # transition rate from exposed to infectious

        redH = 0.  # assume no transmission for hospitalized patients

        # move people from susceptible to vaccinated
        y0 = defineInitCond2D(currentInfections, frac_rec, frac_sym, hosp_rate_16, icu_rate_16, numAgeGroups,
                              oneMinusHospRate, oneMinusICUrate,
                              totalPop16)
        y0 = y0.reshape(((34, numAgeGroups)))
        initCond = np.copy(y0)
        initCond[0, :] = initCond[0, :] - initCond[0, :] * percentageVacWith1D - initCond[0, :] * percentageFullyVac
        initCond[22, :] = initCond[22, :] + initCond[0, :] * percentageFullyVac
        initCond[11, :] = initCond[11, :] + initCond[0, :] * percentageVacWith1D
        # print(initCond[0, :])
        # print(np.where(initCond < 0))
        initCond = initCond.reshape(34 * numAgeGroups)
        # print(np.sum(initCond[:33 * numAgeGroups]))
        # print(np.where(initCond<0))

        # With this values for mymatSD and for 10% of the pop recovered, we get an effective R of 1.2
        allOtherParams = [currentInfections, frac_rec, frac_sym, gammaA, gammaE, gammaH, gammaI, gammaICU, gammaP,
                          groupFracs, hosp_rate_16, icu_rate_16, mortality_rate_16,
                          myContactMats, numAgeGroups, oneMinusHospRate, oneMinusICUrate, oneMinusSymRate,
                          redA, redH, redP, red_sus, R0, sigma, totalPop16, totalPop5]

        allOtherParams2 = [currentInfections, frac_rec, frac_sym, gammaA, gammaE, gammaH, gammaI, gammaICU, gammaP,
                           groupFracs, hosp_rate_16, icu_rate_16, mortality_rate_16,
                           myContactMats, numAgeGroups, oneMinusHospRate, oneMinusICUrate, oneMinusSymRate,
                           redA, redH, redP, red_sus, R0_2, sigma, totalPop16, totalPop5]

        baseline = run_model3(allOtherParams, np.ones((2, 5)),
                              numDosesPerWeek, numVaccinesAvailable, SDcoeffs, 0, 0, 0, 0, 0, 0, initCond)
        highRisk = run_model3(allOtherParams, newProRataVec,
                              numDosesPerWeek, numVaccinesAvailable, SDcoeffs, VE_I1, VE_I2, VE_P1, VE_P2, VE_S1, VE_S2,
                              initCond)
        highRiskIT = run_model3(allOtherParams2, newProRataVec,
                              numDosesPerWeek, numVaccinesAvailable, SDcoeffs, VE_I1, VE_I2, VE_P1, VE_P2, VE_S1, VE_S2,
                              initCond)
        deaths[ivals, jvals] = highRisk[2]
        deathsIT[ivals, jvals] = highRiskIT[2]

        highCurves[:, ivals] = highRisk[0][:392]
        highCurvesIT[:, ivals] = highRiskIT[0][:392]


myresults = [deathsBas, deaths, highRiskCurves, deathsIT, highRiskCurvesIT]
myfilename = 'reopening_short_paper/FLstate_05_04_21_20percent.pickle'
saveResults(myfilename, myresults)