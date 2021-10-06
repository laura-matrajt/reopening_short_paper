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
N = 7.615 * 10 ** (6)  # Washington state pop

mycolors = sns.color_palette('Paired', 5)

N = 7.615 * 10 ** (6)  # Washington state population

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

# Split the population in 16 age groups:
totalPop16 = N * popUS16fracs
# Split the population in 5 vaccine groups:
totalPop5 = N * fracOfTotalPopulationPerVaccineGroup

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
R0_2 = 3*1.2#(1-0.386)*3 + 0.386*(3*1.5)


frac_rec = 0.175 # #taken from https://covidestim.org/ on may 3rd, 2020
numDosesPerWeek = 51309*7 #taken from https://www.doh.wa.gov/Emergencies/COVID19/DataDashboard on may 4th, 2020


currentInfections = 4366 #taken from https://covidestim.org/ on may 3rd, 2020
currentInfections = popUS16fracs*currentInfections

# VE_P1, VE_P2, VE_S1, VE_S2, VE_I1, VE_I2 = [0.44, 0.66, 0.5, 0.7, 0, 0]
VE_P1, VE_P2, VE_S1, VE_S2, VE_I1, VE_I2 = [0.51, 0.75, 0.51, 0.8, 0, 0]

#add current vaccinations #taken from WA dashboad on 05/04/21
fracVac_16_17_atLeastOneDose = (1/100)*25.9
fracVac_18_34_atLeastOneDose = (1/100)*40.2
fracVac_35_49_atLeastOneDose = (1/100)*51.6
fracVac_50_64_atLeastOneDose = (1/100)*59
fracVac_65_atLeastOneDose = (1/100)*76.2


fracVac_16_17_Full = (1/100)*6.4
fracVac_18_34_Full = (1/100)*22.2
fracVac_35_49_Full = (1/100)*32.3
fracVac_50_64_Full = (1/100)*43.2
fracVac_65_Full = (1/100)*69.2

people_16_17 = totalPop16[3]/4
people_18_34 = totalPop16[3]/4 + np.sum(totalPop16[4:7])
people_35_49 = np.sum(totalPop16[7:10])
people_50_64 = np.sum(totalPop16[10:13])
people_65 = np.sum(totalPop16[13:])

pop_18_34 = np.hstack((totalPop16[3]/4, totalPop16[4:7]))
relativeFrac_18_34 = np.divide(pop_18_34, people_18_34)
relativeFrac_35_49 = np.divide(totalPop16[7:10], people_35_49)
relativeFrac_50_64 = np.divide(totalPop16[10:13], people_50_64 )
relativeFrac_65 = np.divide(totalPop16[13:], people_65)


peopleVacAtLeastOne_16_17 = fracVac_16_17_atLeastOneDose*totalPop16[3]/4
peopleVacAtLeastOne_18_34 = fracVac_18_34_atLeastOneDose*people_18_34*relativeFrac_18_34
peopleVacAtLeastOne_35_49 = fracVac_35_49_atLeastOneDose*people_35_49*relativeFrac_35_49
peopleVacAtLeastOne_50_64 = fracVac_50_64_atLeastOneDose*people_50_64*relativeFrac_50_64
peopleVacAtLeastOne_65 = fracVac_65_atLeastOneDose*people_65*relativeFrac_65

peopleVacAtLeastOne_15_19 = peopleVacAtLeastOne_16_17 + peopleVacAtLeastOne_18_34[0]


peopleFullyVac_16_17 = fracVac_16_17_Full*totalPop16[3]/4
peopleFullyVac_18_34 = fracVac_18_34_Full*people_18_34*relativeFrac_18_34
peopleFullyVac_35_49 = fracVac_35_49_Full*people_35_49*relativeFrac_35_49
peopleFullyVac_50_64 = fracVac_50_64_Full*people_50_64*relativeFrac_50_64
peopleFullyVac_65 = fracVac_65_Full*people_65*relativeFrac_65

peopleFullyVac_15_19 = peopleFullyVac_16_17 + peopleFullyVac_18_34[0]

peopleVacWithAtLeastOne = np.hstack((np.zeros(3), peopleVacAtLeastOne_15_19, peopleVacAtLeastOne_18_34[1:],
                                     peopleVacAtLeastOne_35_49, peopleVacAtLeastOne_50_64, peopleVacAtLeastOne_65))

peopleFullyVac = np.hstack((np.zeros(3), peopleFullyVac_15_19, peopleFullyVac_18_34[1:],
                                     peopleFullyVac_35_49, peopleFullyVac_50_64, peopleFullyVac_65))



fracVacWithAtLeast1D = np.divide(peopleVacWithAtLeastOne, totalPop16)
fracVacWith2D = np.divide(peopleFullyVac, totalPop16)
fracVacWith1D = fracVacWithAtLeast1D - fracVacWith2D

peoVacWith1D = np.multiply(fracVacWith1D, totalPop16)


numDosesNeeded = (np.sum(totalPop16[3:]) - np.sum(peopleFullyVac)) #+ N - np.sum(peoVacWith1D)

peoVacWith1D_vaccineGroup = [np.sum(peoVacWith1D[0:4]), np.sum(peoVacWith1D[4:10]), np.sum(peoVacWith1D[10:13]),
                             np.sum(peoVacWith1D[13:15]), np.sum(peoVacWith1D[15])]

peopleFullyVac_vaccineGroup = [np.sum(peopleFullyVac[0:4]), np.sum(peopleFullyVac[4:10]), np.sum(peopleFullyVac[10:13]),
                             np.sum(peopleFullyVac[13:15]), np.sum(peopleFullyVac[15])]


percentageVacWith1D = np.divide(peoVacWith1D, totalPop16)
percentageFullyVac = np.divide(peopleFullyVac, totalPop16)

#calculate how many more doses we need to put in each vaccine group:
peopleNeedingTwoDosesPerVaccineGroup = np.zeros(5)
peopleNeedingSecondDosePerVaccineGroup = np.zeros(5)
for ivals in range(1,5):
    peopleNeedingSecondDosePerVaccineGroup[ivals] = peoVacWith1D_vaccineGroup[ivals]
    peopleNeedingTwoDosesPerVaccineGroup[ivals] = (totalPop5[ivals] - peopleFullyVac_vaccineGroup[ivals] - peoVacWith1D_vaccineGroup[ivals])

#for the first age group, only the people aged 16-20 are vaccinated:
peopleNeedingSecondDosePerVaccineGroup[0] = peoVacWith1D_vaccineGroup[0]
peopleNeedingTwoDosesPerVaccineGroup[0] = (totalPop16[3] - peopleFullyVac_vaccineGroup[0] - peoVacWith1D_vaccineGroup[0])


numOfDosesStillNeededPerVaccineGroup = peopleNeedingSecondDosePerVaccineGroup + 2*peopleNeedingTwoDosesPerVaccineGroup

#check:
print(peopleNeedingTwoDosesPerVaccineGroup + peopleFullyVac_vaccineGroup + peoVacWith1D_vaccineGroup)
print(totalPop5)
totalVaccineStillNeeded = np.sum(numOfDosesStillNeededPerVaccineGroup)

numVaccinesAvailable = numDosesPerWeek*(28)
newProRataVec = np.array([[0,0,0,0,0], np.divide(numOfDosesStillNeededPerVaccineGroup, np.sum(numOfDosesStillNeededPerVaccineGroup))])


red_contacts = [0.3, 0.5, 0.7, 1]
mylabels = ['45% pre-COVID interactions', '50% pre-COVID interactions', '70% pre-COVID interactions',
            '100% pre-COVID interactions', '100% pre-COVID interactions,\nincreased transmission']




# Model parameters
deaths = np.zeros((1000, 4))
deathsIT = np.zeros((1000, 4))
deathsBas = np.zeros((1000, 4))

# load the 1000 parameters mat:
myparamsName = 'randomParamsForModelConfidenceIntervals.pickle'
myparamsMat = loadResults(myparamsName)

# matrices to store results:
highRiskCurves = [np.zeros((28 * 14, 1000)), np.zeros((28 * 14, 1000)), np.zeros((28 * 14, 1000)),
                  np.zeros((28 * 14, 1000))]

highRiskCurvesIT = [np.zeros((28 * 14, 1000)), np.zeros((28 * 14, 1000)), np.zeros((28 * 14, 1000)),
                  np.zeros((28 * 14, 1000))]

# highRiskCurves = np.zeros((28 * 14, 4))
# highRiskCurvesIT = np.zeros((28 * 14, 4))
# newInfections = np.zeros((28 * 14, 4))
# newInfectionsIT = np.zeros((28 * 14, 4))



# myref = 2
for jvals in range(4):
    print(jvals)
    # SDcoeffs = SDMat[jvals]
    SDcoeffs = [1] + 3 * [red_contacts[jvals]]
    highCurves = highRiskCurves[jvals]
    highCurvesIT = highRiskCurvesIT[jvals]
    for ivals in range(1000):
        durE, durI, durP, durA, redA, redP = myparamsMat[0][ivals, :]
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
        deathsBas[ivals, jvals] = baseline[2]
        deaths[ivals, jvals] = highRisk[2]
        deathsIT[ivals, jvals] = highRiskIT[2]


        highCurves[:, ivals] = highRisk[0][:392]
        highCurvesIT[:, ivals] = highRiskIT[0][:392]
        # for t in range(1, 28 * 14):
        #     newInfections[t, jvals] = highRiskCurves[t, jvals] - highRiskCurves[t-1, jvals]
        #     newInfectionsIT[t, jvals] = highRiskCurvesIT[t, jvals] - highRiskCurvesIT[t - 1, jvals]


myresults = [deathsBas, deaths, highRiskCurves, deathsIT, highRiskCurvesIT]
myfilename = 'reopening_short_paper/WAstate_05_04_21_20percent.pickle'
saveResults(myfilename, myresults)


