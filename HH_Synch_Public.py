#======================================================================
# HODGKIN HUXLEY NEURON MODEL Originally Modified Feb 2021 by @conBraun
#======================================================================

# HodgkinHuxley and Gate class structures are inspired by script authored by Peter Rupprecht, modified by Connor Braun for research/personal interest purposes
#----------- REFERENCE------------------------------------------------------------
# P. Rupprecht, "Hodgkin-Huxley-CC-VC", Last commit Dec 9 2020
# Copyright (c) 2020 Peter Rupprecht
# Copyright (c) 2019 Scott W. Harden for original code simulating the HH model
# [Online] Available: https://github.com/PTRRupprecht/Hodgkin-Huxley-CC-VC/blob/main/Hodgkin_Huxley_in_current_and_voltage_clamp.ipynb
#-----------ORIGINAL PROGRAM LICENCE----------------------------------------------
# The software from which this script is modified is licensed under an MIT license.
# For details of the copyright please see: https://github.com/PTRRupprecht/Hodgkin-Huxley-CC-VC/blob/main/LICENSE#L20
#---------------------------------------------------------------------------------
import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from notion.client import NotionClient
from datetime import datetime
from scipy.stats import mode
from scipy.signal import find_peaks, hilbert
from figureFunctions import hh_figtype_7

class HodgkinHuxleyNetwork:

    def __init__(self, parameterDictionary):
        class Gate:
            alpha, beta, state = 0, 0, 0 # Initialize gate state and rate constants

            def update(self, dt):
                # Pass in timestep in ms
                alphaState = self.alpha*(1 - self.state) # Forward rate constant times unopen population
                betaState = self.beta*self.state # Backward rate constant times open population
                self.state += dt*(alphaState - betaState) # dP/dt = a(1-P) - B(P) => dP = dt(a(1-P) - B(P)); add dP to P 
            def setEquilibirumState(self):
                self.state = self.alpha/(self.alpha + self.beta) # [P = Peq - (Peq - P0)*exp(-t/Tau)] => [lim(t->inf)P = Peq = a/(a+B)] 

        # Set average channel conductances per unit area (mS/cm^2)
        self.parameters = parameterDictionary
        self.gK = parameterDictionary['gK']
        self.gNa = parameterDictionary['gNa']
        self.gL = parameterDictionary['gL']
        self.Cm = parameterDictionary['Cm'] # Membrane capacitances (uF/cm^2)
        # Set nernst potentials as passed (mV)
        self.EK = parameterDictionary['EK']
        self.ENa = parameterDictionary['ENa']
        self.EL = parameterDictionary['EL']
        self.Vm = parameterDictionary['Vm'] # Initial membrane potentials (mV)
        self.adjacencyMatrix = parameterDictionary['K'] # Unitless, element ij indicates a unidirectional synapse from neuron j to neuron i
        self.AP_times = parameterDictionary['AP_init']
        self.timeConstants = parameterDictionary['Tau']

        self.IK = [0 for i in range(self.adjacencyMatrix.shape[0])]
        self.INa = [0 for i in range(self.adjacencyMatrix.shape[0])]
        self.IL = [0 for i in range(self.adjacencyMatrix.shape[0])]
        self.Isyn = [0 for i in range(self.adjacencyMatrix.shape[0])]

        self.nGates = []
        self.mGates = []
        self.hGates = []
        for neuronIndex in range(self.adjacencyMatrix.shape[0]):
            self.nGates.append(Gate())
            self.mGates.append(Gate())
            self.hGates.append(Gate())
    def _updateGateRateConstants(self, neuronIndex):
        # Best fit values from literature for convention of -60mV resting potential; HH 1952 took Vm = 0
        self.nGates[neuronIndex].alpha = (0.01*(-50 - self.Vm[neuronIndex]))/(np.exp((-50 - self.Vm[neuronIndex])/10) - 1)
        self.nGates[neuronIndex].beta = 0.125*np.exp((-self.Vm[neuronIndex] - 60)/80)

        self.mGates[neuronIndex].alpha = 0.1*(-35 - self.Vm[neuronIndex])/(np.exp((-35 - self.Vm[neuronIndex])/10) - 1)
        self.mGates[neuronIndex].beta = 4*np.exp((-self.Vm[neuronIndex] - 60)/18)

        self.hGates[neuronIndex].alpha = 0.07*np.exp((-60 - self.Vm[neuronIndex])/20)
        self.hGates[neuronIndex].beta = 1/(np.exp((-30 - self.Vm[neuronIndex])/10) + 1)
    def _updateVm(self, forcing, dt, neuronIndex, timeSeriesIndex):
        self.IK[neuronIndex] = np.power(self.nGates[neuronIndex].state, 4)*self.gK[neuronIndex]*(self.Vm[neuronIndex] - self.EK[neuronIndex]) # 4th order n-gating * average potassium conductance (mS/cm^2) * potassium driving force
        self.INa[neuronIndex] = np.power(self.mGates[neuronIndex].state, 3)*self.hGates[neuronIndex].state*self.gNa[neuronIndex]*(self.Vm[neuronIndex] - self.ENa[neuronIndex]) # 3rd order m-gating * 1st order trenchant h-gate * average sodium conductance (mS/cm^2) * sodium driving force
        self.IL[neuronIndex] = self.gL[neuronIndex]*(self.Vm[neuronIndex] - self.EL[neuronIndex]) # Linear leak conductance combining sodium and potassium variability; how is the reversal potential for this computed?

        # Superimpose all synaptic currents
        self.Isyn[neuronIndex] = 0
        for i, j in enumerate(self.adjacencyMatrix[neuronIndex], start=0):
            self.Isyn[neuronIndex] += j*(((timeSeriesIndex - self.AP_times[i])*dt)/self.timeConstants[i])*np.exp(-(((timeSeriesIndex - self.AP_times[i])*dt)/self.timeConstants[i]))
        
        currentSuperposition = forcing - self.IK[neuronIndex] - self.INa[neuronIndex] - self.IL[neuronIndex] + self.Isyn[neuronIndex] # Current obeys principle of superposition, sum all currents for total
        
        self.Vm[neuronIndex] += dt*(currentSuperposition/self.Cm[neuronIndex]) 
    def _updateGateStates(self, dt, neuronIndex):
        # Step the gating variable ODEs
        self.nGates[neuronIndex].update(dt)
        self.mGates[neuronIndex].update(dt)
        self.hGates[neuronIndex].update(dt)
    def iterate(self, forcing, dt, neuronIndex, timeSeriesIndex):
        # Step rate constants, membrane potential and gating variable ODEs; how should these be ordered? This may be a limitation of discrete time solutions. 
        self._updateGateRateConstants(neuronIndex)
        self._updateVm(forcing, dt, neuronIndex, timeSeriesIndex)
        self._updateGateStates(dt, neuronIndex)
class NetworkSimulation:
    def __init__(self, model):
        self.model = model
        self.numberOfNeurons = model.adjacencyMatrix.shape[0]
        self.actionPotentialThreshold = model.parameters['AP_thresh']
        self.vectorInitialize(0, 0)
        pass
    def vectorInitialize(self, num_timesteps, dt):
        for neuron in range(self.numberOfNeurons):
            times = np.arange(num_timesteps) * dt
            Vm_t = np.empty(num_timesteps)
            IK = np.empty(num_timesteps)
            INa = np.empty(num_timesteps)
            IL = np.empty(num_timesteps)
            Isyn = np.empty(num_timesteps)
            nState = np.empty(num_timesteps)
            mState = np.empty(num_timesteps)
            hState = np.empty(num_timesteps)

            if neuron == 0:
                self.times = np.reshape(times, (1, times.shape[0]))
                self.Vm_t = np.reshape(Vm_t, (1, Vm_t.shape[0]))
                self.IK = np.reshape(IK, (1, IK.shape[0]))
                self.INa = np.reshape(INa, (1, INa.shape[0]))
                self.IL = np.reshape(IL, (1, IL.shape[0]))
                self.Isyn = np.reshape(Isyn, (1, Isyn.shape[0]))
                self.nState = np.reshape(nState, (1, nState.shape[0]))
                self.mState = np.reshape(mState, (1, mState.shape[0]))
                self.hState = np.reshape(hState, (1, hState.shape[0]))
            else:
                self.times = np.concatenate((self.times, np.reshape(times, (1, times.shape[0]))), axis=0)
                self.Vm_t = np.concatenate((self.Vm_t, np.reshape(Vm_t, (1, Vm_t.shape[0]))), axis=0)
                self.IK = np.concatenate((self.IK, np.reshape(IK, (1, IK.shape[0]))), axis=0)
                self.INa = np.concatenate((self.INa, np.reshape(INa, (1, INa.shape[0]))), axis=0)
                self.IL = np.concatenate((self.IL, np.reshape(IL, (1, IL.shape[0]))), axis=0)
                self.Isyn = np.concatenate((self.Isyn, np.reshape(Isyn, (1, Isyn.shape[0]))), axis=0)
                self.nState = np.concatenate((self.nState, np.reshape(nState, (1, nState.shape[0]))), axis=0)
                self.mState = np.concatenate((self.mState, np.reshape(mState, (1, mState.shape[0]))), axis=0)
                self.hState = np.concatenate((self.hState, np.reshape(hState, (1, hState.shape[0]))), axis=0)
    def actionPotentialDetection(self, timeSeriesIndex, dt):
        for neuron in range(self.numberOfNeurons):
            Vm_recent = self.Vm_t[neuron, timeSeriesIndex - 3:timeSeriesIndex]
            if Vm_recent[-1] > self.actionPotentialThreshold:
                finiteDifferences = np.diff(Vm_recent)
            else:
                continue
            if finiteDifferences[0] >= 0 and finiteDifferences[1] < 0:
                self.model.AP_times[neuron] = (timeSeriesIndex - 2)
                #print("Action potential in neuron {} at {} ms".format(neuron + 1, timeSeriesIndex*dt))
                #print("Most recent AP times: {}".format(self.model.AP_times))
    def runSim(self, commandFunction, dt):
        self.vectorInitialize(commandFunction.shape[1], dt)
        print("Initialization complete...")
        print("Simulating {} ms\nParameters: \n{}".format(commandFunction.shape[1]*dt, self.model.parameters))
        for k, i in enumerate(range(commandFunction.shape[1]), start=0):
            if k > 10:
                self.actionPotentialDetection(k, dt)
            for j in range(self.numberOfNeurons):
                self.model.iterate(commandFunction[j, i], dt, j, k)
                self.Vm_t[j, i] = self.model.Vm[j]
                self.IK[j, i] = self.model.IK[j]
                self.INa[j, i] = self.model.INa[j]
                self.IL[j, i] = self.model.IL[j]
                self.Isyn[j, i] = self.model.Isyn[j]
                self.nState[j, i] = self.model.nGates[j].state
                self.mState[j, i] = self.model.mGates[j].state
                self.hState[j, i] = self.model.hGates[j].state
        print("HALT")

def packageParameters(gK, gNa, gL, Cm, EK, ENa, EL, Vm_0, adjacencyMatrix, AP_times, timeConstants, AP_threshold, T):
    """
    Takes all HH class parameters, packages and returns a 
    dictionary class object; the keys are hardcoded in the 
    Hodgkin Huxley class definition 
    """
    parameterDictionary = {
        "gK": gK,
        "gNa": gNa,
        "gL": gL,
        "Cm": Cm,
        "EK": EK,
        "ENa": ENa,
        "EL": EL,
        "Vm": Vm_0,
        "K": adjacencyMatrix,
        "AP_init": AP_times,
        "Tau": timeConstants,
        "AP_thresh": AP_threshold,
        "T": T
    }
    return parameterDictionary
def nernstPotential(concentration_In, concentration_Out, temperature, valence, ionName):
    """
    Takes extra and intracellular ion concentrations (mM), temperature (Kelvin)
    and ion valence as an integer or float WITH the associated polarity, 
    computes reversal potential and returns it. The ion name argument is for 
    a print statement.
    """

    faradayConstant = 96485.33212 # Coulombs/mol
    gasConstant = 8.3145 # J/(mol*K)

    if valence > 0: # For cations the logarithmic argument is uninverted
        nernst = (gasConstant*temperature*np.log(concentration_Out/concentration_In))/(valence*faradayConstant)
    elif valence < 0: # For anions the logarithmic argument is inverted, then the valence made positive
        nernst = (gasConstant*temperature*np.log(concentration_In/concentration_Out))/(np.abs(valence)*faradayConstant)
    else:
        raise Exception("Cannot compute reversal potential for chargeless species") # Uncharged species are not subject to coulomb forces
    nernst *= 1000 #(V => mV) # Dimensional conversion
    print("Nernst potential for {}: {} mV".format(ionName, nernst)) # Print the reversal potential to the console
    return nernst
def stimulusFunction(segmentAmplitudes, segmentDurations, dt):
    """
    Takes for each segment, creates a constant function of specified
    duration and amplitude, concatenates each, then returns the array. 
    Amplitudes are in units uA/cm^2, durations and dt are in ms.
    """
    for i, iteration in enumerate(segmentAmplitudes, start=0):
        for j, amplitude in enumerate(segmentAmplitudes[i], start=0): # for each amplitude in the input iterable
            segmentLength = int(segmentDurations[i][j]/dt) # infer number of segment elements using the specified timestep
            segment = amplitude*np.ones(segmentLength) # scale an array of ones for correct amplitude
            if j == 0: # If this is the first segment...
                stimulus = segment # initialize the stimulus function as the current segment
            else:
                stimulus = np.concatenate((stimulus, segment)) # otherwise concatenate the segment to the preexisting function
        if i == 0: # If this is the first segment...
            stimulationSet = np.reshape(stimulus, (1, stimulus.shape[0])) # initialize the stimulus function as the current segment
        else:
            stimulus = np.reshape(stimulus, (1, stimulus.shape[0]))
            stimulationSet = np.concatenate((stimulationSet, stimulus), axis=0) # otherwise concatenate the segment to the preexisting function

    print("Stimulus protocol shape: {}".format(stimulationSet.shape))

    return stimulationSet
def plotSolutions(SimulationObject, commandFunction, workingDirectory, num_neurons, darkBackground=False, notionOutput=False, legends=True, suppresspopup=False):
    """
    Takes a completed hodgkin huxley simulation object and a command function,
    creates a plot of the solutions specified in *kwargs: baseplot includes membrane
    voltage and injected current, ion currents include Na+, K+ and leak, gate states
    are the proportion of a population of gates that are active -- i.e. permitting
    ion conductance. darkBackground is stylistic and makes plots suitable for Manim,
    Notion and slide decks with dark backgrounds.
    """
    plt.rcParams.update({'font.size': 10})

    if not darkBackground:
        timeAxis = SimulationObject.times[0]

        number_of_plots = 2
        colorSequence = ['deeppink', 'darkslateblue', 'crimson', 'dodgerblue', 'darkturqoise']

        plt.close()
        fig, axes = plt.subplots(nrows=number_of_plots, ncols=1, sharex=True)
        fig.patch.set_alpha(0.0)
        for ax in axes:
            ax.patch.set_alpha(0.0)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

        plotCount = 0

        for neuron in range(num_neurons):
            axes[plotCount].plot(timeAxis, SimulationObject.Vm_t[neuron], linewidth=0.8, color=colorSequence[neuron], label='Vn{}'.format(neuron + 1))
            axes[plotCount].plot(timeAxis, commandFunction[neuron, :], linewidth=0.8, color=colorSequence[neuron], linestyle='dashed', label='Forcing n{} ($\mu$A/$cm^{2}$)')
        #axes[plotCount].plot(timeAxis, SimulationObject.Vm_t[0], linewidth=0.8, color='deeppink', label='$V_{n1}$')
        #axes[plotCount].plot(timeAxis, SimulationObject.Vm_t[1], linewidth=0.8, color='darkslateblue', label='$V_{n2}$')
        #axes[plotCount].plot(timeAxis, commandFunction[0, :], linewidth=0.8, color='deeppink', linestyle='dashed', label='$Forcing_{n1}$ ($\mu$A/$cm^{2}$)')
        #axes[plotCount].plot(timeAxis, commandFunction[1, :], linewidth=0.8, color='darkslateblue', linestyle='dashed', label='$Forcing_{n2}$ ($\mu$A/$cm^{2}$)')
        axes[plotCount].set_ylabel("Vm ($mV$)")
        if legends:
            axes[plotCount].legend(edgecolor='w', framealpha=0.0, bbox_to_anchor=(1.0, 1), loc='upper left')
        plotCount += 1
        if plotCount == number_of_plots:
            axes[plotCount - 1].set_xlabel("Time (ms)")
        for neuron in range(num_neurons):
            axes[plotCount].plot(timeAxis, SimulationObject.Isyn[neuron], linewidth=0.8, color=colorSequence[neuron], label='PSC')
        axes[plotCount].set_ylabel("PSC ($\mu A/cm^{2}$)")
        plotCount += 1
        if plotCount == number_of_plots:
            axes[plotCount - 1].set_xlabel('Time (ms)')

        plt.savefig('{}\\hodgkinHuxleyModel_Research\\modelFigures\\HH_Synch_Solution.png'.format(workingDirectory), bbox_inches='tight')
    else:
        timeAxis = SimulationObject.times[neuronIndex]

        number_of_plots = 0
        if baseplot:
            number_of_plots += 1
        if ionCurrents_t:
            number_of_plots += 1
        if gateStates_t:
            number_of_plots += 1

        plt.close()
        fig, axes = plt.subplots(nrows=number_of_plots, ncols=1, sharex=True)
        fig.patch.set_alpha(0.0)
        for ax in axes:
            ax.patch.set_alpha(0.0)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_color('w')
            ax.spines['bottom'].set_color('w')
            ax.tick_params(axis='x', colors="w")
            ax.tick_params(axis='y', colors="w")

        plotCount = 0
        if baseplot:
            axes[plotCount].plot(timeAxis, SimulationObject.Vm_t[neuronIndex], linewidth=0.8, color='w', label='Vm')
            axes[plotCount].plot(timeAxis, commandFunction, linewidth=0.8, color='w', label='Forcing ($\mu$A/$cm^{2}$)', linestyle='dotted')
            axes[plotCount].set_ylabel("$V_{m}$ ($mV$)", color='w')
            if legends:
                legend = axes[plotCount].legend(edgecolor='w', framealpha=0.0, bbox_to_anchor=(1.0, 1), loc='upper left')
                plt.setp(legend.get_texts(), color='w')
            plotCount += 1
            if plotCount == number_of_plots:
                axes[plotCount - 1].set_xlabel("Time (ms)", color='w')
        if ionCurrents_t:
            axes[plotCount].plot(timeAxis, SimulationObject.IK[neuronIndex], linewidth=0.8, color='w', label="$K^{+}$Current")
            axes[plotCount].plot(timeAxis, SimulationObject.INa[neuronIndex], linewidth=0.8, color='w', label="$Na^{+}$Current", linestyle='dashdot')
            axes[plotCount].plot(timeAxis, SimulationObject.IL[neuronIndex], linewidth=0.8, color='w', label="Leak Current", linestyle='dotted')
            #axes[plotCount].plot(timeAxis, commandFunction, linewidth=0.8, color='w', label='Forcing', linestyle='dashed')
            axes[plotCount].set_ylabel("Current ($\mu A/cm^{2}$)", color='w')
            if legends:
                legend = axes[plotCount].legend(edgecolor='w', framealpha=0.0, bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.setp(legend.get_texts(), color='w')
            plotCount += 1
            if plotCount == number_of_plots:
                axes[plotCount - 1].set_xlabel('Time (ms)', color='w')
        if gateStates_t:
            axes[plotCount].plot(timeAxis, SimulationObject.nState[neuronIndex], linewidth=0.8, color='w', label="n Gate")
            axes[plotCount].plot(timeAxis, SimulationObject.mState[neuronIndex], linewidth=0.8, color='w', label="m Gate", linestyle="dotted")
            axes[plotCount].plot(timeAxis, SimulationObject.hState[neuronIndex], linewidth=0.8, color='w', label="h Gate", linestyle='dashed')
            axes[plotCount].set_ylabel("Active fraction", color='w')
            if legends:
                legend = axes[plotCount].legend(edgecolor='w', framealpha=0.0, bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.setp(legend.get_texts(), color='w')
            plotCount += 1 
            if plotCount == number_of_plots:
                axes[plotCount - 1].set_xlabel('Time (ms)', color='w')

        plt.savefig('{}\\hodgkinHuxleyModel_Research\\modelFigures\\HHSolution_{}.png'.format(workingDirectory, neuronIndex), bbox_inches='tight')

    if not suppresspopup:
        figureManager = plt.get_current_fig_manager()
        figureManager.window.state('zoomed')
        plt.show()
def notionInitialize(notion_token_v2, pageURL="None", collectionURL="None"):
    DEPTHCOUNTER = 0
    client = NotionClient(token_v2=notion_token_v2) # open the client using a token (find using Chrome developer console: Application --> Cookies)
    if pageURL:
        page = client.get_block(pageURL)
        DEPTHCOUNTER += 1
    else:
        print("notionInitialize() RETURNING: NOTION CLIENT")
        return client
    if collectionURL:
        cv = client.get_collection_view(collectionURL)
        DEPTHCOUNTER += 1
    else:
        print("notionInitialize() RETURNING: NOTION PAGE")
        return page
    print("notionInitialize() RETURNING: COLLECTION")
    return cv
def notionLog(notion_cv, iterationNumber, parameterDictionary, concentrationDictionary):
    longestConcentrationIterable = selectIterable(concentrationDictionary['[Na+]_o'], concentrationDictionary['[Na+]_i'], concentrationDictionary['[K+]_o'], concentrationDictionary['[K+]_i'])
    keyList = list(concentrationDictionary.keys())
    valueList = list(concentrationDictionary.values())
    try:
        longestConcentrationIterable_Key = keyList[valueList.index(longestConcentrationIterable)]
    except ValueError:
        longestConcentrationIterable_Key = False

    newEntry = notion_cv.collection.add_row()
    newEntry.Date = datetime.now().strftime("%d/%m/%Y %H:%M:%S") + ": {}".format(iterationNumber)
    newEntry.Conductances = "gK: {} mS/cm^2\ngNa: {} mS/cm^2\ngL: {} mS/cm^2".format(parameterDictionary['gK'], parameterDictionary['gNa'], parameterDictionary['gL'])
    newEntry.Nernst_Potentials = "EK: {} mV\nENa: {} mV\nEL: {} mV".format(round(parameterDictionary['EK'], 2), round(parameterDictionary['ENa'], 2), round(parameterDictionary['EL'], 2))
    try:
        newEntry.Concentrations = "[Na+]out: {} mM\n[Na+]in: {} mM\n[K+]out: {} mM\n[K+]in: {} mM".format(concentrationDictionary['[Na+]_o'][iterationNumber - 1], concentrationDictionary['[Na+]_i'][iterationNumber - 1], concentrationDictionary['[K+]_o'][iterationNumber - 1], concentrationDictionary['[K+]_i'][iterationNumber - 1])
    except IndexError:
        if longestConcentrationIterable_Key:
            if longestConcentrationIterable_Key == "[Na+]_o":
                newEntry.Concentrations = "[Na+]out: {} mM\n[Na+]in: {} mM\n[K+]out: {} mM\n[K+]in: {} mM".format(concentrationDictionary['[Na+]_o'][iterationNumber - 1], concentrationDictionary['[Na+]_i'][-1], concentrationDictionary['[K+]_o'][-1], concentrationDictionary['[K+]_i'][-1])
            elif longestConcentrationIterable_Key == "[Na+]_i":
                newEntry.Concentrations = "[Na+]out: {} mM\n[Na+]in: {} mM\n[K+]out: {} mM\n[K+]in: {} mM".format(concentrationDictionary['[Na+]_o'][-1], concentrationDictionary['[Na+]_i'][iterationNumber - 1], concentrationDictionary['[K+]_o'][-1], concentrationDictionary['[K+]_i'][-1])
            elif longestConcentrationIterable_Key == "[K+]_o":
                newEntry.Concentrations = "[Na+]out: {} mM\n[Na+]in: {} mM\n[K+]out: {} mM\n[K+]in: {} mM".format(concentrationDictionary['[Na+]_o'][-1], concentrationDictionary['[Na+]_i'][-1], concentrationDictionary['[K+]_o'][iterationNumber - 1], concentrationDictionary['[K+]_i'][-1])
            elif longestConcentrationIterable_Key == "[K+]_i":
                newEntry.Concentrations = "[Na+]out: {} mM\n[Na+]in: {} mM\n[K+]out: {} mM\n[K+]in: {} mM".format(concentrationDictionary['[Na+]_o'][-1], concentrationDictionary['[Na+]_i'][-1], concentrationDictionary['[K+]_o'][-1], concentrationDictionary['[K+]_i'][iterationNumber - 1])
        else:
            newEntry.Concentrations = "[Na+]out: {} mM\n[Na+]in: {} mM\n[K+]out: {} mM\n[K+]in: {} mM".format(concentrationDictionary['[Na+]_o'][-1], concentrationDictionary['[Na+]_i'][-1], concentrationDictionary['[K+]_o'][-1], concentrationDictionary['[K+]_i'][-1])

    newEntry.Miscellaneous = "Initial Vm: {} mV\nTemperature: {} K".format(parameterDictionary['Vm'], parameterDictionary['T'])
def selectIterable(*args):
    equalCount = 0
    for index, iterable in enumerate(args, start=0):
        if index == 0:
            longest = index
            pass
        else:
            if len(iterable) > len(args[longest]):
                longest = index
            elif len(iterable) == len(args[longest]):
                equalCount += 1
    if equalCount == len(args) - 1:
        return [random.uniform(0, 1) for i in args[longest]]
    else:
        return args[longest]
def computeFrequency(SimulationObject, neuronIndex):
    membranePotential = SimulationObject.Vm_t[neuronIndex, :]
    actionPotential_Indices = find_peaks(membranePotential, height=0.00)
    timeSeriesLength_seconds = round(SimulationObject.times[neuronIndex, -1])/1000
    oscillationFrequency = len(actionPotential_Indices[0])/timeSeriesLength_seconds

    return oscillationFrequency
def phaseSpaceAnalysis(SimulationObject, num_neurons, neurons_to_compare, plotAnalyticSignals=True, computePLV=True):

    assert len(neurons_to_compare) == 2

    signalDictionary = {}
    phaseListDictionary = {}
    for neuron in range(num_neurons):
        signalDictionary['analyticSignal_{}'.format(neuron + 1)] = hilbert(SimulationObject.Vm_t[neuron, :])
        phaseListDictionary['phaseList_{}'.format(neuron + 1)] = []
        for z in signalDictionary['analyticSignal_{}'.format(neuron + 1)]:
            phi = np.arctan2(z.imag, z.real) # What could be interpretted as phase is the quadrant-preserving arctan of the imaginary component over the real component
            phaseListDictionary['phaseList_{}'.format(neuron + 1)].append(phi)

    neuron1_analyticSignal = signalDictionary['analyticSignal_{}'.format(neurons_to_compare[0])]
    neuron1_phase_t = phaseListDictionary['phaseList_{}'.format(neurons_to_compare[0])]
    neuron2_analyticSignal = signalDictionary['analyticSignal_{}'.format(neurons_to_compare[1])]
    neuron2_phase_t = phaseListDictionary['phaseList_{}'.format(neurons_to_compare[1])]

    phaseOffsets = [phi1 - phi2 for (phi1, phi2) in zip(neuron1_phase_t, neuron2_phase_t)]

    for index, offset in enumerate(phaseOffsets, start=0):
        if offset < -np.pi:
            phaseOffsets[index] = -(offset + 2*np.pi)
        elif offset > np.pi:
            phaseOffsets[index] = -(offset - 2*np.pi)

    if plotAnalyticSignals:
        plotStart = 0
        plt.close()
        fig, ax = plt.subplots(nrows=1, ncols=1)
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        ax.plot(neuron1_analyticSignal[plotStart:].real, neuron1_analyticSignal[plotStart:].imag, color='deeppink')
        ax.plot(neuron2_analyticSignal[plotStart:].real, neuron2_analyticSignal[plotStart:].imag, color='darkslateblue')
        plt.show()

    if computePLV:
        vectorList = []
        for timePoint in phaseOffsets:
            polarVector = np.exp(timePoint*1j)
            vectorList.append(polarVector)
        meanVector = sum(vectorList)/len(vectorList)
        PLV = abs(meanVector)
        print("Neuron {} and {} PLV: {}".format(neurons_to_compare[0], neurons_to_compare[1], PLV))
        return PLV
def timeDelayEmbed(SimulationObject, neuronIndex, asymp_Start, delay_ms, dt):

    delay_step = int(delay_ms/dt)
    asymp_Start = int(asymp_Start/dt)

    membraneVoltage = SimulationObject.Vm_t[neuronIndex, asymp_Start:]
    embedVector_1 = [Vm for Vm in membraneVoltage[:-2*delay_step]]
    embedVector_2 = [Vm for Vm in membraneVoltage[delay_step:-delay_step]]
    embedVector_3 = [Vm for Vm in membraneVoltage[2*delay_step:]]

    hh_figtype_7(embedVector_1, embedVector_2, embedVector_3)


def main():
    #=======================================================================================
    # ============================= CONTROL PANEL ==========================================
    #=======================================================================================
    # Specify the stimuli by amplitudes and durations; INDICES MUST MATCH
    #list(map(lambda x: x/1.0, range(1, 21, 2)))
    startTime = time.time()
    dt = 0.01 # (ms)
    amplitudes = [[14], [-3, 14]] # (uA/cm^2); MUST BE A 2D LIST; same units as SNNAP
    durations = [[500], [3, 497]] # (ms); MUST BE A 2D LIST
    commandFunction = stimulusFunction(amplitudes, durations, dt)
    DARKBACKGROUND = False # Suitable for Manim, Notion or slide decks with dark background
    SUPPRESSPOPUP = False
    NOTIONLOG = False
    HARDCODE_NERNST = False # False computes reversal potentials from concentrations
    COMPUTE_FREQUENCIES = True
    PLOT_ANALYTIC_SIGNALS = False
    TAKENS_DYNAMICS = False
    LEGENDS = False
    NUMBER_OF_NEURONS = 2
    iteration = 0
    # Set membrane capacitance, temperature and initial conditions
    Temp = 291.15 # (Kelvin) 291.15 is 18 Celsius
    Cm = [1.0 for i in range(NUMBER_OF_NEURONS)] # (uF/cm^2)
    Vm_0 = [-60.0 for i in range(NUMBER_OF_NEURONS)] # (mV)

    # Set average conductance per membrane area unit (mS/cm^2)
    gK = [36 for i in range(NUMBER_OF_NEURONS)] # Potassium conductance; 36 normally
    gNa = [120 for i in range(NUMBER_OF_NEURONS)] # Sodium conductance; 120 normally
    gL = [0.3 for i in range(NUMBER_OF_NEURONS)] # Leak conductance; 0.3 normally

    # Construct an adjacency matrix
    K = 1
    #adjacencyMatrix = np.array([[0]])
    adjacencyMatrix = np.array([[0, K],
                               [K, 0]])
    #adjacencyMatrix = np.array([[0, K, K],
    #                            [K, 0, K],
    #                            [K, K, 0]])
    # adjacencyMatrix = np.array([[0, 0, 0, K],
    #                            [K, 0, 0, 0],
    #                            [0, K, 0, 0],
    #                            [0, 0, K, 0]])
    # Initialize negative AP_times such that the alpha function is zero at simulation start
    AP_times = [-1000 for i in range(NUMBER_OF_NEURONS)]
    AP_threshold = 10 # mV

    # Select time constants for each presynaptic neuron
    timeConstants = [2 for i in range(NUMBER_OF_NEURONS)]

    # Specify ion concentrations (mM)
    if not HARDCODE_NERNST:
        # These must be list type iterables
        concentration_Na_Out = [145 for i in range(NUMBER_OF_NEURONS)] # Extracellular sodium; Normal: 145
        concentration_Na_In = [10.0 for i in range(NUMBER_OF_NEURONS)] # Intracellular sodium; Normal: 18
        concentration_K_Out = [3 for i in range(NUMBER_OF_NEURONS)] #list(map(lambda x: x/1.0, range(1, 21, 2))) #list(map(lambda x: x/1.0, range(1, 101, 1))) # Extracellular potassium; Normal: 3
        concentration_K_In = [135.0 for i in range(NUMBER_OF_NEURONS)] # Intracellular potassium; Normal: 135
        CONCENTRATIONS = {
            '[Na+]_o': concentration_Na_Out,
            '[Na+]_i': concentration_Na_In,
            '[K+]_o': concentration_K_Out,
            '[K+]_i': concentration_K_In
        }
        # Prepare parameters for model construction
        EK = []
        ENa = []
        EL = []
        for i in range(NUMBER_OF_NEURONS):
            EK.append(nernstPotential(concentration_K_In[i], concentration_K_Out[i], Temp, 1.0, "K+")) # Compute reversal potential for potassium
            ENa.append(nernstPotential(concentration_Na_In[i], concentration_Na_Out[i], Temp, 1.0, "Na+")) # Compute reversal potential for sodium
        EL = [-49.387 for i in range(NUMBER_OF_NEURONS)]
    else: # Explicitly specify reversal potentials for all ions and leaks
        EK = [-72.0]
        ENa = [55]
        EL = [-50.6]
        print("Nernst potential for K+: {}".format(EK))
        print("Nernst potential for Na+: {}".format(ENa))
        print("Nernst potential for Leak: {}".format(EL))
    
    # Package parameters for passing into simulation
    parameterDictionary = packageParameters(gK, gNa, gL, Cm, EK, ENa, EL, Vm_0, adjacencyMatrix, AP_times, timeConstants, AP_threshold, Temp)
    # ======================================================================================
    #=======================================================================================
    #=======================================================================================
    # ============================= INITIALIZE FOR NOTION API ==============================
    WORKING_DIRECTORY = os.path.dirname(os.path.realpath("modelFigures")).encode('unicode-escape').decode()
    if NOTIONLOG:
        NOTION_TOKEN_V2 =  # From dev tools; can change with logout
        NOTION_PAGE_URL =  # From profile, the page object with the database
        NOTION_COLLECTION_URL =  # The database object in the page
        notionCollection = notionInitialize(NOTION_TOKEN_V2, pageURL=NOTION_PAGE_URL, collectionURL=NOTION_COLLECTION_URL) # passable object for editing via the Notion API
    # ======================================================================================
    # ============================= EXECUTE SIMULATION =====================================
    if COMPUTE_FREQUENCIES:
        frequency_list = []

    # Construct model and simulation, run simulation, plot output
    hh_Network = HodgkinHuxleyNetwork(parameterDictionary) # Create the Hodgkin Huxley model class
    simulation = NetworkSimulation(hh_Network) # Create simulation object and pass in the model

    simulation.runSim(commandFunction, dt) # Run the simulation, duration is defined by the stimulus function (set amplitude 0 for desired duration for unperturbed simulation)
    print("===== {} m {} s to simulate =====".format((time.time() - startTime)/60, ((time.time() - startTime)%60)/1))
    plotSolutions(simulation, commandFunction, WORKING_DIRECTORY, NUMBER_OF_NEURONS, darkBackground=DARKBACKGROUND, notionOutput=False, legends=LEGENDS, suppresspopup=SUPPRESSPOPUP) # Kwargs: baseplot=True, ionCurrents_t=True, gateStates_t=True

    if NOTIONLOG:
        notionLog(notionCollection, iteration + 1, PARAMETERS, CONCENTRATIONS)
    if COMPUTE_FREQUENCIES:
        for neuronIndex in range(NUMBER_OF_NEURONS):
            frequency_list.append(computeFrequency(simulation, neuronIndex))
    print("Frequencies were: {}".format(frequency_list))
    phaseSpaceAnalysis(simulation, NUMBER_OF_NEURONS, (1, 2), plotAnalyticSignals=PLOT_ANALYTIC_SIGNALS)
    if TAKENS_DYNAMICS:
        timeDelayEmbed(simulation, 0, 10, 2, dt)
    print("===== {} m {} s to compute dynamics =====".format((time.time() - startTime)//60, ((time.time() - startTime)%60)//1))
    # ======================================================================================

if __name__ == "__main__":
    main()