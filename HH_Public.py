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
import random
import numpy as np
import matplotlib.pyplot as plt
from notion.client import NotionClient
from datetime import datetime
from scipy.stats import mode
from figureFunctions import hh_figtype_1

class HodgkinHuxley:
    class Gate:
        alpha, beta, state = 0, 0, 0 # Initialize gate state and rate constants

        def update(self, dt):
            # Pass in timestep in ms
            alphaState = self.alpha*(1 - self.state) # Forward rate constant times unopen population
            betaState = self.beta*self.state # Backward rate constant times open population
            self.state += dt*(alphaState - betaState) # dP/dt = a(1-P) - B(P) => dP = dt(a(1-P) - B(P)); add dP to P 
        def setEquilibirumState(self):
            self.state = self.alpha/(self.alpha + self.beta) # [P = Peq - (Peq - P0)*exp(-t/Tau)] => [lim(t->inf)P = Peq = a/(a+B)] 
    
    n, m, h = Gate(), Gate(), Gate() 
    
    def __init__(self, parameterDictionary, leak_only=False):
        # Set average channel conductances per unit area (mS/cm^2)
        self.parameters = parameterDictionary
        self.gK = parameterDictionary['gK']
        self.gNa = parameterDictionary['gNa']
        self.gL = parameterDictionary['gL']
        self.Cm = parameterDictionary['Cm'] # Membrane capacitance (uF/cm^2)
        # Set nernst potentials as passed (mV)
        self.EK = parameterDictionary['EK']
        self.ENa = parameterDictionary['ENa']
        self.EL = parameterDictionary['EL']
        self.Vm = parameterDictionary['Vm'] # Initial membrane potential (mV)
        self.leakChannelsOnly = leak_only
        
        self._updateGateRateConstants(self.Vm) # Initialize rate constants with initial Vm
        # Set equilibrium states for each gate based on the analytical solution to their ODEs
        self.n.setEquilibirumState() 
        self.m.setEquilibirumState()
        self.h.setEquilibirumState()
    def _updateGateRateConstants(self, Vm):
        # Best fit values from literature for convention of -60mV resting potential; HH 1952 took Vm = 0
        self.n.alpha = (0.01*(-50 - Vm))/(np.exp((-50 - Vm)/10) - 1)
        self.n.beta = 0.125*np.exp((-Vm - 60)/80)

        self.m.alpha = 0.1*(-35 - Vm)/(np.exp((-35 - Vm)/10) - 1)
        self.m.beta = 4*np.exp((-Vm - 60)/18)

        self.h.alpha = 0.07*np.exp((-60 - Vm)/20)
        self.h.beta = 1/(np.exp((-30 - Vm)/10) + 1)
    def _updateVm(self, forcing, dt):
        self.IK = np.power(self.n.state, 4)*self.gK*(self.Vm - self.EK) # 4th order n-gating * average potassium conductance (mS/cm^2) * potassium driving force
        self.INa = np.power(self.m.state, 3)*self.h.state*self.gNa*(self.Vm - self.ENa) # 3rd order m-gating * 1st order trenchant h-gate * average sodium conductance (mS/cm^2) * sodium driving force
        self.IL = self.gL*(self.Vm - self.EL) # Linear leak conductance combining sodium and potassium variability; how is the reversal potential for this computed?
        if not self.leakChannelsOnly:
            currentSuperposition = forcing - self.IK - self.INa - self.IL # Current obeys principle of superposition, sum all currents for total
        else: 
            currentSuperposition = forcing - self.IL
        self.Vm += dt*(currentSuperposition/self.Cm) 
    def _updateGateStates(self, dt):
        # Step the gating variable ODEs
        self.n.update(dt)
        self.m.update(dt)
        self.h.update(dt)
    def iterate(self, forcing, dt):
        # Step rate constants, membrane potential and gating variable ODEs; how should these be ordered? This may be a limitation of discrete time solutions. 
        self._updateGateRateConstants(self.Vm)
        self._updateVm(forcing, dt)
        self._updateGateStates(dt)
class Simulation:
    def __init__(self, model):
        self.model = model
        self.vectorInitialize(0, 0)
        pass
    def vectorInitialize(self, num_timesteps, dt):
        self.times = np.arange(num_timesteps) * dt
        self.Vm = np.empty(num_timesteps)
        self.IK = np.empty(num_timesteps)
        self.INa = np.empty(num_timesteps)
        self.IL = np.empty(num_timesteps)
        self.nState = np.empty(num_timesteps)
        self.mState = np.empty(num_timesteps)
        self.hState = np.empty(num_timesteps)
    def runSim(self, commandFunction, dt):
        self.vectorInitialize(len(commandFunction), dt)
        print("Initialization complete...")
        print("Simulating {} ms;\nParameters: {}".format(len(commandFunction)*dt, self.model.parameters))
        for i in range(len(commandFunction)):
            self.model.iterate(commandFunction[i], dt)
            self.Vm[i] = self.model.Vm
            self.IK[i] = self.model.IK
            self.INa[i] = self.model.Vm
            self.IL[i] = self.model.IL
            self.nState[i] = self.model.n.state
            self.mState[i] = self.model.m.state
            self.hState[i] = self.model.h.state
        print("HALT")

def packageParameters(gK, gNa, gL, Cm, EK, ENa, EL, Vm_0, T):
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
    iterable = selectIterable(concentration_In, concentration_Out, temperature)

    faradayConstant = 96485.33212 # Coulombs/mol
    gasConstant = 8.3145 # J/(mol*K)
    nernstList = []
    for i, concentration in enumerate(iterable, start=0):
        try:
            if valence > 0: # For cations the logarithmic argument is uninverted
                nernst = (gasConstant*temperature[i]*np.log(concentration_Out[i]/concentration_In[i]))/(valence*faradayConstant)
            elif valence < 0: # For anions the logarithmic argument is inverted, then the valence made positive
                nernst = (gasConstant*temperature[i]*np.log(concentration_In[i]/concentration_Out[i]))/(np.abs(valence)*faradayConstant)
            else:
                raise Exception("Cannot compute reversal potential for chargeless species") # Uncharged species are not subject to coulomb forces
            nernst *= 1000 #(V => mV) # Dimensional conversion
            nernstList.append(nernst)
        except IndexError:
            if iterable == concentration_In:
                if valence > 0: # For cations the logarithmic argument is uninverted
                    nernst = (gasConstant*temperature[-1]*np.log(concentration_Out[-1]/concentration_In[i]))/(valence*faradayConstant)
                elif valence < 0: # For anions the logarithmic argument is inverted, then the valence made positive
                    nernst = (gasConstant*temperature[-1]*np.log(concentration_In[i]/concentration_Out[-1]))/(np.abs(valence)*faradayConstant)
                else:
                    raise Exception("Cannot compute reversal potential for chargeless species") # Uncharged species are not subject to coulomb forces
                nernst *= 1000 #(V => mV) # Dimensional conversion
                nernstList.append(nernst)
            elif iterable == concentration_Out:
                if valence > 0: # For cations the logarithmic argument is uninverted
                    nernst = (gasConstant*temperature[-1]*np.log(concentration_Out[i]/concentration_In[-1]))/(valence*faradayConstant)
                elif valence < 0: # For anions the logarithmic argument is inverted, then the valence made positive
                    nernst = (gasConstant*temperature[-1]*np.log(concentration_In[-1]/concentration_Out[i]))/(np.abs(valence)*faradayConstant)
                else:
                    raise Exception("Cannot compute reversal potential for chargeless species") # Uncharged species are not subject to coulomb forces
                nernst *= 1000 #(V => mV) # Dimensional conversion
                nernstList.append(nernst)
            elif iterable == temperature:
                if valence > 0: # For cations the logarithmic argument is uninverted
                    nernst = (gasConstant*temperature[i]*np.log(concentration_Out[-1]/concentration_In[-1]))/(valence*faradayConstant)
                elif valence < 0: # For anions the logarithmic argument is inverted, then the valence made positive
                    nernst = (gasConstant*temperature[i]*np.log(concentration_In[-1]/concentration_Out[-1]))/(np.abs(valence)*faradayConstant)
                else:
                    raise Exception("Cannot compute reversal potential for chargeless species") # Uncharged species are not subject to coulomb forces
                nernst *= 1000 #(V => mV) # Dimensional conversion
                nernstList.append(nernst)
    print("Nernst potential for {}: {} mV".format(ionName, nernstList)) # Print the reversal potential to the console
    return nernstList
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
def plotSolutions(SimulationObject, commandFunction, workingDirectory, iterationNumber, baseplot=True, ionCurrents_t=True, gateStates_t=True, darkBackground=False, notionOutput=False, suppresspopup=False):
    """
    Takes a completed hodgkin huxley simulation object and a command function,
    creates a plot of the solutions specified in *kwargs: baseplot includes membrane
    voltage and injected current, ion currents include Na+, K+ and leak, gate states
    are the proportion of a population of gates that are active -- i.e. permitting
    ion conductance. darkBackground is stylistic and makes plots suitable for Manim,
    Notion and slide decks with dark backgrounds.
    """
    if not darkBackground:
        timeAxis = SimulationObject.times

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

        plotCount = 0
        if baseplot:
            axes[plotCount].plot(timeAxis, SimulationObject.Vm, linewidth=0.8, color='deeppink', label='Vm')
            axes[plotCount].plot(timeAxis, commandFunction, linewidth=0.8, color='dodgerblue', label='Forcing ($\mu$A/$cm^{2}$)')
            axes[plotCount].set_ylabel("Vm ($mV$)")
            if plotCount == number_of_plots:
                axes[plotCount].set_xlabel("Time (ms)")
            axes[plotCount].legend(edgecolor='w', framealpha=0.0, bbox_to_anchor=(1.05, 1), loc='upper left')
            plotCount += 1
        if ionCurrents_t:
            axes[plotCount].plot(timeAxis, SimulationObject.IK, linewidth=0.8, color='cyan', label="$K^{+}$")
            axes[plotCount].plot(timeAxis, SimulationObject.INa, linewidth=0.8, color='orangered', label="$Na^{+}$")
            axes[plotCount].plot(timeAxis, SimulationObject.IL, linewidth=0.8, color='blueviolet', label="Leak")
            axes[plotCount].set_ylabel("Current ($\mu A/cm^{2}$)")
            if plotCount == number_of_plots:
                axes[plotCount].set_xlabel('Time (ms)')
            axes[plotCount].legend(title='Ion Current', edgecolor='w', framealpha=0.0, bbox_to_anchor=(1.05, 1), loc='upper left')
            plotCount += 1
        if gateStates_t:
            axes[plotCount].plot(timeAxis, SimulationObject.nState, linewidth=0.8, color='olivedrab', label="n")
            axes[plotCount].plot(timeAxis, SimulationObject.mState, linewidth=0.8, color='red', label="m")
            axes[plotCount].plot(timeAxis, SimulationObject.hState, linewidth=0.8, color='gold', label="h")
            axes[plotCount].set_ylabel("Active fraction")
            if plotCount == number_of_plots:
                axes[plotCount].set_xlabel('Time (ms)')
            axes[plotCount].legend(title='Gate', edgecolor='w', framealpha=0.0, bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.savefig('{}\\HodgkinHuxleyModel\\modelFigures\\HHSolution_{}.png'.format(workingDirectory, iterationNumber), bbox_inches='tight')
    else:
        timeAxis = SimulationObject.times

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
            axes[plotCount].plot(timeAxis, SimulationObject.Vm, linewidth=0.8, color='w', label='Vm')
            axes[plotCount].plot(timeAxis, commandFunction, linewidth=0.8, color='w', label='Forcing ($\mu$A/$cm^{2}$)', linestyle='dotted')
            axes[plotCount].set_ylabel("$V_{m}$ ($mV$)", color='w')
            legend = axes[plotCount].legend(edgecolor='w', framealpha=0.0, bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.setp(legend.get_texts(), color='w')
            plotCount += 1
            if plotCount == number_of_plots:
                axes[plotCount - 1].set_xlabel("Time (ms)", color='w')
        if ionCurrents_t:
            axes[plotCount].plot(timeAxis, SimulationObject.IK, linewidth=0.8, color='w', label="$K^{+}$Current")
            axes[plotCount].plot(timeAxis, SimulationObject.INa, linewidth=0.8, color='w', label="$Na^{+}$Current", linestyle='dashdot')
            axes[plotCount].plot(timeAxis, SimulationObject.IL, linewidth=0.8, color='w', label="Leak Current", linestyle='dotted')
            #axes[plotCount].plot(timeAxis, commandFunction, linewidth=0.8, color='w', label='Forcing', linestyle='dashed')
            axes[plotCount].set_ylabel("Current ($\mu A/cm^{2}$)", color='w')
            legend = axes[plotCount].legend(edgecolor='w', framealpha=0.0, bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.setp(legend.get_texts(), color='w')
            plotCount += 1
            if plotCount == number_of_plots:
                axes[plotCount - 1].set_xlabel('Time (ms)', color='w')
        if gateStates_t:
            axes[plotCount].plot(timeAxis, SimulationObject.nState, linewidth=0.8, color='w', label="n Gate")
            axes[plotCount].plot(timeAxis, SimulationObject.mState, linewidth=0.8, color='w', label="m Gate", linestyle="dotted")
            axes[plotCount].plot(timeAxis, SimulationObject.hState, linewidth=0.8, color='w', label="h Gate", linestyle='dashed')
            axes[plotCount].set_ylabel("Active fraction", color='w')
            legend = axes[plotCount].legend(edgecolor='w', framealpha=0.0, bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.setp(legend.get_texts(), color='w')
            plotCount += 1 
            if plotCount == number_of_plots:
                axes[plotCount - 1].set_xlabel('Time (ms)', color='w')

        plt.savefig('{}\\HodgkinHuxleyModel\\modelFigures\\HHSolution_{}.png'.format(workingDirectory, iterationNumber), bbox_inches='tight')

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
def updateParameters(iteration, temps, Cms, Vm_0s, gKs, gNas, gLs, EKs, ENas, ELs):
        try:
            currentTemp = temps[iteration]
        except IndexError:
            currentTemp = temps[-1]
        try:
            currentCm = Cms[iteration]
        except IndexError:
            currentCm = Cms[-1]
        try:
            currentVm_0 = Vm_0s[iteration]
        except IndexError:
            currentVm_0 = Vm_0s[-1]
        try:
            currentgK = gKs[iteration]
        except IndexError:
            currentgK = gKs[-1]
        try:
            currentgNa = gNas[iteration]
        except IndexError:
            currentgNa = gNas[-1]
        try:
            currentgL = gLs[iteration]
        except IndexError:
            currentgL = gLs[-1]
        try:
            currentEK = EKs[iteration]
        except IndexError:
            currentEK = EKs[-1]
        try:
            currentENa = ENas[iteration]
        except IndexError:
            currentENa = ENas[-1]
        try:
            currentEL = ELs[iteration]
        except IndexError:
            currentEL = ELs[-1]
        parameters = packageParameters(currentgK, currentgNa, currentgL, currentCm, currentEK, currentENa, currentEL, currentVm_0, currentTemp)
        return parameters
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
def restingVm(SimulationObject, flat_segment_max_derivative):
    #return mode(SimulationObject.Vm)[0][0]
    try:
        Vm_solution = SimulationObject.Vm[len(SimulationObject.Vm)//2:]
        dVm_dt_Discrete = np.diff(Vm_solution)
        flat_Segment_Running_Sum = 0
        num_datapoints = 0
        for i, dVm_dt in enumerate(dVm_dt_Discrete, start=0):
            if dVm_dt <= flat_segment_max_derivative:
                flat_Segment_Running_Sum += Vm_solution[i]
                num_datapoints += 1
        print("Guess resting voltage: {} mV".format(flat_Segment_Running_Sum/num_datapoints))
    except ZeroDivisionError:
        flat_segment_max_derivative *= 2
        print("restingVm: INCREASED MAX VOLTAGE DERIVATIVE TO {} mV/s".format(flat_segment_max_derivative))
        Vm_solution = SimulationObject.Vm[len(SimulationObject.Vm)//2:]
        dVm_dt_Discrete = np.diff(Vm_solution)
        flat_Segment_Running_Sum = 0
        num_datapoints = 0
        for i, dVm_dt in enumerate(dVm_dt_Discrete, start=0):
            if dVm_dt <= flat_segment_max_derivative:
                flat_Segment_Running_Sum += Vm_solution[i]
                num_datapoints += 1
        print("Guess resting voltage: {} mV".format(flat_Segment_Running_Sum/num_datapoints))
    return flat_Segment_Running_Sum/num_datapoints

def main():
    #=======================================================================================
    # ============================= CONTROL PANEL ==========================================
    #=======================================================================================
    # Specify the stimuli by amplitudes and durations; INDICES MUST MATCH
    dt = 0.01 # (ms)
    amplitudes = [[0, 55, 0]] # (uA/cm^2); MUST BE A 2D LIST; same units as SNNAP
    durations = [[30, 1, 30]] # (ms); MUST BE A 2D LIST
    commandFunction = stimulusFunction(amplitudes, durations, dt)
    DARKBACKGROUND = False # Suitable for Manim, Notion or slide decks with dark background
    SUPPRESSPOPUP = False
    NOTIONLOG = False
    HARDCODE_NERNST = False # False computes reversal potentials from concentrations
    PLOT_RESTING_VOLTAGES = False
    LEAK_ONLY = False
    KtoNaLEAK_PERMEABILITY_RATIO = 20
    # Set membrane capacitance, temperature and initial conditions
    Temp = [291.15] # (Kelvin) 291.15 is 18 Celsius
    Cm = [1.0] # (uF/cm^2)
    Vm_0 = [-60.0] # (mV)

    # Set average conductance per membrane area unit (mS/cm^2)
    gK = [36.0] # Potassium conductance
    gNa = [120.0] # Sodium conductance
    gL = [3.0] # Leak conductance 0.3 normally

    # Specify ion concentrations (mM)
    if not HARDCODE_NERNST:
        # These must be list type iterables
        concentration_Na_Out = [145] # Extracellular sodium; Normal: 145
        concentration_Na_In = [10.0] # Intracellular sodium; Normal: 18
        concentration_K_Out = [3.0] #list(map(lambda x: x/1.0, range(1, 200, 2))) #list(map(lambda x: x/1.0, range(1, 101, 1))) # Extracellular potassium; Normal: 3
        concentration_K_In = [135.0] # Intracellular potassium; Normal: 135
        CONCENTRATIONS = {
            '[Na+]_o': concentration_Na_Out,
            '[Na+]_i': concentration_Na_In,
            '[K+]_o': concentration_K_Out,
            '[K+]_i': concentration_K_In
        }
        concentration_L_Out = []
        concentration_L_In = []
        iterable = selectIterable(concentration_K_In, concentration_K_Out, concentration_Na_In, concentration_Na_Out)
        print("iterable is {}".format(iterable))
        for i in range(len(iterable)):
            try:
                concentration_L_Out.append(concentration_Na_Out[i] + concentration_K_Out[i]*KtoNaLEAK_PERMEABILITY_RATIO) # Leak extracellular is a sum of Na and K (?)
                concentration_L_In.append(concentration_Na_In[i] + concentration_K_In[i]*KtoNaLEAK_PERMEABILITY_RATIO) # Leak intracellular is a sum of Na and K (?)
            except IndexError:
                if iterable == concentration_Na_Out:
                    concentration_L_Out.append(concentration_Na_Out[i] + concentration_K_Out[-1]*KtoNaLEAK_PERMEABILITY_RATIO) # Leak extracellular is a sum of Na and K (?)
                elif iterable == concentration_Na_In:
                    concentration_L_In.append(concentration_Na_In[i] + concentration_K_In[-1]*KtoNaLEAK_PERMEABILITY_RATIO) # Leak intracellular is a sum of Na and K (?)
                elif iterable == concentration_K_Out:
                    concentration_L_Out.append(concentration_Na_Out[-1] + concentration_K_Out[i]*KtoNaLEAK_PERMEABILITY_RATIO) # Leak extracellular is a sum of Na and K (?)
                elif iterable == concentration_K_In:
                    concentration_L_In.append(concentration_Na_In[-1] + concentration_K_In[i]*KtoNaLEAK_PERMEABILITY_RATIO) # Leak intracellular is a sum of Na and K (?)
        # Prepare parameters for model construction
        EK = nernstPotential(concentration_K_In, concentration_K_Out, Temp, 1.0, "K+") # Compute reversal potential for potassium
        try:
            ENa = nernstPotential(concentration_Na_In, concentration_Na_Out, Temp, 1.0, "Na+") # Compute reversal potential for sodium
        except ZeroDivisionError:
            ENa = [-430.5]
        EL = nernstPotential(concentration_L_In, concentration_L_Out, Temp, 1.0, "Leak") # Compute reversal potential for leak
        #EL = [-50.6]
    else: # Explicitly specify reversal potentials for all ions and leaks
        EK = [-72.0]
        ENa = [55]
        EL = [-50.6]
        print("Nernst potential for K+: {}".format(EK))
        print("Nernst potential for Na+: {}".format(ENa))
        print("Nernst potential for Leak: {}".format(EL))
    # ======================================================================================
    #=======================================================================================
    #=======================================================================================
    # ============================= INITIALIZE FOR NOTION API ==============================
    WORKING_DIRECTORY = # for me: os.path.dirname(os.path.realpath("modelFigures")).encode('unicode-escape').decode()
    if NOTIONLOG:
        NOTION_TOKEN_V2 = "" # From dev tools; can change with logout
        NOTION_PAGE_URL = "" # From profile, the page object with the database
        NOTION_COLLECTION_URL = "" # The database object in the page
        notionCollection = notionInitialize(NOTION_TOKEN_V2, pageURL=NOTION_PAGE_URL, collectionURL=NOTION_COLLECTION_URL) # passable object for editing via the Notion API
    # ======================================================================================
    # ============================= EXECUTE SIMULATION =====================================
    if PLOT_RESTING_VOLTAGES:
        Vm_resting = []
    longestIterable = selectIterable(amplitudes, durations, Temp, Cm, Vm_0, concentration_Na_Out, concentration_Na_In, concentration_K_Out, concentration_K_In, EK, ENa, EL, gK, gNa, gL)
    for iteration in range(len(longestIterable)):
        print("===== ITERATION {} =====".format(iteration + 1))
        PARAMETERS = updateParameters(iteration, Temp, Cm, Vm_0, gK, gNa, gL, EK, ENa, EL) # Package parameters in dictionary with preset keys

        # Construct model and simulation, run simulation, plot output
        hodgkinHuxleyModel = HodgkinHuxley(PARAMETERS, leak_only=LEAK_ONLY) # Create the Hodgkin Huxley model class
        simulation = Simulation(hodgkinHuxleyModel) # Create simulation object and pass in the model
        try:
            simulation.runSim(commandFunction[iteration], dt) # Run the simulation, duration is defined by the stimulus function (set amplitude 0 for desired duration for unperturbed simulation)
            plotSolutions(simulation, commandFunction[iteration], WORKING_DIRECTORY, iteration + 1, darkBackground=DARKBACKGROUND, notionOutput=False, suppresspopup=SUPPRESSPOPUP) # Kwargs: baseplot=True, ionCurrents_t=True, gateStates_t=True
        except IndexError:
            simulation.runSim(commandFunction[-1], dt) # Run the simulation, duration is defined by the stimulus function (set amplitude 0 for desired duration for unperturbed simulation)
            plotSolutions(simulation, commandFunction[-1], WORKING_DIRECTORY, iteration + 1, darkBackground=DARKBACKGROUND, notionOutput=False, suppresspopup=SUPPRESSPOPUP) # Kwargs: baseplot=True, ionCurrents_t=True, gateStates_t=True
        if NOTIONLOG:
            notionLog(notionCollection, iteration + 1, PARAMETERS, CONCENTRATIONS)
        if PLOT_RESTING_VOLTAGES:
            Vm_resting.append(restingVm(simulation, dt/10))
    if PLOT_RESTING_VOLTAGES:
        print("Resting membrane potentials were:\n{}\n=======================================================================".format(Vm_resting))
        hh_figtype_1(Vm_resting, concentration_K_Out)
    # ======================================================================================

if __name__ == "__main__":
    main()