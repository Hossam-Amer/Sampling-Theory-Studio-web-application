import numpy as np
from numpy import fft
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd

def generate_sin(Fs, F0, amp, length):

    tStep = 1 / Fs  # Every (tStep) We Get One Sample

    t = np.linspace(0, (length - 1) * tStep, length)
    return amp * np.sin(2 * np.pi * F0 * t )
    
def generate_sin_with_phase(Fs, F0, amp, length,phase):
    tStep = 1 / Fs  # Every (tStep) We Get One Sample
    t = np.linspace(0, (length - 1) * tStep, length)
    return amp * np.sin(2 * np.pi * F0 * t + phase*(np.pi/180)) #Turn from degree to redian
    
def generate_sinusoidal_signal(amplitude, phase, frequency, duration, sampling_rate):
    t = np.linspace(0, duration, int(duration * sampling_rate), endpoint=False)
    signal = amplitude * np.sin(2 * np.pi * frequency * t + phase)
    return signal

def save_csv(signal):
    df=pd.DataFrame(signal,columns=None)
    df.to_csv("saved_file.csv", index=True)

def plotGraphs(time,time_sampled,signal,samples,signal_sampled):
    col1,col2= st.columns((1,1))
    # plt.style.use(
    #       'https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle')
    with col1:
        fig1, axs1 = plt.subplots()
        fig1, axs1 = plt.subplots(figsize=(10, 5))
        axs1.plot(time, signal)
        axs1.scatter(time_sampled, samples, color="white")

        axs1.set_xlabel('Time')
        axs1.set_title('Original Signal', fontsize=24)
        axs1.set_ylabel('Amplitude', fontsize=20)
        fig3, axs3 = plt.subplots(figsize=(10, 5))
        # axs3.plot(time, signal)
        axs3.plot(time, signal_sampled,color="white")
        axs3.plot(time, signal)
        
        axs3.set_title('Original & Constructed ', fontsize=24)
        axs3.set_ylabel('Amplitude', fontsize=13)
        st.pyplot(fig1)
        st.pyplot(fig3)
    with col2:
        fig2, axs2 = plt.subplots(figsize=(10, 5))
        axs2.plot(time, signal_sampled)
        axs2.set_title('Reconstructed Signal', fontsize=24)
        fig4, axs4 = plt.subplots(figsize=(10, 5))
        differenceSignal=signal-signal_sampled

        axs4.plot(time, differenceSignal)
        axs4.plot(time, signal_sampled, alpha=0)

        axs4.set_title('Diffrence', fontsize=24)
        axs4.set_ylabel('Amplitude', fontsize=13)
        st.pyplot(fig2)
        st.pyplot(fig4)
    plt.tight_layout()
    plt.show()

def plotgraphs():
    col1, col2 = st.columns((1, 1))
    # plt.style.use(
    #     'https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle')
    with col1:
        fig1, axs1 = plt.subplots()
        fig1, axs1 = plt.subplots(figsize=(10, 5))
        axs1.set_xlabel('Time')
        axs1.set_title('Original Signal', fontsize=24)
        axs1.set_ylabel('Amplitude', fontsize=20)
        fig3, axs3 = plt.subplots(figsize=(10, 5))
        axs3.set_title('Original & Constructed ', fontsize=24)
        axs3.set_ylabel('Amplitude', fontsize=13)
        st.pyplot(fig1)
        st.pyplot(fig3)
    with col2:
        fig2, axs2 = plt.subplots(figsize=(10, 5))
        axs2.set_title('Sampled Signal', fontsize=24)
        fig4, axs4 = plt.subplots(figsize=(10, 5))        
        axs4.set_title('Diffrence', fontsize=24)
        axs4.set_ylabel('Amplitude', fontsize=13)
        st.pyplot(fig2)
        st.pyplot(fig4)
    plt.tight_layout()
    plt.show()

def plotPhase(sin,time,signal):
     plt.style.use(
          'https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle')
     if sin == 'With Phase':
            fig, axs = plt.subplots()
            fig, axs = plt.subplots(figsize=(20, 2))
            axs.plot(time, np.angle(signal))
            axs.set_xlabel('Time')
            axs.set_title('Phase', fontsize=25)
            axs.set_ylabel('Amplitude')
            st.pyplot(fig)    

def markdown():
    st.markdown("""
        <style>
        #MainMenu
        {
            visibility: hidden;
        }
        .css-10pw50.egzxvld1 
        {
            visibility: hidden;
        }
        </style>
        """, unsafe_allow_html=True)
    
    
def interpolate(time_new, signal_time, signal_amplitude):

    
            
## Interpolation using the whittaker-shannon interpolation formula that sums shifted and weighted sinc functions to give the interpolated signal
## Each sample in original signal corresponds to a sinc function centered at the sample and weighted by the sample amplitude
## summing all these sinc functions at the new time array points gives us the interploated signal at these points.     



    # sincM is a 2D matrix of shape (len(signal_time), len(time_new))
    
    # By subtracting the sampled time points from the interpolated time points
    
    sincMatrix = np.tile(time_new, (len(signal_time), 1)) - np.tile(signal_time[:, np.newaxis], (1, len(time_new)))

    # sinc interpolation 
    
    #This dot product results in a new set of amplitude values that approximate the original signal at the interpolated time points.
    signal_interpolated = np.dot(signal_amplitude, np.sinc(sincMatrix/(signal_time[1] - signal_time[0])))   
    return signal_interpolated