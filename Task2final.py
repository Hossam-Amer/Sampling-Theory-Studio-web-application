import numpy as np
from numpy import fft
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# -------------------functoin---------------------------------
def generate_sin(sin_cos, Fs, F0, amp, length):

    tStep = 1 / Fs  # Every (tStep) We Get One Sample

    t = np.linspace(0, (length - 1) * tStep, length)
    if sin_cos == "sin":
        y = amp * np.sin(2 * np.pi * F0 * t)
    else:
        y = amp * np.cos(2 * np.pi * F0 * t)

    return y
st.markdown("""
<style>
#MainMenu
{
    visibility: hidden;
}
.css-cio0dv.egzxvld1 
{
    visibility: hidden;
}
 </style>
""", unsafe_allow_html=True)
st.title("Sampling Theory Studio")
opt = st.sidebar.radio("Select Signal", options=(
    "Input file", "Create sinusoidal"))
if opt == "Input file":
    with st.sidebar:
        csv_file1 = st.file_uploader("Upload CSV file")
    if csv_file1 is not None:
        data = np.genfromtxt(csv_file1, delimiter=',')
        # Get the number of samples and the sampling rate from the signal data
        num_samples = data.shape[0]
        Fs = 1 / (data[1, 0] - data[0, 0])
        with st.sidebar:
            factor = st.slider("Sampling Frequency x0.1 Fmax", 1, 50, 1)*0.1
            factor_noise = st.slider('Signal to Noise Ratio (SNR)', 1, 5, 100)
        # Compute the Nyquist frequency and the Nyquist rate
        Fmax = Fs / 2
        new_Fs = factor * Fmax
        # Get the signal values and time stamps from the data
        signal = data[:, 1]
        time = data[:, 0]
        # Perform sampling using the Nyquist rate
        num_samples = int(new_Fs * (time[-1] - time[0]))
        new_time_sampled = np.linspace(time[0], time[-1], num_samples)
        signal_sampled = np.interp(new_time_sampled, time, signal)+ (factor_noise*0.01) *np.random.randn(len(new_time_sampled))
        # Plot the original signal
        plt.style.use(
            'https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle')
        fig1, axs1 = plt.subplots()
        fig2, axs2 = plt.subplots()
        fig3, axs3 = plt.subplots()
        axs1.plot(time, signal)
        axs1.set_xlabel('Time')
        axs1.set_ylabel('Amplitude')
        axs2.plot(new_time_sampled, signal_sampled)
        axs2.set_title('Sampled Signal')
        axs3.plot(time, signal)
        axs3.plot(new_time_sampled, signal_sampled)
        axs3.set_title('Reconstructed Signal')
        plt.tight_layout()
        plt.show()
        st.pyplot(fig1)
        st.pyplot(fig2)
        st.pyplot(fig3)


if opt == "Create sinusoidal":
    time = np.linspace(0, 2*np.pi, 500)
    freqs = [1, 1, 1]
    signal=0
    ambs = [1, 1, 1]
    with st.sidebar:
        sin_cos = st.radio("Wave (1)", options=("sin", "cos"))
        ambs[0] = st.number_input('amplitude  (1)', step=5, value=10)
        freqs[0] = st.number_input('frequency (1)', step=5, value=10)
        sin_cos = st.radio("Wave (2)", options=("sin", "cos"))
        ambs[1] = st.number_input('amplitude  (2)', step=5, value=10)
        freqs[1] = st.number_input('frequency (2)', step=5, value=10)
        sin_cos = st.radio("Wave (3)", options=("sin", "cos"))
        ambs[2] = st.number_input('amplitude  (3)', step=5, value=10)
        freqs[2] = st.number_input('frequency (3)', step=5, value=10)
        #Add noise slider:
        factor_noise = st.number_input('Noise', step=5, value=10)
        for idx in range(len(freqs)):
            signal += generate_sin(sin_cos, 2000,int(freqs[idx]), int(ambs[idx]), len(time)) +factor_noise
    # Plot the original signal
        plt.style.use(
        'https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle')
  
  
  
  
  
  
        Fs = 2*np.max(freqs)
        with st.sidebar:
            factor = st.slider("Sampling Frequency x0.1 Fmax", 1, 50, 1)*0.1

            factor_noise = st.slider('Signal to Noise Ratio (SNR)', 1, 5, 100)

        # Compute the Nyquist frequency and the Nyquist rate
        Fmax = Fs / 2
        new_Fs = factor * Fmax
        # Get the signal values and time stamps from the data
        # Perform sampling using the Nyquist rate
        num_samples = int(new_Fs * (time[-1] - time[0]))
        new_time_sampled = np.linspace(time[0], time[-1], num_samples)
        signal_sampled = np.interp(new_time_sampled, time, signal)+ (factor_noise*0.01) *np.random.randn(len(new_time_sampled))
        # Plot the original signal
        plt.style.use(
            'https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle')
    fig1, axs1 = plt.subplots()
    fig2, axs2 = plt.subplots()
    fig3, axs3 = plt.subplots()
    axs1.plot(time, signal)
    axs1.set_xlabel('Time')
    axs1.set_ylabel('Amplitude')
    axs2.plot(new_time_sampled, signal_sampled)
    axs2.set_title('Sampled Signal')
    axs3.plot(time, signal)
    axs3.plot(new_time_sampled, signal_sampled)
    axs3.set_title('Reconstructed Signal')
    plt.tight_layout()
    plt.show()
    st.pyplot(fig1)
    st.pyplot(fig2)
    st.pyplot(fig3)

  
  
  
  
  
  
  
  
  
  
  
  
    # fig1, ax1 = plt.subplots()
    # ax1.plot(t, y)
    # ax1.set_xlabel('Time')
    # ax1.set_ylabel('Amplitude')
    # st.pyplot(fig1)
    # n = np.arange(0, len(t))
    # idx = np.arange(0, len(t), int(10))  # Sample indices
    # y_sampled = y[idx]
    # # Plot the sampled signal
    # fig3, ax3 = plt.subplots()
    # ax3.stem(n[idx], y_sampled, label='Sampled')
    # ax3.set_xlabel('Time')
    # ax3.set_ylabel('Amplitude')
    # ax3.legend()
    # st.pyplot(fig3)
    # # Reconstruct the signal
    # y_reconstructed = np.zeros(len(t))
    # for i in range(len(idx)):
    #     y_reconstructed[idx[i]] = y_sampled[i]
    # # Plot the reconstructed signal over the original signal
    # fig4, ax4 = plt.subplots()
    # ax4.plot(t, y, label='Original')
    # ax4.plot(t, y_reconstructed, label='Reconstructed')
    # ax4.set_xlabel('Time')
    # ax4.set_ylabel('Amplitude')
    # ax4.legend()
    # st.pyplot(fig4)
