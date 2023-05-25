import numpy as np
from numpy import fft
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import functions as fn

st.set_page_config(
page_title="Sampling theory studio",
layout = "wide",
initial_sidebar_state="expanded",
page_icon="./icons/picture.png"
)

opt = st.sidebar.radio("Select Signal", options=(
    "Input file", "Create sinusoidal"))

if opt == "Input file":
    with st.sidebar:
        csv_file = st.file_uploader("Upload CSV file")
    if csv_file is None:
        fn.plotgraphs()
    if csv_file is not None:
        data = np.genfromtxt(csv_file, delimiter=',')
        # Get the number of samples and the sampling rate from the signal data
        num_samples = data.shape[0]
        Fs = 1 / (data[1, 0] - data[0, 0])
        with st.sidebar:
            factor = st.slider("Sampling Frequency x0.1 Fs", 1,int(Fs) , 10)
            factor_noise = st.slider(
                'Signal to Noise Ratio (SNR)', min_value=0, max_value=100, value=0)
        # Compute the Nyquist frequency and the Nyquist rate
        Fmax = Fs / 2
        new_Fs = factor
        # Get the signal values and time stamps from the data
        signal = data[:, 1]
        time = data[:, 0]
        # Perform sampling using the Nyquist rate
        num_samples = int(new_Fs * (time[-1] - time[0]))
        new_time_sampled = np.linspace(time[0], time[-1], num_samples)
        signal_sampled = np.interp(new_time_sampled, time, signal)+ (factor_noise*1000) *np.random.randn(len(new_time_sampled))
        differenceSignal=signal-np.interp(time, new_time_sampled, signal_sampled)
        #save signal 
        with st.sidebar:
            if st.button("Save me"):
                fn.save_csv(signal_sampled)
 
        
    ##--  Calling the Plotting function  --##
        fn.plotGraphs(time,new_time_sampled,signal,signal_sampled)

if opt == "Create sinusoidal":

    #initialize variable
    signal=0
    time = np.linspace(0, 5*np.pi, 500)
    sin_list = []
      # Dictionary to store variables with IDs

    with st.sidebar:
        num = st.number_input('Number of sinusoidals',min_value=1, max_value=100 , value =1)
        signal_dict = {}

        for idx in range(num):
            sin_list.append(f'Sinusoidal ({idx+1})')
        freqs = [5]*num
        phase = [0]*num
        ambs  = [10]*num

        st.write('Sinusoidals Info:')
        df=pd.DataFrame({
                        'Freq':freqs,
                        'Mag':ambs,
                         'Phase': phase})
        edited_df = st.experimental_data_editor(df)  # 👈 An editable dataframe
        ambs = edited_df['Mag']
        freqs = edited_df['Freq']
        phase = edited_df['Phase']
        Fs = 2*np.max(freqs)
        with st.sidebar:
            # factor = st.slider("Sampling Frequency (0 : 10*Fmax)",
            #                    1, int(Fs)*5,value=int(Fs))
            
            factor = st.slider("Fs",1.0,4.0*float(np.max(freqs)),1.5*float(np.max(freqs)),1.0,format="%f")
            
            factor_noise = st.slider('Signal to Noise Ratio (SNR)', min_value=1, max_value=100, value=100)
            snr_db =  np.log10(( 100/factor_noise))
        
        for idx in range(num):
                signal += fn.generate_sin_with_phase(
                    2000, int(freqs[idx]), int(ambs[idx]), len(time), phase[idx])
        # Compute the Nyquist frequency and the Nyquist rate
        Fmax = Fs / 2
        new_Fs = factor*0.01

        # Perform sampling using the Nyquist rate
    
        num_samples =(new_Fs *0.5* (time[-1] - time[0]))
    
        if factor==2*freqs.min():        
            num_samples =(new_Fs *0.15* (time[-1] - time[0]))
        new_time_sampled = np.arange( time[0], time[-1],1/num_samples)
        noise=  np.random.randn(len(signal))*snr_db
        # signal_sampled = np.interp(new_time_sampled, time, signal)
        # Store the sampled signal in the signal_dict dictionary
        # signal_dict['signal_sampled'] = signal_sampled
        # signal_sampled = np.interp(new_time_sampled, time, signal  +noise)
        signal+=noise
        
        y_samples = fn.interpolate(new_time_sampled, time, signal )  #sampling/samples taken with input sampling frequency

        # amp of reconstructed 
        y_interpolated = fn.interpolate(time, new_time_sampled, y_samples)   # interploated signal or reconstructed signal
        # differenceSignal=signal-np.interp(time, new_time_sampled, signal_sampled)
        
        with st.sidebar:
             if st.button("Save"):
                fn.save_csv(signal_sampled)
                fn.save_csv(signal_dict['signal_sampled'])

    # Plot signals
    fn.plotGraphs(time,new_time_sampled,signal,y_samples,y_interpolated)


fn.markdown() #for style