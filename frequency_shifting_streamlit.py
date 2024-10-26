import streamlit as st
import soundfile as sf
import librosa
import scipy
import os
import numpy as np
import matplotlib.pyplot as plt

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    cutoff = 2 * np.array([lowcut, highcut]) / fs
    b, a = scipy.signal.butter(order, cutoff, btype='band', analog=False)
    y = scipy.signal.filtfilt(b, a, data)
    return y

def butter_highpass_filter(data, cut, fs, order=5):
    cutoff = 2 * cut / fs
    b, a = scipy.signal.butter(order, cutoff, btype='high', analog=False)
    y = scipy.signal.filtfilt(b, a, data)
    return y

def butter_lowpass_filter(data, cut, fs, order=5):
    cutoff = 2 * cut / fs
    b, a = scipy.signal.butter(order, cutoff, btype='low', analog=False)
    y = scipy.signal.filtfilt(b, a, data)
    return y

def calculateAndPlotSpectrogram(signal, sr, title, hlines=None, vlim=None):
    # calculate spectrogram
    f, t, Z = scipy.signal.stft(signal, fs=sr)
    Zabs = np.log10(np.abs(Z))

    # plot spectrogram
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    if vlim is not None:
        pcol = ax.pcolormesh(t, f, Zabs, vmin=vlim[0], vmax=vlim[1])
    else:
        pcol = ax.pcolormesh(t, f, Zabs)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(title)
    if hlines is not None:
        for h in hlines:
            ax.hlines(h, 0, len(signal)/sr, 'r', linestyle='--')
    
    cbar = fig.colorbar(pcol)
    return fig, [Zabs.min(), Zabs.max()]



st.set_page_config(layout="centered", page_title="Frequency-Shifting App", page_icon="🐢")

st.title("Frequency-Shifting App 🐢 🦇 🐬")

st.markdown('This application performs frequency demodulation on audio samples. First, please select the \
        frequency range you need for your data, and preview the results on a 10-second \
        audio sample. Then, select the folder containing your data, and run this \
        calculation across your dataset.')


# --- load, plot, and listen to the original sample --- #

st.markdown('### Preview for a sample of your dataset:')

# get audio from user
uploaded_file = st.file_uploader("Choose an audio file:", accept_multiple_files=False)

if uploaded_file is not None:
    # load in audio sample
    signal, sr = sf.read(uploaded_file)
    signal = signal[0:(10 * sr)]

    # listen to it
    st.markdown('Here is the first 10 seconds of this audio sample:')
    st.audio(signal, sample_rate=sr)

    # choose frequencies
    st.markdown('Please select the frequency range you would like to keep. The audio will be shifted \
                such that the minimum frequency will become 0 Hz, and all frequencies outside of this range will be filtered.')
    frequency_lower, frequency_upper = st.slider("Select Frequency Range:", 0, sr//2, value=(sr//8, sr//4), step=1000)

    # calculate spectrogram
    fig, clim = calculateAndPlotSpectrogram(signal, sr, 'Spectrogram of Original Audio Sample', hlines=[frequency_lower, frequency_upper])
    st.pyplot(fig)

    # bandpass filter between those frequencies
    signal = butter_bandpass_filter(signal, frequency_lower, frequency_upper, sr, order=5)

    # calculate again
    fig_filtered, _ = calculateAndPlotSpectrogram(signal, sr, 'Spectrogram of Audio Sample After Bandpass Filter',
                                               hlines=[frequency_lower, frequency_upper], vlim=clim)
    
    st.pyplot(fig_filtered)

    # now try also frequency modulation
    n = np.arange(len(signal))
    shifting_wave = np.exp(-1j * 2 * np.pi * (-frequency_lower) * n / sr)
    demod = np.real(signal * shifting_wave)

    # lowpass filter
    demod = butter_lowpass_filter(demod, frequency_upper - frequency_lower, sr, order=5)

    # calculate again
    fig_shifted, _ = calculateAndPlotSpectrogram(demod, sr, 'Spectrogram of Frequency-Shifted Audio Sample',
                                              hlines=[0, frequency_upper - frequency_lower], vlim=clim)
    st.pyplot(fig_shifted)

    # the final resulting audio
    st.markdown('Here is the resulting frequency-shifted audio sample:')
    st.audio(demod, sample_rate=sr)


    ## --- Next section: running in bulk on a dataset! --- ##

    st.markdown('### Perform frequency-shifting on your dataset:')

    st.markdown('Below, please enter the full paths to the folder containing your dataset, and to the folder \
                where you want the frequency-shifted files to be written. For example, the former might be:')
    
    st.text('/Users/user/Documents/2024/Data/')

    st.markdown('And maybe the latter is:')

    st.text('/Users/user/Documents/2024/ShiftedData/')

    st.markdown('This app will then \
                iterate through all files, apply the same frequency-shift as shown above, and write a new file \
                to the output folder, with the specified sampling rate. Please check that the path is correct -- this is a common source of errors! ')

    # get various input from the user
    with st.form('Analyze Data'):
        data_folder = st.text_input("Path to the folder containing your dataset:")
        output_folder = st.text_input("Path to the folder for writing frequency-shifted audio:")
        output_sr = st.number_input('Sampling Rate:', min_value = 2 * frequency_upper, max_value = sr, value=sr, step=100)
        submitted = st.form_submit_button("Analyze Data")

    if submitted:
        files = os.listdir(data_folder)
        files = [f for f in files if f[-4:] == '.wav' or f[-4:] == '.mp3' or f[-4:] == '.aif']

        st.markdown('Detected %d files...' % len(files))
        st.text(files)

        progress_bar = st.progress(0, text='Analyzing...')

        for ind, file in enumerate(files):
            signal, sr = sf.read(data_folder + file)

            signal = butter_bandpass_filter(signal, frequency_lower, frequency_upper, sr, order=5)

            n = np.arange(len(signal))
            shifting_wave = np.exp(-1j * 2 * np.pi * (-frequency_lower) * n / sr)
            signal_shifted = np.real(signal * shifting_wave)

            signal_shifted = butter_lowpass_filter(signal_shifted, frequency_upper - frequency_lower, sr, order=5)

            signal_resampled = librosa.resample(signal_shifted, orig_sr=sr, target_sr=output_sr)

            sf.write(output_folder + file[:-4] + '_shifted_down_by_%d.wav' % frequency_lower, signal_resampled, output_sr)

            progress_bar.progress((ind + 1)/len(files))
        
        progress_bar.empty()
        st.markdown('Analysis complete!')