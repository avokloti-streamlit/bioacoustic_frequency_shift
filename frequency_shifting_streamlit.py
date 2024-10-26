import streamlit as st
import soundfile as sf
import librosa
import scipy
import io
import zipfile
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

    st.markdown('Please select the files you would like to be analyzed below. This app will then apply a frequency shift to all audio samples,\
                resample to the specified sampling rate, and create a ZIP archive that can bedownloaded to your computer.')
    
    with st.form('Analyze Data'):
        uploaded_files = st.file_uploader("Choose audio files:", accept_multiple_files=True)
        output_sr = st.number_input('Sampling Rate:', min_value = 2 * frequency_upper, max_value = sr, value=sr, step=100)
        submitted = st.form_submit_button("Analyze Data")

    if submitted and uploaded_files is not None:
        st.markdown('Uploaded %d files...' % len(uploaded_files))

        progress_bar = st.progress(0, text='Analyzing...')

        buffers = []

        for ind, file in enumerate(uploaded_files):
            signal, sr = sf.read(file)

            signal = butter_bandpass_filter(signal, frequency_lower, frequency_upper, sr, order=5)

            n = np.arange(len(signal))
            shifting_wave = np.exp(-1j * 2 * np.pi * (-frequency_lower) * n / sr)
            signal_shifted = np.real(signal * shifting_wave)

            signal_shifted = butter_lowpass_filter(signal_shifted, frequency_upper - frequency_lower, sr, order=5)

            signal_resampled = librosa.resample(signal_shifted, orig_sr=sr, target_sr=output_sr)

            buffer = io.BytesIO()
            sf.write(buffer, signal_resampled, output_sr, format='wav')
            buffers.append(buffer)
            #sf.write(file[:-4] + '_shifted_down_by_%d.wav' % frequency_lower, signal_resampled, output_sr)

            progress_bar.progress((ind + 1)/len(uploaded_files))
        
        progress_bar.empty()
        st.markdown('Analysis complete!')

        # Create an in-memory zip file
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zip_file:
            for i, buffer in enumerate(buffers):
                # Write each buffer to the zip file
                new_filename = uploaded_files[i].name[:-4] + '_shifted_down_by_%d.wav' % frequency_lower
                zip_file.writestr(new_filename, buffer.getvalue())

        # Get the bytes of the compressed zip file
        zip_bytes = zip_buffer.getvalue()

        st.download_button('Download ZIP archive of frequency-shifted mp3 files!', data = zip_bytes,
                           file_name = 'audio_shifted_by_%d_at_sampling_rate_%d.zip' % (frequency_lower, output_sr))
