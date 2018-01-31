import IPython.display
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from scipy.io import wavfile


def get_wav_info(wav_file):
    """
    Read a wav file returning the sample rate (number of audio samples per second, in Hz) and the actual raw data.
    """
    rate, data = wavfile.read(wav_file)
    return rate, data


def display_audio(path_to_wav):
    """
    Simple function for displaying playable audio files in jupyter.
    """
    # get rate and data
    r, d = get_wav_info(path_to_wav)
    # show playable audio
    return IPython.display.Audio(data=d, rate=r)


def graph_waveform(wav_file):
    data, sampling_rate = librosa.load(wav_file)
    plt.figure(figsize=(12, 4))
    librosa.display.waveplot(data, sr=sampling_rate)


def graph_multiple_waveforms(list_of_wavs, grid_dimensions=(3, 3)):
    """
    Draw multiple graphs of waveforms side by side.
    """

    # Assume 3x3 grid maximum
    grid_height = grid_dimensions[0]
    grid_width = grid_dimensions[1]

    # Create a figure for all subplots
    plt.figure(figsize=(24, 18))

    for idx, wav_file in enumerate(list_of_wavs):
        data, sampling_rate = librosa.load(wav_file)

        # Subplot
        plt.subplot(grid_height, grid_width, idx + 1)

        # Display the waveform
        librosa.display.waveplot(data, sr=sampling_rate)

        # Describe
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.title('{}'.format(wav_file))


def graph_spectrogram(wav_file, size=(10, 6)):
    """
    Draw a graph of a spectrogram using pyplot.
    """
    rate, data = get_wav_info(wav_file)
    nfft = 256  # windowing segments length
    fs = 16000  # sampling frequency of our files is 16000
    plt.figure(figsize=size)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('Spectrogram of {}'.format(wav_file))
    pxx, freqs, bins, im = plt.specgram(data, nfft, fs)


def graph_multiple_spectrograms(list_of_wavs, grid_dimensions=(3, 3)):
    """
    Draw multiple graphs of spectrograms side by side.
    """

    # Assume 3x3 grid maximum
    grid_height = grid_dimensions[0]
    grid_width = grid_dimensions[1]

    nfft = 256  # windowing segments length
    fs = 16000  # sampling frequency of our files is 16000

    # Create a figure for all subplots
    plt.figure(figsize=(24, 18))

    for idx, wav_file in enumerate(list_of_wavs):
        rate, data = get_wav_info(wav_file)

        # Subplot
        plt.subplot(grid_height, grid_width, idx + 1)

        # Display the spectrogram
        pxx, freqs, bins, im = plt.specgram(data, nfft, fs)

        # Describe
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.title('{}'.format(wav_file))


def graph_mel_spectrogram(wav_file):
    """
    Draw a graph of a mel spectrogram using librosa.
    """
    y, sr = librosa.load(wav_file)
    S = librosa.feature.melspectrogram(y, sr=sr)

    # Convert to log scale (dB). We'll use the peak power (max) as reference.
    log_S = librosa.power_to_db(S, ref=np.max)

    # Make a new figure
    plt.figure(figsize=(10, 6))

    # Display the spectrogram on a mel scale
    # sample rate and hop length parameters are used to render the time axis
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max),
                             y_axis='mel', fmax=32000,
                             x_axis='time')

    # Descriptions
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('Mel spectrogram of {}'.format(wav_file))

    # draw a color bar
    plt.colorbar(format='%+02.0f dB')

    # Make the figure layout compact
    plt.tight_layout()


def graph_multiple_mel_spectrograms(list_of_wavs, grid_dimensions=(3, 3)):
    """
    Draw multiple graphs of mel spectrograms side by side.
    """

    # Assume 3x3 grid maximum
    grid_height = grid_dimensions[0]
    grid_width = grid_dimensions[1]

    # Create a figure for all subplots
    plt.figure(figsize=(24, 18))

    for idx, wav_file in enumerate(list_of_wavs):
        y, sr = librosa.load(wav_file)
        S = librosa.feature.melspectrogram(y, sr=sr)
        log_S = librosa.power_to_db(S, ref=np.max)

        # Subplot
        plt.subplot(grid_height, grid_width, idx + 1)

        # Display the spectrogram on a mel scale
        librosa.display.specshow(librosa.power_to_db(S, ref=np.max),
                                 y_axis='mel', fmax=32000,
                                 x_axis='time')

        # Describe
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.title('{}'.format(wav_file))


def graph_chromagrams(list_of_wavs, grid_dimensions=(3, 3)):
    """
    Draw multiple chromagrams side by side.
    """

    # Assume 3x3 grid maximum
    grid_height = grid_dimensions[0]
    grid_width = grid_dimensions[1]

    # Create a figure for all subplots
    plt.figure(figsize=(24, 18))

    for idx, wav_file in enumerate(list_of_wavs):
        y, sr = librosa.load(wav_file)
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        C = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)

        # Subplot
        plt.subplot(grid_height, grid_width, idx + 1)

        # Display the spectrogram on a mel scale
        librosa.display.specshow(C, sr=sr, x_axis='time', y_axis='chroma', vmin=0, vmax=1)

        # Describe
        plt.title('{}'.format(wav_file))
