import IPython.display
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from pydub import AudioSegment
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
    """
    Graph a waveform from a .wav file.
    """
    data, sampling_rate = librosa.load(wav_file)
    plt.figure(figsize=(12, 4))
    librosa.display.waveplot(data, sr=sampling_rate)


def graph_waveform_from_data(data, sampling_rate):
    """
    Graph a waveform directly from data and sampling rate.
    """
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


def mix_audio(input_file, background_file, output_path, start_position=0, volume_adjustment=0):
    """
    A function for mixing 2 .wav files together, starting at start_position,
    adjusting volume of background_file by volume_adjustment (in decibels).
    It creates a new .wav file in the location specified by output_path.
    """
    # grab the two files
    sound1 = AudioSegment.from_wav(input_file)
    sound2 = AudioSegment.from_wav(background_file)

    # adjust volume if necessary
    sound2 = sound2 - volume_adjustment

    # mix sound2 with sound1, starting at start_position into sound1)
    output = sound1.overlay(sound2, position=start_position)

    # save the result
    output.export(output_path, format="wav")

    # return the path to the new .wav file
    return output_path


def augment_with_white_noise(input_file, output_path, wn_factor=0.01):
    """
    Take a .wav input_file, add random white noise to it and store
    the new .wav file in output_path. You can also adjust the amount of white noise
    via the wn_factor.
    """
    # grab the content of the .wav input_file and the sampling rate
    sr, data = get_wav_info(input_file)

    # add random white noise
    wn = np.random.randn(len(data))
    data_wn = data + wn_factor * wn

    # turn to integers
    data_wn = data_wn.astype(np.int16)

    # create the file with white noise
    wavfile.write(output_path, sr, data_wn)

    # return the path to the created file
    return output_path


def augment_with_shift(input_file, output_path, shift_factor=None):
    """
    Take a .wav input_file, shift its content by value of shift_factor.
    Save the new .wav file in output_path. You can also adjust the amount of shift
    via shift_factor.
    """
    # grab the content of the .wav input_file and the sampling rate
    sr, data = get_wav_info(input_file)

    # shift the data by 1/8th of the sampling rate, unless another value was provided
    if not shift_factor:
        shift_factor = sr // 8

    data_shifted = np.roll(data, shift_factor)

    # create the file with shifted data
    wavfile.write(output_path, sr, data_shifted)

    # return the path to the created file
    return output_path


def extract_mfccs(wav_file):
    """
    Take a file and return the mel-frequency cepstrum.
    Use the file's default sampling rate (instead of librosa's 22050Hz).
    """
    X, sample_rate = librosa.load(wav_file, res_type='kaiser_fast', sr=None)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    return mfccs