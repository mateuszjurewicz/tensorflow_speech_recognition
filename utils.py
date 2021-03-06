import bcolz
import glob
import IPython.display
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

from pydub import AudioSegment
from scipy.io import wavfile
from scipy.fftpack import fft


def get_wav_info(wav_file):
    """
    Read a wav file returning the sample rate (number of audio samples per second, in Hz) and the actual raw data.
    """
    # read the data 
    rate, data = wavfile.read(wav_file)

    # on linux systems we have to cast the data to float64 (as opposed to default int16)
    data = data.astype(np.float64)
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
    sampling_rate, data = get_wav_info(wav_file)
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


def augment_with_stretch(input_file, output_path, stretch_factor=1.2):
    """
    Take a .wav input_file, stretcht its content by the value of stretch_factor.
    Save the new .wav file in output_path. You can also adjust the amount of stretching
    via the stretch_factor. Otherwise a default value will be used.
    """
    # grab the content of the .wav input_file and the sampling rate
    sr, data = get_wav_info(input_file)

    # stretch the data by the stretch_factor
    data_stretched = librosa.effects.time_stretch(data, stretch_factor)

    # create the file with shifted data
    wavfile.write(output_path, sr, data_stretched)

    # return the path to the created file
    return output_path


def extract_mfccs(wav_file, columns=100, mean=True):
    """
    Take a file and return the mel-frequency cepstrum coefficients.
    Use the file's default sampling rate (instead of librosa's 22050Hz).
    """
    X, sample_rate = librosa.load(wav_file, res_type='kaiser_fast', sr=None)

    # extract MFCCs
    mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=columns)

    # if we were told to extract the mean and return a 1D vector per .wav...
    if mean:
        mfccs = np.mean(mfccs.T, axis=0)

    return mfccs


def extract_tempogram(wav_file):
    """
    Take a file and return the temopgram matrix. 
    """
    sr, raw_data = get_wav_info(wav_file)
    tempogram_data = librosa.feature.tempogram(raw_data, sr)

    return tempogram_data


def extract_fft(wav_file):
    """
    Take a file and return the fast fourier transform.
    """
    sr, raw_data = get_wav_info(wav_file)

    # normalize
    normalized_data = [(e/2**8)*2-1 for e in raw_data]

    # fast fourier transform
    fft_data = fft(normalized_data)

    return fft_data


def plot_fft(wavfile):
    """
    Take a .wav file, extract fast fourier transfrom and plot, taking into consideration signal symmetry.
    """
    fft_data = extract_fft(wavfile)
    symmetry_cutoff = len(fft_data)//2
    plt.plot(abs(fft_data[:symmetry_cutoff-1]), 'b')


def grab_wavs(path):
    """
    Take a path to a directory and return a list containing
    all .wav files within it.
    """

    # create regular expression
    glob_regex = os.path.join(path, "*.wav")

    # create a list of files
    wav_files = glob.glob(glob_regex)

    return wav_files


def one_hot_encode_path(path_to_wav, category_index, categories_in_order):
    """
    Take a path to a wave file and the index at which the category name starts.
    Uses the categories_in_order to figure out the index at which to put the 1.
    Return a numpy array, one hot encoded.
    """
    # take the slice of the path that should begin with the category name
    path_from_category = path_to_wav[category_index:]

    # create a placeholder array
    placeholder = np.zeros(len(categories_in_order))

    # check if we found a match
    is_matched = False

    for i, category in enumerate(categories_in_order):
        if path_from_category.startswith(category):
            placeholder[i] = 1
            is_matched = True
            return placeholder

    if not is_matched:
        raise Exception("one_hot_encode_path failed to find a category match in the path!")


def get_y(list_of_wavs, character_offset, categories):
    """
    Take a list of paths to .wav files, an integer specifying where the name of the category
    folder begins in the path string (.e.g "left" or "down") and a list of categories in order.

    Retrun a numpy ndarray with the one-hot-encoded target.
    """
    result = np.array([])

    # append each row to the result matrix
    for path_to_wav in list_of_wavs:
        row = one_hot_encode_path(path_to_wav, character_offset, categories)
        result = np.append(result, row)

    # this results in a flattened vector, so reshape it
    rows = len(list_of_wavs)
    columns = len(categories)
    result = np.reshape(result, (rows, columns))

    return result


def get_X(list_of_paths, column_num):
    """
    Take a list of paths to .wav files, and the desired number of output columns.
    Return a matrix with the raw extracted .wav data, padding the rows to the desired column number.
    """
    # get dimensions
    rows = len(list_of_paths)
    dimensions = (rows, column_num)

    # placeholder
    X = np.array([])

    # iterate
    for path_to_wav in list_of_paths:
        row = get_wav_info(path_to_wav)[1]

        # trim  & pad to desired dimensions
        row = row[:column_num]
        padding = column_num - len(row)
        row = np.pad(row, (0, padding), mode='constant', constant_values=0)

        # append
        X = np.append(X, row)

    # reshape (unroll)
    X = np.reshape(X, dimensions)

    return X


def get_X_mfccs(list_of_paths, shape=(100, 32), mean=True):
    """
    A version of get_X_with_padding that uses extract_mfccs instead of get_wav_info.
    Iterates over all file paths and extracts mfcc data from them, with a default shape.
    """
    # get shape data
    rows = len(list_of_paths)
    dimensions = (rows, shape[0])

    # create placeholder
    matrix = np.array([])

    # if we are supposed to output 1D-vector (mean=True)...
    if mean:

        # go through every file path in the list
        for path_to_wav in list_of_paths:

            # get raw array of signed ints (or matrix, if mean=False)
            row = extract_mfccs(path_to_wav, shape[0], mean)

            # some of our sample have less (or slightly more) than 16000 values, so let's adjust them
            # trim to fixed length
            row = row[:shape[0]]

            # pad with zeros, calculating amount of padding needed
            padding = shape[0] - len(row)
            row = np.pad(row, (0, padding), mode='constant', constant_values=0)

            # append the new row
            matrix = np.append(matrix, row)

        # reshape (unroll)
        matrix = np.reshape(matrix, dimensions)

        return matrix

    # else, we're returning a 2D matrix for every call to extract_mfccs()
    else:
        for path_to_wav in list_of_paths:

            # get the 2D matrix
            mfcc_matrix = extract_mfccs(path_to_wav, columns=shape[0], mean=mean)

            # trim and pad
            placeholder = np.array([])
            for row in mfcc_matrix:

                # trim first
                row = row[:shape[1]]

                # pad
                padding = shape[1] - len(row)
                row = np.pad(row, (0, padding), mode="constant", constant_values=0)

                # append to placeholder (flattened)
                placeholder = np.append(placeholder, row)

            # append the placeholder to the final matrix
            matrix =  np.append(matrix, placeholder)

        # reshape into a 3D matrix [constant 32]
        matrix = np.reshape(matrix, (len(list_of_paths), shape[0], shape[1]))

        return matrix


def get_X_mel_spectrogram(list_of_paths, shape=(128, 32)):
    """
    A version of get_X  that uses extract_mel_spectrogram instead of get_wav_info.
    Iterates over all file paths and extracts mfcc data from them, with default matrix shape
    of 128x32
    """
    # get shape data
    rows = len(list_of_paths)

    # create a placeholder for final result
    result = np.array([])

    # go through every file path in the list
    for path_to_wav in list_of_paths:

        # get raw array of signed ints
        sr, raw_data = get_wav_info(path_to_wav)
        mel_spectrogram = librosa.feature.melspectrogram(raw_data, sr)

        # trim and pad samples with more/less columns than expected
        placeholder = np.array([])
        for row in mel_spectrogram:

            # trim first
            row = row[:shape[1]]

            # pad with zeros
            padding = shape[1] - len(row)
            row = np.pad(row, (0, padding), mode="constant", constant_values=0)

            # append to placeholder (flattened)
            placeholder = np.append(placeholder, row)

        # append the placeholder to final matrix
        result = np.append(result, placeholder)

    # reshape into a 3-dim matrix
    result = np.reshape(result, (len(list_of_paths), shape[0], shape[1]))

    return result


def get_X_fft(list_of_paths, columns=16000):
    """
    A version of get_X that uses extract_fft instead of get_wav_info.
    Iterates over all file paths and extracts fft data from them, with default column
    number equal to 16000.
    """
    # get shape data
    rows = len(list_of_paths)
    dimensions = (rows, columns)

    # create placeholder
    matrix = np.array([])

    # go through every file path in the list
    for path_to_wav in list_of_paths:

        # get FFT data
        row = extract_fft(path_to_wav)

        # some of our sample have less (or slightly more) than 16000 values, so let's adjust them
        # trim to fixed length
        row = row[:columns]

        # pad with zeros, calculating amount of padding needed
        padding = columns - len(row)
        row = np.pad(row, (0, padding), mode='constant', constant_values=0)

        # append the new row
        matrix = np.append(matrix, row)

    # reshape (unroll)
    matrix = np.reshape(matrix, dimensions)

    # cast to float64 to be able to persist via bcolz
    matrix = matrix.astype(np.float64)

    return matrix


def get_X_tempogram(list_of_paths, shape=(384, 32)):
    """
    A version of get_X  that uses extract_tempogram instead of get_wav_info.
    Iterates over all file paths and extracts tepogram data from them, with default matrix shape
    of 384x32.
    """
    # get shape data
    rows = len(list_of_paths)

    # create a placeholder for final result
    result = np.array([])

    # go through every file path in the list
    for path_to_wav in list_of_paths:

        # get raw array of signed ints
        tempogram = extract_tempogram(path_to_wav)

        # trim and pad samples with more/less columns than expected
        placeholder = np.array([])
        for row in tempogram:

            # trim first
            row = row[:shape[1]]

            # pad with zeros
            padding = shape[1] - len(row)
            row = np.pad(row, (0, padding), mode="constant", constant_values=0)

            # append to placeholder (flattened)
            placeholder = np.append(placeholder, row)

        # append the placeholder to final matrix
        result = np.append(result, placeholder)

    # reshape into a 3-dim matrix
    result = np.reshape(result, (len(list_of_paths), shape[0], shape[1]))

    return result


def one_hot_encode(a_matrix):
    """
    Takes a matrix of presumably softmaxed predictions and turns it into
    a one-hot-encoded matrix, to be passed to e.g. F1 score calculating function,
    """

    # store results in a new matrix
    original_shape = a_matrix.shape
    one_hot_encoded_matrix = np.zeros(original_shape)

    # obtain the index of the highest value in a row
    for i, row in enumerate(a_matrix):
        max_row_value = 0
        max_row_value_index = 0

        for j, elem in enumerate(row):

            # if the current element has a higher value then any previously found
            if elem > max_row_value:
                # update highest value found for this row
                max_row_value = elem
                # update the highest value index
                max_row_value_index = j

        # once we iterated over all elements in a row, we can alter our end-result matrix
        one_hot_encoded_matrix[i][max_row_value_index] = 1

    return one_hot_encoded_matrix


def reverse_one_hot_encoding(a_matrix):
    """
    Take a one-hot encoded matrix and return a vector with the row value
    representing the original column index of the 1, per row in original matrix.
    """

    # get shape
    result_vector = np.zeros((a_matrix.shape[0], 1))

    # fill
    for i, row in enumerate(a_matrix):
        for j, elem in enumerate(row):
            if elem == 1:
                # to avoid class 0, add 1
                result_vector[i] = j + 1

    return result_vector


# define the bcolz array saving function
def bcolz_save(fname, arr):
    c = bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()


# define the bcolz array loading function
def bcolz_load(fname):
    return bcolz.open(fname)[:]
