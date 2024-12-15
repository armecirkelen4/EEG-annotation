import os
import mne
import numpy as np

# Step 1: Set the directory and load the files
dir_disk = 'H:/EEG_Roskilde/Samlet'
os.chdir(dir_disk)
files = [f for f in os.listdir() if f.endswith('.mat')]

# Save the list of files
np.save('files_names2.npy', files)

# Step 2: Processing files
for ii in range(60, 61):  # Adjust this to 139 if you need the full range
    eeglab_startup()  # Placeholder for EEGLAB equivalent
    
    loaded_file = files[ii]
    find_file = loaded_file[:-4] + '_p00.set'

    if os.path.isfile(find_file):
        # Load the file (adapted to MNE-Python)
        EEG = mne.io.read_raw_mat(loaded_file)
        
        # Set EEG object properties
        EEG.setname = 'NFC_' + loaded_file[:-4]
        EEG.filepath = 'H:/EEG_Roskilde'
        EEG.filename = loaded_file
        EEG.srate = EEG.info['sfreq']
        EEG.pnts = EEG.n_times
        EEG.trials = 1
        EEG.event = []  # Add events if needed
        EEG.xmin = 0
        EEG.xmax = EEG.n_times / EEG.info['sfreq']
        EEG.icasphere = 'No'
        EEG.icaweights = None
        EEG.icawinv = None
        EEG.nbchan = len(EEG.info['ch_names'])
        EEG.icaact = []

        ALLEEG = EEG
        CURRENTSET = 1

        # Resampling to 200 Hz
        EEG_resampled = EEG.copy().resample(200)

        # Apply bandpass filtering and notch filtering
        EEG_filtered = EEG_resampled.copy().filter(l_freq=1, h_freq=70)
        EEG_filtered = EEG_filtered.notch_filter(freqs=np.arange(45, 55, 1))

        # Save the filtered file
        file_save_name0 = loaded_file[:-4] + '_p00_epoched'
        EEG_filtered.save(file_save_name0 + '.fif', overwrite=True)

        # Load the saved file
        EEG_loaded = mne.io.read_raw_fif(file_save_name0 + '.fif')

        # Remove channels if necessary (if EEG has 25 channels)
        if EEG_loaded.info['nchan'] == 25:
            EEG_loaded.drop_channels(EEG_loaded.info['ch_names'][19:])

        # Create events for 1-second epochs
        events = mne.make_fixed_length_events(EEG_loaded, start=0, stop=EEG_loaded.times[-1], duration=1)

        # Epoch the data
        epochs = mne.Epochs(EEG_loaded, events, tmin=0, tmax=1, baseline=None)

        # Save the epoched data
        file_save_name1 = loaded_file[:-4] + '_p01_epoched'
        epochs.save(file_save_name1 + '-epo.fif', overwrite=True)

        # Mark all epochs for rejection
        epochs.drop_bad()

        # Save the final result
        file_save_name2 = loaded_file[:-4] + '_p02_epoched'
        epochs.save(file_save_name2 + '-epo.fif', overwrite=True)

# Step 3: Mark the first 60 epochs
for ii in range(len(files)):
    EEG = mne.read_epochs(files[ii])
    rejmanual = EEG.drop_log
    zero_indices = [i for i, log in enumerate(rejmanual) if not log]
    if len(zero_indices) > 60:
        for i in zero_indices[60:]:
            rejmanual[i] = ['Manual rejection']
    else:
        print('ERROR!! MISSING EPOCHS')

    EEG.drop_log = rejmanual
    name_save_60epochs = files[ii][:-4] + '_60EpochsMarked'
    EEG.save(name_save_60epochs + '-epo.fif', overwrite=True)

# Step 4: Process eyes-open epochs
for ii in range(50, 60):
    name_EO = files[ii][:-19] + '_EyesOpen_marked'
    EEG = mne.read_epochs(files[ii])

    # Mark all epochs as rejected
    EEG.drop_log = [['Manual rejection'] for _ in range(len(EEG.events))]

    # Plot the EEG data (optional, for manual inspection)
    EEG.plot()

    # Save the marked epochs
    EEG.save(name_EO + '-epo.fif', overwrite=True)

# Step 5: Count the number of epochs marked as Eyes Open (non-rejected)
count_epochs_EO = []
for ii in range(len(files)):
    EEG = mne.read_epochs(files[ii])
    rejmanual = EEG.drop_log
    count_epochs_EO.append(len([log for log in rejmanual if not log]))

# Step 6: Optional processing steps, like power calculations, ICA, etc.
# The following sections can be expanded based on your needs for calculating coherence, spectral power, etc.

# Step 7: Power Spectral Density (PSD) and Relative Power Calculations
for ii in range(len(files)):
    EEG = mne.read_epochs(files[ii])
    
    # Define the frequency ranges
    delta_range = [1, 4]
    theta_range = [4, 8]
    alpha_range = [8, 13]
    beta_range = [13, 30]
    
    num_channels = len(EEG.ch_names)
    num_epochs = len(EEG)
    
    # Initialize arrays to store power values
    delta_power = np.zeros((num_channels, num_epochs))
    theta_power = np.zeros((num_channels, num_epochs))
    alpha_power = np.zeros((num_channels, num_epochs))
    beta_power = np.zeros((num_channels, num_epochs))
    
    relative_delta_power = np.zeros((num_channels, num_epochs))
    relative_theta_power = np.zeros((num_channels, num_epochs))
    relative_alpha_power = np.zeros((num_channels, num_epochs))
    relative_beta_power = np.zeros((num_channels, num_epochs))

    # Loop through each channel and epoch
    for chan_idx in range(num_channels):
        for epoch_idx in range(num_epochs):
            epoch_data = EEG.get_data(picks=chan_idx)[epoch_idx]
            
            # Compute the Power Spectral Density (PSD) using MNE's Welch method
            psd, freqs = mne.time_frequency.psd_array_welch(epoch_data, sfreq=EEG.info['sfreq'], fmin=1, fmax=30)
            
            # Get indices for different frequency bands
            delta_idx = np.where((freqs >= delta_range[0]) & (freqs <= delta_range[1]))[0]
            theta_idx = np.where((freqs >= theta_range[0]) & (freqs <= theta_range[1]))[0]
            alpha_idx = np.where((freqs >= alpha_range[0]) & (freqs <= alpha_range[1]))[0]
            beta_idx = np.where((freqs >= beta_range[0]) & (freqs <= beta_range[1]))[0]

            # Compute power by summing PSD over the frequency range
            delta_power[chan_idx, epoch_idx] = np.sum(psd[delta_idx])
            theta_power[chan_idx, epoch_idx] = np.sum(psd[theta_idx])
            alpha_power[chan_idx, epoch_idx] = np.sum(psd[alpha_idx])
            beta_power[chan_idx, epoch_idx] = np.sum(psd[beta_idx])

            # Total power (across all bands)
            total_power = np.sum(psd)

            # Compute relative power
            relative_delta_power[chan_idx, epoch_idx] = delta_power[chan_idx, epoch_idx] / total_power
            relative_theta_power[chan_idx, epoch_idx] = theta_power[chan_idx, epoch_idx] / total_power
            relative_alpha_power[chan_idx, epoch_idx] = alpha_power[chan_idx, epoch_idx] / total_power
            relative_beta_power[chan_idx, epoch_idx] = beta_power[chan_idx, epoch_idx] / total_power

    # Save PSD and relative power data
    file_psd_name = files[ii][:-4] + '_psd'
    np.savez(file_psd_name, 
             delta_power=delta_power, theta_power=theta_power, alpha_power=alpha_power, beta_power=beta_power,
             relative_delta_power=relative_delta_power, relative_theta_power=relative_theta_power, 
             relative_alpha_power=relative_alpha_power, relative_beta_power=relative_beta_power)
# Step 8: ICA Component Removal
for ii in range(len(files)):
    EEG = mne.read_epochs(files[ii])
    
    # Perform ICA to remove artifacts (such as eye blinks)
    ica = mne.preprocessing.ICA(n_components=15, random_state=97, max_iter='auto')
    ica.fit(EEG)
    
    # Automatically label ICA components using ICLabel
    ica_labels = mne.preprocessing.ica.ICALabeler(ica)
    ica_labels.label_components(EEG)
    
    # Flag components for removal (e.g., muscle artifacts, eye blinks)
    components_to_remove = ica_labels.get_components_to_remove(EOG=True, EMG=True)
    
    # Remove flagged components from the data
    EEG_clean = ica.apply(EEG, exclude=components_to_remove)
    
    # Save the cleaned data
    file_ica_cleaned = files[ii][:-4] + '_p05'
    EEG_clean.save(file_ica_cleaned + '-epo.fif', overwrite=True)

    # Save the number of removed components
    num_removed_components = len(components_to_remove)
    np.save(files[ii][:-4] + '_removed_components.npy', num_removed_components)
# Step 9: Coherence Calculation
for ii in range(len(files)):
    EEG = mne.read_epochs(files[ii])
    
    # Define frequency bands
    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30)
    }

    # Initialize arrays to store coherence values for each band
    coherence_values = {band: np.zeros((EEG.info['nchan'], EEG.info['nchan'])) for band in bands}
    
    # Loop through each frequency band
    for band_name, (fmin, fmax) in bands.items():
        # Compute coherence between all channel pairs
        fmin, fmax = bands[band_name]
        con, freqs, times, n_epochs, n_tapers = mne.connectivity.spectral_connectivity(
            EEG, method='coh', mode='multitaper', sfreq=EEG.info['sfreq'], 
            fmin=fmin, fmax=fmax, faverage=True, verbose=False
        )
        coherence_values[band_name] = con

    # Save the coherence values
    file_coherence_name = files[ii][:-4] + '_coherence'
    np.savez(file_coherence_name, **coherence_values)
    
# Step 10: Segment EEG Data Based on Rejection Marks
for ii in range(len(files)):
    EEG = mne.read_epochs(files[ii])

    # Get the rejection marks
    rejmanual = EEG.drop_log
    valid_epochs = [i for i, log in enumerate(rejmanual) if not log]

    # Segment the EEG data based on rejection marks
    for idx, valid_epoch in enumerate(valid_epochs):
        segment = EEG[valid_epoch]
        file_segment_name = files[ii][:-4] + f'_seg_{idx}'
        segment.save(file_segment_name + '-epo.fif', overwrite=True)

    # Save the number of valid epochs
    np.save(files[ii][:-4] + '_valid_epochs.npy', valid_epochs)
