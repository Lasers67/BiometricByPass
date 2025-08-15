import torch
import numpy as np
from scipy.interpolate import interp1d

def time_wrap(ecg_data):
    T, C = ecg_data.shape
    warp_factor = np.random.uniform(0.8, 1.2)  # Randomly stretch or compress

    # Create original time indices and new (warped) time indices
    original_time = np.linspace(0, 1, T)
    new_time = np.linspace(0, 1, int(T * warp_factor))

    warped = []
    for c in range(C):
        interp = interp1d(original_time, ecg_data[:, c], kind='linear', fill_value="extrapolate")
        warped_channel = interp(new_time)
        # Resize back to original length to maintain consistent shape
        if len(warped_channel) > T:
            warped_channel = warped_channel[:T]
        else:
            warped_channel = np.pad(warped_channel, (0, T - len(warped_channel)), mode='edge')
        warped.append(warped_channel)

    return np.stack(warped, axis=1)


def time_shifting(ecg_data):
    w = np.random.uniform(0.15, 0.45)
    if not (0 <= w <= 1):
        raise ValueError("w must be between 0 and 1.")
    T = ecg_data.shape[0]
    shift_length = int(w * T)
    # Perform a circular shift (rotation)
    ecg_data = np.roll(ecg_data, shift=shift_length, axis=0)
    return ecg_data
def add_noise(ecg_data):
    noise = np.random.normal(0, 0.005, ecg_data.shape)
    ecg_data += noise
    return ecg_data

def time_masking(ecg_data):
    w = np.random.uniform(0.05, 0.2)
    if not (0 <= w <= 1):
        raise ValueError("w must be between 0 and 1.")
    T = ecg_data.shape[0]
    mask_length = int(w * T)
    ts = np.random.randint(0, T - mask_length + 1)
    ecg_data[ts:ts + mask_length, :] = 0
    return ecg_data

def augment(X,Y=[],num_masks=5):
    factor = len(X)//len(Y) if len(Y) > 0 else 1
    augmented_X = []
    augmented_Y = []
    for i in range(len(X)):
        augmented_X.append(X[i].clone())
        if len(Y) > 0:
            augmented_Y.append(Y[i//factor])
        for _ in range(num_masks):
            ecg_data = X[i].clone()
            #augmented_data = add_noise(time_shifting(ecg_data.numpy()))  # Apply time masking to numpy array
            masked_data = add_noise(time_masking(ecg_data.numpy().copy()))
            time_wraped = add_noise(time_wrap(ecg_data.numpy().copy()))
            #augmented_X.append(torch.from_numpy(augmented_data))  # Convert back to torch tensor
            augmented_X.append(torch.from_numpy(masked_data))  # Convert back to torch tensor
            augmented_X.append(torch.from_numpy(time_wraped))
            # Append the corresponding label if Y is provided
            if len(Y) > 0:
                augmented_Y.append(Y[i//factor])
                augmented_Y.append(Y[i//factor])
    # Convert augmented lists back to tensors
    augmented_X = torch.stack(augmented_X)
    if len(Y) > 0:
        augmented_Y = torch.stack(augmented_Y)
        return augmented_X, augmented_Y
    return augmented_X