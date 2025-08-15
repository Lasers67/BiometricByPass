# BiometricByPass

**Generative Model–Driven Presentation Attacks against ECG-based authentication systems**  
This project explores how **Generative Adversarial Networks (GANs)** can create counterfeit ECG signals to bypass biometric authentication systems.  
We focus on the **PTB-XL ECG dataset** and experiment with **1D-CNN**, **LSTM**, and **STFT-based** GAN architectures, combining **Conditional GAN (CGAN)** and **Wasserstein GAN (WGAN)** techniques.

---

## 📜 Project Summary

Biometric authentication, particularly **ECG-based biometrics**, offers enhanced security over passwords. However, advances in **deep learning** and **generative models** have introduced new vulnerabilities. This project demonstrates that:

- Robust ECG authenticators can be trained using limited leads.
- GANs can reconstruct target ECG leads from alternative leads.
- These reconstructions can successfully perform **presentation attacks**, reducing the reliability of ECG-based systems.

---

## 📥 Download Dataset

We use the **PTB-XL ECG dataset** from PhysioNet. Download it with:

```bash
wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/


Summary of Files:-
2DGAN.ipynb:- Jupyter notebook for Short Time Fourier Transform based GAN
augmentation.py:- Python file for data augmentation
CCGAN-WGAN.ipynb:- Jupyter notebook for running LSTM and 1D-CNN based CGAN-WGAN model.
data.py:- Python file for datapreprocessing from PTB-XL dataset.
main.ipynb:- Jupyter notebook for training ECG based authenticator.
models.py:- Helper python file for different types of models.
