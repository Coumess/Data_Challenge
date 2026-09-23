# Image Processing for Infrared Sensor Defect Correction: LYNRED Data Challenge

**Authors: Aurane Bourgault, Alexandre Coumes, Le Thanh Hien Nguyen, Eliott Poisson, Shivraj Sharode** \
**Supervisors: N. Vannier, Ronald Phlypo & M. Dohen** \
**Institution: INP - UGA, Phelma, Academic Year 2025-2026** \

This project addresses image quality issues found in certain infrared sensors (VGA, HD, and SXGA) caused by hardware amplifier faults. These faults manifest as noisy, blinking, or noisy-blinking defective pixels. Because replacing faulty hardware is expensive, our goal was to develop a software-based post-processing algorithm to automatically detect and correct these defects. To achieve this, we designed a modular, two-phase architecture: strict defect detection followed by targeted image correction.

## Methodology:
*   **Detection Phase:** We evaluated several signal processing methods to identify defective columns, including Local Thresholding (LT) and a 1D adaptation of Constant False Alarm Rate (CFAR). To pinpoint the exact vertical boundaries of fragmented defects, we implemented a Region Growing algorithm.
*   **Machine Learning Integration:** To push predictive performance further, we established a Random Forest baseline and ultimately optimized our detection using an XGBoost algorithm. By injecting custom spatial and temporal features, this model successfully synthesized the strengths of our signal-processing methods.
*   **Hyperparameter Optimization:** Because the different sensors feature drastically different resolutions and noise profiles, we utilized Bayesian Optimization (via the Optuna library) to dynamically tune parameters for specific sequences.
*   **Correction Phase:** Various techniques were tested to fix the identified defects without altering healthy pixels. These included 2D local smoothing, copy-paste methods, OpenCV Navier-Stokes inpainting, Non-Uniformity Correction (NUC), and deep learning models (ASCNet, DestripeCycleGAN).
*   **Frequency Interpolation:** This hybrid method proved to be the most robust for correction. It separates the image into low and high-frequency components, correcting only the defective parts of the low-frequency background via horizontal linear interpolation, and then fusing the high-frequency details back in.

## Results and Discussion:
*   Our final detection architecture operates in two sequential steps: first identifying the x-coordinates of anomalous columns (using LT or XGBoost), and then measuring the exact vertical bounds using Region Growing. 
*   Among the classical correction methods, Frequency Interpolation achieved the best result with a score of 94%, providing an excellent balance between stripe removal and detail preservation.
*   We observed an interesting discrepancy between strict mathematical metrics and human visual perception. While our XGBoost model introduced a marginal number of false positives (penalized by numerical scoring), the resulting images often looked cleaner and more natural to the naked eye.
*   Deep learning models (like ASCNet and DestripeCycleGAN) showed promise for image correction but struggled to generalize to our specific fragmented defect patterns without extensive retraining and larger datasets. 

# Read the full report (PDF)
This README is only an overview of our work. [Click here to read the full report](https://github.com/Coumess/Data_Challenge/blob/main/REPORT_DATA_CHALLENGE.pdf)
