# 🎨 AI-Powered Seasonal Color Analysis

## 📌 Project Goal

The primary objective of this project was to develop a specialized Artificial Intelligence (AI) model capable of automatically determining a person's **Seasonal Color Type** based on the analysis of a digital portrait.

Essentially, the system was designed to mimic the expert human process of personal color analysis. The AI learns to detect and isolate the natural colors of a subject's skin, hair, and eyes from a standard photograph. By synthesizing these color characteristics, the model categorizes the individual into one of the traditional seasonal distinct types (Spring, Summer, Autumn, Winter), thereby recommending the most flattering clothing, makeup, and accessory colors.

## ✨ Key Features & Workflow

The project follows a structured Computer Vision and Machine Learning pipeline:

1.  **Face & Feature Detection:** Automated detection of faces within digital portraits.
2.  **Facial Feature Segmentation:** Precise isolation (segmentation) of the **skin**, **hair**, and **eyes** areas.
3.  **Color Extraction:** Analysis and measurement of dominant and representative color values from the segmented regions.
4.  **Color Space Synthesis:** Combining these values to create a complete "color profile" of the individual.
5.  **Seasonal Classification:** A trained Machine Learning model predicts the seasonal type by matching the synthesized color profile to established color theory rules.

## 🛠️ Tech Stack (Example/To be Filled)

*   **Programming Language:** [e.g., Python]
*   **Computer Vision Libraries:** [e.g., OpenCV, MediaPipe, dlib]
*   **Machine Learning/Deep Learning Frameworks:** [e.g. PyTorch, scikit-learn]
*   **Image Processing:** [e.g., NumPy, Pillow (PIL)]
*   **Deployment/Tools:** [e.g., Google Colab, Jupyter Notebooks]

## 🚀 Future Enhancements

*   Integration of a complete 12 or 16 Seasonal Flow System (e.g., Light Spring vs. Clear Spring).
*   Real-time video analysis for on-the-go color typing.
*   Virtual try-on feature integrated with the recommended color palette.

---
### 👤 About the Authors

This project was developed by:

*   Katarina Petrović
*   Nataša Radmilović
