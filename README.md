# Finger-print-Analysis

# 🔍 Fingerprint Matching and Analysis using SIFT & OpenCV

A fingerprint recognition and matching system built using Python and OpenCV.  
The project uses SIFT (Scale-Invariant Feature Transform) and FLANN-based matching to compare altered fingerprints against real fingerprint datasets and identify the best match.

---

## 📌 Project Overview

This project performs fingerprint matching by:
- Extracting fingerprint features using SIFT
- Comparing fingerprints using FLANN matcher
- Finding the best matching fingerprint
- Displaying similarity scores and matched keypoints

The system is useful for:
- Biometric authentication
- Fingerprint verification
- Digital forensics
- Image feature matching research

---

## 🚀 Features

- ✅ Fingerprint image selection using GUI
- ✅ SIFT feature extraction
- ✅ FLANN-based feature matching
- ✅ Similarity score calculation
- ✅ Best fingerprint match detection
- ✅ Match visualization using OpenCV

---

## 🛠️ Technologies Used

### Language
- Python

### Libraries
- OpenCV
- Tkinter
- NumPy

---

## 📂 Project Structure

```bash
Fingerprint-Matching/
│
├── main.py
├── dataset/
│   ├── Real/
│   └── Altered/
├── README.md
└── requirements.txt

please download dataset from this link : https://www.kaggle.com/datasets/ruizgara/socofing/data
update the file path according to your system only
update the sample dictionary and finger print dictionary according to your system where the dataset is stored
