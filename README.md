# Personalized Medicine Recommendation Using AI

A machine learning-based web application that recommends the most suitable medicine type based on a patient's medical condition and description.

> This project was completed as part of a **3-month internship** at **Vcodez**.

🔗 **Live Demo:** https://personalizedmedicineappproject-iayznvvrtjdlj3zmpausy6.streamlit.app/

---

## Description
This project is a machine learning-based system that recommends the most suitable drug type (Oral, Injectable, Topical, Liquid, or Other) based on a patient's medical condition and description. It uses Natural Language Processing (TF-IDF) and classification models to analyze text data and predict the appropriate medicine category. The final model is deployed as an interactive web application using Streamlit.

## Dataset

This project uses a medicine dataset containing 9,720 records with features such as Drug Name, Reason, and Description.

The original dataset source was not publicly available, so it has been uploaded to Kaggle for accessibility:

🔗 https://www.kaggle.com/datasets/subhasrii/medicine-drug-name-and-description-dataset

> Note: This dataset was used during the internship project and is shared here for educational and demonstration purposes only. It is published under CC0 (Public Domain).


## Technologies Used
- **Language:** Python
- **Environment:** Jupyter Notebook / Streamlit
- **Libraries:** pandas, numpy, scikit-learn, matplotlib, seaborn, streamlit, joblib
- **Techniques:** TF-IDF, Machine Learning Classification


## Installation
To run this project locally:

1. Clone the repository:
git clone https://github.com/SubhasriiT/Personalized_Medicine_App_Project.git
2. Navigate to the project folder:
cd Personalized_Medicine_App_Project
3. Install required libraries:
pip install pandas numpy scikit-learn matplotlib seaborn streamlit joblib
4. Run the Streamlit app:
streamlit run app.py


## Usage
1. Enter a medical condition or symptom description in the input box
2. Click on **"Predict Drug Type"**
3. The model predicts the most suitable drug category
4. View probability scores for all classes


## Project Report
Since the Jupyter Notebook does not render on GitHub, the full project report is available as a PDF in this repository.


## Contributing
Contributions are welcome. If you would like to improve this project:

1. Fork the repository
2. Create a new branch
3. Make your changes
4. Submit a pull request


## License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT) and is intended for educational purposes.


## Contact
For any queries or suggestions, feel free to reach out:

**Name:** Subhasri  
**Email:** ktsubhasri2005@gmail.com  
**GitHub:** [SubhasriiT](https://github.com/SubhasriiT)
