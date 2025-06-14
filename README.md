# Material Redundancy Machine Learning Model

This repository contains a machine learning pipeline designed for predicting outcomes using data related to Material Redundancy operations. The model uses a CSV file as input, performs preprocessing, trains on various machine learning algorithms, and outputs predictions and performance metrics.

## 🔧 Features

- Automatically loads and preprocesses the data
- Trains multiple ML models using LazyPredict
- User selects the desired algorithm for evaluation
- Generates and saves structured reports
- Displays key metrics such as accuracy, R², and mean squared error

## 🗂️ File Structure

```bash
Material Redundancy/
├── MM-model.py          # Main script to run the model
├── sample.csv           # Sample dataset file (replaceable with your own)
└── README.md            # Project documentation
```

## 🚀 How to Run

1. Clone the Repository
```bash
git clone https://github.com/Kanha02052002/BHEL.git
cd BHEL
```
2. Install Requirements
```bash
pip install -r requirements.txt
```
 - If requirements.txt is not available, make sure to manually install the necessary packages:
```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn torch sentence_transformers nltk torch_geometric fuzzywuzzy
```
3. Add Your Dataset
Replace or modify <b>sample.csv</b> with your own dataset. Ensure it follows a similar format.
4. Run the Model
```bash
   python MM-model.py
```
5. Follow the Prompts
   - The script will show the dataset size and column names

   - You’ll be asked to select input features and the output column  

   - Set the train-test split ratio

   - Choose a model based on the displayed list

   - Final results and metrics will be printed and optionally saved
  
## 📊 Output

- Displays classification or regression performance
- Saves structured reports with details such as:
  - Initial dataset stats
  - Selected features and target
  - Model name and performance metrics


## 📝 Notes
- All warnings are suppressed for clean terminal output.
- You can enhance preprocessing or model evaluation in `MM-model.py`.

## 👤 Author

**Kanha Khantaal**

Feel free to fork, use, or suggest improvements!

## 📄 License

This project is licensed under the **MIT License**.
