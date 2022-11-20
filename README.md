# HousingPricePredictor
This model was built for regression problems. Specifically made to predict housing price looking at data like its coordinates, distance from certain commodities and a couple other factors. The dataset can be found and downloaded here: https://archive.ics.uci.edu/ml/datasets/Real+estate+valuation+data+set. In the code you just have to download this data set and specify the path to it on your local computer or in your venv depending on what your setup is like. 

Look for the read_csv line right under imports that looks like this:
df_raw = pd.read_csv(r'C:\Users\wizar\PycharmProjects\HousingRegression\Real estate valuation data set.csv')

Paste the path to your dataset file in the single quotation. Note *You might need to open the file after downloading it in excel and save it as a .csv file* 

As for how to run the code you need to download a few libraries which I will list.
1) torch
2) numpy
3) pandas

You can use Pycharms virtual environment features to install these onto your interpreter in the settings of Pycharm. 

Other than that credit goes to Jovian.ai for teaching me about the concepts and fundamentals of AI. They also synchronously taught me about the functionalities of PyTorch and the best practices for building with this amazing framework. Without the organized efforts from everyone over there I might still be stuck in a learning curve. 
