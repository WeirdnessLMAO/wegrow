
from flask import Flask, request, render_template_string, render_template, request, redirect, url_for
from flask_ngrok import run_with_ngrok
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pyngrok import ngrok
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
#--------------------------------------------------------------------

# Reading the Dataset
cropdf = pd.read_csv("Crop_recommendation (1).csv")

# Print some sample data from dataset
cropdf.head()

cropdf.tail()

# Extract only parametres and drop end results
x = cropdf.drop('label',axis=1)

# Extract only end results
y = cropdf['label']

# import library to split training and testing data
from sklearn.model_selection import train_test_split

# X_train - Training Dataset
# y_train - Expected Training Results

# X_test - Testing Dataset
# y_test - Expected Testing Results

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3,
                                                    shuffle =True, random_state = 0)
                                                    
# Creating a lightgbm model
import lightgbm as lgb

model = lgb.LGBMClassifier()

# Training the model using Training Data
model.fit(x_train, y_train)

# Predicting the outputs over testing data
y_pred = model.predict(x_test)

# Library to measure accuracy of model.


from sklearn.metrics import accuracy_score

# Find accuracy on Expected Output and Predicted Output on Testing Data
accuracy=accuracy_score(y_pred, y_test)
print('LightGBM Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_pred)))

# Find Training Score on Expected Output and Predicted Output on Training Data
y_pred_train = model.predict(x_train)
print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))

# Find Training Score on Expected Output and Predicted Output on Training Data
y_pred_test = model.predict(x_test)
print('Testing-set accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred_test)))

print("Enter the value of Nitrogen:")
#N=int(input())
print("Enter the value of Phosphorus :")
#P=int(input())
print("Enter the value of Potassium :")
#K=int(input())
print("Enter the value of temperature:")
#temperature=float(input())
print("Enter the value of humidity:")
#humidity=float(input())
print("Enter the value of pH:")
#pH=float(input())
print("Enter the value of rainfall:")
#rainfall=float(input())



#model.predict([[N,P,K,temperature,humidity,pH,rainfall]])

#-----------------------------------------------------------------------

port_no =  5000
app = Flask(__name__)
#ngrok.set_auth_token("")#2VhO8oHg5am7RjLs666GeeAtV8x_6MD6LCep7yQoR1bgPzCom")
#public_url =  ngrok.connect(port_no).public_url
# Reading the Dataset
cropdf = pd.read_csv("Crop_recommendation (1).csv")
x = cropdf.drop('label',axis=1)
# Extract only end results
y = cropdf['label']

# Load your data (replace this with your actual data loading code)
# Assuming you have a DataFrame with features (x) and labels (y)
# For example:
# x = pd.read_csv("your_features.csv")
# y = pd.read_csv("your_labels.csv")

app = Flask(__name__)
run_with_ngrok(app)  # Start ngrok when app is run



# Function to train the model and make predictions
def predict_result(param1, param2, param3, param4, param5, param6, param7):

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=0)

    # Create and train your model (replace with your model creation and training code)
    model = lgb.LGBMClassifier()
    model.fit(x_train, y_train)

    # Make predictions based on the input parameters
    input_params = [param1, param2, param3, param4, param5, param6, param7]
    result = model.predict([input_params])

    return result

@app.route('/')
def index():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username')
    password = request.form.get('password')

    # Implement your authentication logic here (e.g., check username and password)

    # For demonstration, we'll assume a successful login
    if username == 'test' and password == 'test':
        return redirect(url_for('landing'))
    elif username == 'dev' and password == 'dev':
        return redirect(url_for('distributor'))
    else:
        print(username,password)
        return ("Login failed. Please try again.<script>alert('lol')</script>")

@app.route('/distributor')
def distributor():
    return render_template('distributor.html')

@app.route('/grow')
def grow():
    return render_template('grow.html')

@app.route('/landing')
def landing():
    return render_template('landing.html')
    
@app.route('/mlmodel', methods=['GET', 'POST'])
def ml():
    input_values = None
    result = None

    if request.method == 'POST':
        param1 = float(request.form['param1'])
        param2 = float(request.form['param2'])
        param3 = float(request.form['param3'])
        param4 = float(request.form['param4'])
        param5 = float(request.form['param5'])
        param6 = float(request.form['param6'])
        param7 = float(request.form['param7'])

        input_values = (param1, param2, param3, param4, param5, param6, param7)

        # Call your model to predict the result
        result = predict_result(param1, param2, param3, param4, param5, param6, param7)
        print(result)
    return render_template('grow.html', input_values=input_values, result=result)

if __name__ == '__main__':
    # Run the Flask app
    app.run()
    
#----------------------------------------------------------------------------------

# Generate some example data
# Replace this with your dataset and target variable
from sklearn.datasets import load_iris
data = load_iris()
X = data.data
y = data.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3,
                                                    shuffle =True, random_state = 0)

# Train a LightGBM model
params = {
    "objective": "multiclass",
    "num_class": len(np.unique(y_train)),
    "boosting_type": "gbdt",
    "metric": "multi_logloss",
}
train_data = lgb.Dataset(X_train, label=y_train)
model = lgb.train(params, train_data, num_boost_round=100)

# Evaluate the model on the test set
y_pred = model.predict(X_test)
y_pred_class = np.argmax(y_pred, axis=1)

# Calculate and print classification report and confusion matrix
print("Classification Report:")
print(classification_report(y_test, y_pred_class))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_class))

# Plot ROC curves for each class (binary classification only)
if len(np.unique(y_train)) == 2:
    fpr, tpr, thresholds = roc_curve(y_test, y_pred[:, 1])
    auc = roc_auc_score(y_test, y_pred[:, 1])

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()

# Plot feature importance
lgb.plot_importance(model, max_num_features=10, figsize=(8, 6))
plt.title("Feature Importance")
plt.show()
