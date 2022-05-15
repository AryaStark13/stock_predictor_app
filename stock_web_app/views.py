from django.shortcuts import render
import tensorflow as tf
import numpy as np
import pandas as pd
import pandas_datareader as data
import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


model = tf.keras.models.load_model(r'C:\Users\ariha\Desktop\Django\django_stock_predictor\stock_predictor_app\Saved Model\Future Stock Prediction Using Last 7 Days Values.h5')

def predict(request):

    if request.method == 'POST':
        stock_ticker = request.POST['stock_ticker']

        start_date = datetime.date.today() - datetime.timedelta(30)
        end_date = datetime.date.today()

        df = data.DataReader(stock_ticker, 'yahoo', start_date, end_date)

        df.reset_index(inplace=True)

        x_future = np.array(df.Close[-7:])

        scaler = MinMaxScaler(feature_range=(0, 1))

        x_future = scaler.fit_transform(x_future)
        x_future = np.reshape(x_future, (1, 7, 1))

        y_future = []

        while len(y_future) < 7:
        #     Predicting future values using 7-day moving averages of the last day 7 days.
            p = model.predict(x_future)[0]
            
        #     Appending the predicted value to y_future
            y_future.append(p)
            
        #     Updating input variable, x_future
            x_future = np.roll(x_future, -1)
            x_future[-1] = p
        
        y_future = np.array(y_future)
        y_future = np.reshape(y_future, (7))

        y_future = scaler.inverse_transform(y_future)

        y_future = np.reshape(y_future, (7, 1))

        last7 = pd.DataFrame(df.Close[-7:])
        last7.reset_index(drop=True, inplace=True)
        y_future = pd.DataFrame(y_future, columns=['Close'])
        predictions = pd.concat([last7, y_future], ignore_index=True)

        return render(request, 'index.html', {'predictions': predictions, 
                                            'stock_ticker': stock_ticker})

    return render(request, 'index.html')




    # plt.figure(figsize=(12, 6))
    # plt.plot(predictions[:7], '-', label='Past 7 Days True Values')
    # plt.plot(predictions[6:8], ':')
    # plt.plot(predictions[7:8], 'o', label='Next Day Predicted Value')
    # plt.legend(fontsize=12)
    # plt.show()
    # plt.savefig('Next Day Price.jpg')