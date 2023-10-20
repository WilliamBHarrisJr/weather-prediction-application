import pandas as pd
import numpy as np
from PIL import Image
from sklearn import linear_model, metrics, model_selection
from matplotlib import pyplot
import customtkinter
from pandas.plotting import scatter_matrix

df = pd.read_csv("data/seattle-weather.csv")
df.drop(['date'], axis=1, inplace=True)

model = linear_model.LogisticRegression()

y = df.values[:, 4]
x = df.values[:, 0:4]
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=.25)

model.max_iter = 1300
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

test_accuracy = "Test accuracy: " + str(metrics.accuracy_score(y_test, y_pred))


def show_histo():
    pyplot.close()
    df.hist()
    pyplot.show()


def show_scatter():
    pyplot.close()
    scatter_matrix(df)
    pyplot.show()


def show_heatmap():
    data = np.random.random((12, 12))
    pyplot.imshow(data)
    pyplot.colorbar()
    pyplot.title("Heat Map")
    pyplot.show()


def show_pie_chart():
    pyplot.close()
    fog = df.loc[df['weather'] == 'fog'].count().iloc[0]
    drizzle = df.loc[df['weather'] == 'drizzle'].count().iloc[0]
    rain = df.loc[df['weather'] == 'rain'].count().iloc[0]
    snow = df.loc[df['weather'] == 'snow'].count().iloc[0]
    sun = df.loc[df['weather'] == 'sun'].count().iloc[0]
    labels = ['Fog', 'Drizzle', 'Rain', 'Snow', 'Sunny']
    pyplot.pie([fog, drizzle, rain, snow, sun], labels=labels, autopct='%.2f %%')
    pyplot.title("Weather")
    pyplot.show()


def fahrenheit_to_celsius(fahrenheit):
    celsius = (fahrenheit - 32) * (5 / 9)
    return celsius


def inches_to_mm(inches):
    mm = inches * 25.4
    return mm


def predict():
    try:
        if radio_var.get() == 1:
            precipitation = float(entry_precipitation.get())
            maxtemp = float(entry_maxtemp.get())
            mintemp = float(entry_mintemp.get())
            windspeed = float(entry_windspeed.get())
            valuelist = [precipitation, maxtemp, mintemp, windspeed]
            prediction = model.predict([valuelist])
            display_prediction(prediction)

        elif radio_var.get() == 2:
            precipitation = inches_to_mm(float(entry_precipitation.get()))
            maxtemp = fahrenheit_to_celsius(float(entry_maxtemp.get()))
            mintemp = fahrenheit_to_celsius(float(entry_mintemp.get()))
            windspeed = float(entry_windspeed.get())
            valuelist = [precipitation, maxtemp, mintemp, windspeed]
            prediction = model.predict([valuelist])
            display_prediction(prediction)


    except:
        label_image.configure(image=image_home, text="INVALID INPUT!", text_color="white", font=("Roboto", 20, "bold"))
        label_image_credit.configure(text="")


def display_prediction(prediction):
    if "snow" in prediction:
        label_image.configure(image=image_snow, text="SNOW", font=("Roboto", 90, "bold"), text_color="white")
        label_image_credit.configure(text="Image Credit: Dzmitrock/Shutterstock")

    elif "rain" in prediction:
        label_image.configure(image=image_rain, text="RAIN", font=("Roboto", 100, "bold"), text_color="white")
        label_image_credit.configure(text="Image Credit: Julia_Sudnitskaya")

    elif "drizzle" in prediction:
        label_image.configure(image=image_drizzle, text="DRIZZLE", font=("Roboto", 70, "bold"), text_color="white")
        label_image_credit.configure(text="Image Credit: Tonya West")

    elif "sun" in prediction:
        label_image.configure(image=image_sun, text="SUNNY", font=("Roboto", 80, "bold"), text_color="yellow")
        label_image_credit.configure(text="Image Credit: NBC Chicago")

    elif "fog" in prediction:
        label_image.configure(image=image_fog, text="FOG", font=("Roboto", 90, "bold"), text_color="white")
        label_image_credit.configure(text="Image Credit: Dene' Miles")


def radiobutton_event():
    if radio_var.get() == 1:
        label_precipitation.configure(text="Precipitation (mm)")
        label_maxtemp.configure(text="Max Temperature (Celsius)")
        label_mintemp.configure(text="Min Temperature (Celsius)")

    elif radio_var.get() == 2:
        label_precipitation.configure(text="Precipitation (In)")
        label_maxtemp.configure(text="Max Temperature (Fahrenheit)")
        label_mintemp.configure(text="Min Temperature (Fahrenheit)")


customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("blue")

root = customtkinter.CTk()
root.title('Weather Prediction')
root.geometry('600x930')
root.iconbitmap("images/icon.ico")

frame_input = customtkinter.CTkFrame(master=root)
frame_input.pack(pady=20, padx=60, fill="both", expand=True)

label_main = customtkinter.CTkLabel(master=frame_input, text="Weather Prediction", font=("Roboto", 24))
label_main.pack(pady=12, padx=10)

label_accuracy = customtkinter.CTkLabel(master=frame_input, text=test_accuracy, font=("Roboto", 12))
label_accuracy.pack(pady=6, padx=10)

radio_var = customtkinter.IntVar()

frame_radio = customtkinter.CTkFrame(master=frame_input)
frame_radio.pack(pady=10, padx=60, fill="both", expand=True)

radiobutton_metric = customtkinter.CTkRadioButton(master=frame_radio, text="Metric", command=radiobutton_event,
                                                  variable=radio_var, value=1)
radiobutton_metric.pack(padx=10, pady=10, side=customtkinter.LEFT, anchor='e', expand=True)
radiobutton_metric.select()

radiobutton_imperial = customtkinter.CTkRadioButton(master=frame_radio, text="Imperial", command=radiobutton_event,
                                                    variable=radio_var, value=2)
radiobutton_imperial.pack(padx=10, pady=10, side=customtkinter.LEFT, anchor='w', expand=True)

label_precipitation = customtkinter.CTkLabel(master=frame_input, text="Precipitation (mm)", font=("Roboto", 12))
label_precipitation.pack(padx=10)

entry_precipitation = customtkinter.CTkEntry(master=frame_input, placeholder_text="Precipitation", justify="center")
entry_precipitation.pack(pady=(5, 18))

label_maxtemp = customtkinter.CTkLabel(master=frame_input, text="Max Temperature (Celsius)", font=("Roboto", 12))
label_maxtemp.pack(padx=10)

entry_maxtemp = customtkinter.CTkEntry(master=frame_input, placeholder_text="Max Temperature", justify="center")
entry_maxtemp.pack(pady=(5, 18))

label_mintemp = customtkinter.CTkLabel(master=frame_input, text="Min Temperature (Celsius)", font=("Roboto", 12))
label_mintemp.pack(padx=10)

entry_mintemp = customtkinter.CTkEntry(master=frame_input, placeholder_text="Min Temperature", justify="center")
entry_mintemp.pack(pady=(5, 18))

label_windspeed = customtkinter.CTkLabel(master=frame_input, text="Wind Speed (MPH)", font=("Roboto", 12))
label_windspeed.pack(padx=10)

entry_windspeed = customtkinter.CTkEntry(master=frame_input, placeholder_text="Wind Speed", justify="center")
entry_windspeed.pack(pady=(5, 18))

button_predict = customtkinter.CTkButton(master=frame_input, text="Predict", command=predict)
button_predict.pack(pady=(15, 12))

label_prediction = customtkinter.CTkLabel(master=root, text="Prediction:", font=("Roboto", 24))
label_prediction.pack(pady=(5, 0), padx=10)

frame_prediction = customtkinter.CTkFrame(master=root)
frame_prediction.pack(pady=12, padx=60, fill="both", expand=True)

image_snow = customtkinter.CTkImage(light_image=Image.open('images/snow.jpg'), size=(300, 200))
image_sun = customtkinter.CTkImage(light_image=Image.open('images/sun.jpg'), size=(300, 200))
image_rain = customtkinter.CTkImage(light_image=Image.open('images/rain.png'), size=(300, 200))
image_drizzle = customtkinter.CTkImage(light_image=Image.open('images/drizzle.jpg'), size=(300, 200))
image_fog = customtkinter.CTkImage(light_image=Image.open('images/fog.jpg'), size=(300, 200))
image_home = customtkinter.CTkImage(light_image=Image.open('images/home.PNG'), size=(300, 200))

label_image = customtkinter.CTkLabel(master=frame_prediction, text="", image=image_home, font=("Roboto", 80, "bold"))
label_image.pack(pady=(15, 5), padx=10)

label_image_credit = customtkinter.CTkLabel(master=frame_prediction, text="", font=("roboto", 10))
label_image_credit.pack(padx=10)

button_show_hist = customtkinter.CTkButton(master=root, text="Histogram", command=show_histo)
button_show_hist.pack(pady=15, padx=5, side=customtkinter.LEFT, anchor='e', expand=True)

button_show_chart = customtkinter.CTkButton(master=root, text="Scatter Matrix", command=show_scatter)
button_show_chart.pack(pady=15, padx=5, side=customtkinter.LEFT, anchor='center', expand=True)

button_show_heatmap = customtkinter.CTkButton(master=root, text="Pie Chart", command=show_pie_chart)
button_show_heatmap.pack(pady=15, padx=5, side=customtkinter.LEFT, anchor='w', expand=True)

root.mainloop()