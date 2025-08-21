import pyttsx3
import speech_recognition as sr
import webbrowser
import requests
import os
import psutil
from datetime import date, time
import winshell


from dotenv import load_dotenv
load_dotenv()

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    engine.stop()




def listen_command(prompt="Listening...", timeout=5, phrase_time_limit=5):
    """Helper to listen once and return recognized text"""
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        print(prompt)
        try:
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            text = recognizer.recognize_google(audio)
            print("Heard:", text)
            return text.lower()
        except sr.UnknownValueError:
            print("Could not understand audio")
            return ""
        except sr.RequestError:
            print("Error: check your internet connection.")
            return ""
        except sr.WaitTimeoutError:
            print("Error: Wait Time Out Error Occured")
            return ""


def open_website(site_to_open):
    
    print(f"{site_to_open} is Opened")
    url = f"https://{site_to_open}.com"  
    print(url)
    webbrowser.open(url)




def play_music(song):
    # Stub: Replace with your music dict
    speak(f"Playing {song}")
    webbrowser.open("https://youtube.com/results?search_query=" + song)


def get_news():
    url = "https://newsapi.org/v2/top-headlines"
    params = {
        "country": "us",
        "apiKey": os.getenv("NEWS_API_KEY")
    }
    response = requests.get(url, params=params)
    articles = response.json().get("articles", [])
    
    headlines = [article["title"] for article in articles[:5]]
    return headlines


def get_weather(city="London"):
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": os.getenv("Weather_API_KEY"), "units": "metric"}
    response = requests.get(url, params=params)
    data = response.json()
    
    if data.get("main"):
        temp = data["main"]["temp"]
        desc = data["weather"][0]["description"]
        return f"The temperature in {city} is {temp}°C with {desc}."
    else:
        return "Sorry, I couldn't fetch the weather."



def get_cpu_usage():
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    return f"CPU usage is {cpu_percent}% and memory usage is {memory.percent}%."

def date_time():
    
    current_date = date.now()
    current_time = time.now()
    return f"Time Right now is {current_time} and Date is {current_date}"


def empty_recyclebin():
    confirm = input(f"⚠️ Are you sure you want to Empty Recycle Bin? (y/n): ")
    if confirm.lower() == "y":
        try:
            winshell.recycle_bin().empty(confirm=False, show_progress=False, sound=False)
            print("Recycle Bin emptied successfully.")
        except Exception as e:
            print(f"An error occurred: {e}")
    else:
        return "Deletion cancelled."

def decide(command):
    
    if "website" in command or "open" in command:
        parts = command.split(" ")[1]
        print(parts)
        if len(parts) > 1:
            open_website(parts)
        else:
            print("please tell me which website you want to open")
            speak("Please tell me which website you want to open.")
                    

    elif "play" in command:
        parts = command.split(" ", 1)
        if len(parts) > 1:
            play_music(parts[1])
        else:
            print("please tell me what to play")
            speak("Please tell me what to play.")
        #News 
    elif "news" in command:
        news = get_news()
        speak(f"Here are the top headlines: {news}")
        print(f"Top Headlines: {news}")

        #For Weather
    elif "weather" in command:
        city = command.split(" ")
                    
        if len(city) > 1:
            weather = get_weather(city[len(city)-1])
            speak(weather)
            print(weather)
        else:
            listen_command("Please tell me the city name")
            print("Please tell me the city name")
    #to print cpu performence
    elif "check" in command or "CPU" in command:
        
        speak(get_cpu_usage())
        print(get_cpu_usage())
    #To show date and time
    elif "time" in command or "date" in command:
        speak(date_time())
        print(date_time())
    #To Empty Recycle bin
    elif "empty" in command or "bin" in command: 
        speak(empty_recyclebin())
        print(empty_recyclebin())

    elif "exit" in command or "quit" in command:
        print("Exiting the Program, Goodbye!")
        speak("Exiting the Program, Goodbye!")
        return "exit"

    else:
        return "repeat"

import time, sys
def listen_for_wake_word(wake_word="wake up", active_timeout=15):
    """Wake word once → stay active for commands → sleep after timeout"""
    active_mode = False
    last_command_time = 0

    while True:
        if not active_mode:
            # Waiting for wake word
            speak(f"Listening for wake word '{wake_word}'...")
            text = listen_command(f"Listening for wake word '{wake_word}'...")
            if wake_word in text:
                speak("          Bolo Ustad G")
                active_mode = True
                last_command_time = time.time()
        else:
            # Already awake, listen for commands
            while True:
                command = listen_command("Listening for your command...")
                last_command_time = time.time()
                checking = decide(command)
                if checking != "repeat" or checking != "exit":
                    print("yes coming")
                    sys.exit("Exiting as User asked")
            

            else:
                
                # No command heard: check if timeout expired
                if time.time() - last_command_time > active_timeout:
                    print("Going back to sleep, say Alexa to wake me up again")
                    speak("Going back to sleep, say Alexa to wake me up again.")
                    break
                    active_mode = False


if __name__ == "__main__":
    
    listen_for_wake_word("wake up")
