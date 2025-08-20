import speech_recognition as sr
import pyttsx3
import webbrowser
import music_names # to import dictionary containing names of songs from another python file

#Function to open websites
def open_website(query):
    if "open google " in query.lower():
        speak("Google is opened")
        webbrowser.open("https://google.com")

    elif "open facebook" in query.lower():
        speak("Facebook is opened")
        webbrowser.open("https://facebook.com")

    elif "open youtube" in query.lower():
        speak("YouTube is opened")
        webbrowser.open("https://youtube.com")

    elif "open linkedin" in query.lower():
        speak("Linkedin is opened")
        webbrowser.open("https://linkedin.com")

#function to fetch links of song to play
def play_music(song):
    link = music_names.music[song]
    webbrowser.open(link)

# Initialize TTS engine once
engine = pyttsx3.init()
def speak(text):
    engine.say(text)
    engine.runAndWait()

def listen_for_wake_word(wake_word="alexa"):
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        print(f"Listening for wake word '{wake_word}'...")

        while True:
            try:
                # Listen for speech
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
                text = recognizer.recognize_google(audio).lower()
                print("Heard:", text)

                if "website" in text.lower():
                    open_website(text)

                elif "play" in text.lower():
                    song = text.lower().split(" ")[1]
                    print(song, type(song))
                    play_music(song)

                # Check for wake word
                elif wake_word in text.lower():
                    speak("Yes? What can I do for you today?")
                    # Here you can call another function for commands after wake word
                    print("Wake word detected! Ready for command.")
                
                #Exits the loop and stops execution
                elif "exit" in text.lower():
                    speak("Exiting the Program, GoodBye")
                    break

            except sr.UnknownValueError:
                # Could not understand audio
                pass
            except sr.RequestError:
                print("Could not request results; check your internet connection.")
            except sr.WaitTimeoutError:
                # No speech detected within timeout, just continue listening
                pass

if __name__ == "__main__":
    listen_for_wake_word("bitty")
