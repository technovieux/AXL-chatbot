from langchain_ollama import OllamaLLM
import cv2
import pyttsx4
import speech_recognition as sr
from playsound3 import playsound
import numpy as np


model = OllamaLLM(model="stablelm-zephyr")

recognizer = sr.Recognizer()






def charger_reponses(fichier):
  reponses = {}
  with open(fichier, 'r', encoding='utf-8') as f:
    for ligne in f:
      question, reponse = ligne.strip().split(':')
      reponses[question.lower()] = reponse
  return reponses








def quantize(reponse):
    reponse = reponse.replace("\n","")
    reponse = reponse.replace("😊","")
    reponse = reponse.replace("😉","")
    reponse = reponse.replace("😊","")

    reponse = reponse.strip()
    







def filter(reponse):

    reponse = reponse.replace("😁","")
    reponse = reponse.replace("🤣","")
    reponse = reponse.replace("😅","")
    reponse = reponse.replace("🤩","")
    reponse = reponse.replace("😉","")
    reponse = reponse.replace("😬","")
    reponse = reponse.replace("🥺","")
    reponse = reponse.replace("😔","")
    reponse = reponse.replace("🎉","")
    reponse = reponse.replace("😍","")
    reponse = reponse.replace("💕","")
    reponse = reponse.replace("😜","")
    reponse = reponse.replace("🤔","")
    reponse = reponse.replace("😊","")
    reponse = reponse.replace("🔥","")
    reponse = reponse.replace("🥶","")
    reponse = reponse.replace("🐶","")
    reponse = reponse.replace("🐱","")
    reponse = reponse.replace("🦜","")
    reponse = reponse.replace("🦄","")
    reponse = reponse.replace("🍔","")
    reponse = reponse.replace("😋","")
    reponse = reponse.replace("☕","")
    reponse = reponse.replace("👍","")
    reponse = reponse.replace("💯","")
    reponse = reponse.replace("🌎","")
    reponse = reponse.replace("👋","")

    return reponse








def add(a,b):

	question = a
	reponse = b

	search_text = "#:#"
	replace_text = question + ":" + reponse + "\n" + "#:#"

	with open(r'questions-reponses.txt', 'r') as file: 
		data = file.read() 
		data = data.replace(search_text, replace_text) 
	with open(r'questions-reponses.txt', 'w', encoding="utf-8") as file: 
		file.write(data)







def AI(question):
    
    result = model.invoke(question)
    return(result)










def chatbot():
  fichier_reponses = "questions-reponses.txt"
  reponses = charger_reponses(fichier_reponses)
  
  print("Bonjour ! vous avez une question ?")

  while True :
    question = input("> ").lower()
    if (question == 'quit' or question == 'exit'):
      exit()

    if (question == 'listen'):
        end_program = False
        while not end_program:
            audio = capture_voice_input()
            text = convert_voice_to_text(audio)
            question = text
            end_program = True

    #if any(location in question for location in ["where"]):
        


    reponse = reponses.get(question, "None")

    if reponse == 'None':
      reponse = AI(question)
      quantize(reponse)
      #reponse = 'None'
    
    #TTS(reponse)
    print(reponse)
    reponse = reponse.replace("à""â""á","a")
    reponse = reponse.replace("À""Á""Â","A")
    reponse = reponse.replace("È""É""Ê""Ë","E")
    reponse = reponse.replace("è""é""ê""ë","e")
    reponse = reponse.replace("Ì""Í""Î""Ï","I")
    reponse = reponse.replace("ì""í""î""ï","i")
    
    add(question,reponse)













def camera_simple():
    cap = cv2.VideoCapture(0)

    if not (cap.isOpened()):
      print("Could not open video device")


    while(True):

      ret, frame = cap.read()

      cv2.imshow('preview',frame)

      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cap.release()
    cv2.destroyAllWindows()










def objects_recognition():

    thres = 0.5 # Threshold to detect object
    nms_threshold = 0.2 #(0.1 to 1) 1 means no suppress , 0.1 means high suppress 
    cap = cv2.VideoCapture(0) # Use 0 for built-in webcam, 1 for external webcam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,800) #width 
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,600) #height 
    cap.set(cv2.CAP_PROP_BRIGHTNESS,100) #brightness 

    classNames = []
    with open('objects.txt','r') as f:
        classNames = f.read().splitlines()
    #print(classNames)

    font = cv2.FONT_HERSHEY_PLAIN
    #font = cv2.FONT_HERSHEY_COMPLEX
    Colors = np.random.uniform(0, 255, size=(len(classNames), 3))

    weightsPath = "frozen_inference_graph.pb"
    configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
    net = cv2.dnn_DetectionModel(weightsPath,configPath)
    net.setInputSize(320,320)
    net.setInputScale(1.0/ 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    while True:
        success,img = cap.read()
        classIds, confs, bbox = net.detect(img,confThreshold=thres)
        bbox = list(bbox)
        confs = list(np.array(confs).reshape(1,-1)[0])
        confs = list(map(float,confs))
    
        indices = cv2.dnn.NMSBoxes(bbox,confs,thres,nms_threshold)
        #print("Indices:", indices) # print to debug
        #print("Type of indices:", type(indices)) # print to debug
    
        if len(classIds) != 0:
            for i in indices:
                box = bbox[i]
                color = Colors[classIds[i]-1]
                confidence = str(round(confs[i],2))
                x,y,w,h = box[0],box[1],box[2],box[3]
                cv2.rectangle(img, (x,y), (x+w,y+h), color, thickness=2)
                cv2.putText(img, classNames[classIds[i]-1]+" "+confidence,(x+10,y+20),font,1,color,2)
                box[2],box[3] = box[2]/2,box[3]/2
                cv2.circle(img, (x+box[2]  ,  y+box[3]), radius=5, color=color, thickness=10)

                #print(classNames[classIds[i]-1] + " : " + str(x+box[2]) +"   "+ str(y+box[3]))
    


        cv2.imshow("Output",img)
        if cv2.waitKey(1) & 0xFF == ord('q'): # Add a condition to quit the loop
            break
    cap.release()
    cv2.destroyAllWindows()

















  
def TTS(reponse):
  
  engine = pyttsx4.init()
  rate = engine.getProperty('rate')   # getting details of current speaking rate
  #print (rate)                        #printing current voice rate
  engine.setProperty('rate', 125)     # setting up new voice rate

  volume = engine.getProperty('volume')   #getting to know current volume level (min=0 and max=1)
  #print (volume)                          #printing current volume level
  engine.setProperty('volume',1.0)    # setting up volume level  between 0 and 1

  voices = engine.getProperty('voices')       #getting details of current voice
  engine.setProperty('voice', voices[1].id)   #changing index, changes voices. 1 for female and 0 for male

  engine.say(reponse)
  engine.runAndWait()
  engine.stop()
  title = './audio_files/' + reponse + '.mp3'
  engine.save_to_file(reponse, title)
  engine.runAndWait()









def play_sound(reponse):
   file = 'C:/Users/mathias/OneDrive/Bureau/projets personnels/projet AXL/audio_file/' + reponse + '.mp3'
   playsound(file)























def capture_voice_input():
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
    return audio







def convert_voice_to_text(audio):
    try:
        text = recognizer.recognize_google(audio)
        print(text)
    except sr.UnknownValueError:
        text = ""
        print("Sorry, I didn't understand that.")
    except sr.RequestError as e:
        text = ""
        print("Error; {0}".format(e))
    return text






def process_voice_command(text):
    if "hello" in text.lower():
        print("Hello! How can I help you?")
    elif "goodbye" in text.lower():
        print("Goodbye! Have a great day!")
        return True
    else:
        print("I didn't understand that command. Please try again.")
    return False



def full_recognizer():
    end_program = False
    while not end_program:
        audio = capture_voice_input()
        text = convert_voice_to_text(audio)
        end_program = True