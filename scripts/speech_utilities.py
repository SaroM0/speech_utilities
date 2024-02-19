#!/usr/bin/env python3
import time
import rospy
import numpy
import pydub
import threading
import rosservice
import subprocess
import ConsoleFormatter
import sounddevice
from scipy.io.wavfile import write

# Speech_msgs
from speech_msgs.srv import speech2text_srv, chatgpt_srv

# Robot_msgs
from robot_toolkit_msgs.srv import audio_tools_srv, misc_tools_srv
from robot_toolkit_msgs.msg import audio_tools_msg, speech_msg, text_to_speech_status_msg, misc_tools_msg


# Naoqi_msgs
from naoqi_bridge_msgs.msg import AudioBuffer

# AssemblyAI
import assemblyai as aai

# openAI
from openai import OpenAI


class SpeechUtilities:
    
    # ===================================================== INIT ==================================================================
    
    def __init__(self):
        
        try:
            availableServices = rosservice.get_service_list()
            self.ROS=True
        except:
            self.ROS=False
        
        # ================================== GLOBAL VARIABLES ========================================
        
        aai.settings.api_key = "5a5ee5020ab94e57a97dcc98e8b8ef81"
        self.file_path = 'output_audio.mp3'
        # self.client = OpenAI(organization='',)
            
        # ==================================== IF LOCAL ==============================================
            
        if not self.ROS or '/robot_toolkit/audio_tools_srv' not in availableServices:
            subprocess.Popen('roscore')
            time.sleep(2)
            rospy.init_node('SpeechUtilities', anonymous=True)
            self.audio_pub=rospy.Publisher('/mic', AudioBuffer, queue_size=10)
            local_audio = threading.Thread(target=self.publish_local_audio)
            local_audio.start()
            print(consoleFormatter.format("Speech utilities using local mic", "OKGREEN"))
            
        # =============================== IF PEPPER AVAILABLE ========================================
            
        if self.ROS:
            rospy.init_node('SpeechUtilities', anonymous=True)
            rospy.wait_for_service('/robot_toolkit/audio_tools_srv')
            self.audioToolsService = rospy.ServiceProxy('/robot_toolkit/audio_tools_srv', audio_tools_srv)
            self.enableSpeech = audio_tools_msg()
            self.enableSpeech.command = "enable_tts"
            self.audioToolsService(self.enableSpeech)

            # Custom speech parameters for robot
            self.customSpeech = audio_tools_msg()
            self.customSpeech.command = "set_speech_params"
            self.customSpeech.speech_parameters.pitch_shift=1 # Grueso (1) o Agudo (2)
            self.customSpeech.speech_parameters.double_voice_level= 0.0
            self.customSpeech.speech_parameters.double_voice_time_shift= 0.0
            self.customSpeech.speech_parameters.speed= 120.0 # Velocidad al hablar
            self.audioToolsService(self.customSpeech)

            # Publisher toolkit
            self.speech_pub=rospy.Publisher('/speech', speech_msg, queue_size=10)
            if "/pytoolkit/ALAudioDevice/set_output_volume_srv" in availableServices:
                self.talkinSubscriber = rospy.Subscriber('/pytoolkit/ALTextToSpeech/status', text_to_speech_status_msg, self.check_speaking)
                self.pytoolkit = True
            print(consoleFormatter.format("--Speech utilities Running in PEPPER--", "OKGREEN"))
            
        # =============================== SERVICES DECLARATION ========================================
            
        print(consoleFormatter.format('waiting for speech2text service!', 'WARNING'))  
        self.speech2text= rospy.Service("speech_utilities/speech2text", speech2text_srv, self.callback_speech2text)
        print(consoleFormatter.format('speech2text on!', 'OKGREEN'))
        
        print(consoleFormatter.format('waiting for realTimeTranscription service!', 'WARNING'))  
        self.speech2text= rospy.Service("speech_utilities/realTimeTranscription", speech2text_srv, self.callback_real_time_transcription)
        print(consoleFormatter.format('realTimeTranscription on!', 'OKGREEN'))
        
        print(consoleFormatter.format('waiting for chatGPT_question_answer service!', 'WARNING'))  
        self.chatGPT_question_answer= rospy.Service("speech_utilities/chatGPT_question_answer", chatgpt_srv , self.callback_gpt_question_answer)
        print(consoleFormatter.format('chatGPT_question_answer on!', 'OKGREEN'))
        
    # ============================================== SPEECH SERVICES ============================================================

    # ================================== SPEECH2TEXT ========================================
    
    def callback_speech2text(self, req):
        print("Recording...")
        req.duration = 5
        myrecording = sounddevice.rec(int(req.duration * 16000), samplerate=16000, channels=2, dtype='int16')
        sounddevice.wait()
        write('output_audio.wav', 16000, myrecording)
        sound = pydub.AudioSegment.from_wav('output_audio.wav')
        sound.export(self.file_path, format="mp3")
        print(f"Recording saved as {self.file_path}")
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe('output_audio.wav')
        print(transcript.text)
        
    # =============================== REALTIMETRANSCRIPTION ================================

    def callback_real_time_transcription(self, x=0):
        def on_open(session_opened: aai.RealtimeSessionOpened):
            print("Session ID:", session_opened.session_id)

        def on_data(transcript: aai.RealtimeTranscript):
            if not transcript.text:
                return

            if isinstance(transcript, aai.RealtimeFinalTranscript):
                print(transcript.text, end="\r\n")
            else:
                print(transcript.text, end="\r")

        def on_error(error: aai.RealtimeError):
            print("An error occured:", error)

        def on_close():
            print("Closing Session")
            
        transcriber = aai.RealtimeTranscriber(
            on_data=on_data,
            on_error=on_error,
            sample_rate=44_100,
            on_open=on_open,
            on_close=on_close,
            )
        # Start the connection
        transcriber.connect()
        microphone_stream = aai.extras.MicrophoneStream()
        transcriber.stream(microphone_stream)
        
        # =============================== GPT Q&A ======================================
        
    def callback_gpt_question_answer(self, req):
        completion = self.client.chat.completions.create(
            model="gpt-3.5-turbo-instruct",
            prompt=req.question,
        )
        print(completion.choises[0].text)

    # ========================================== LOCAL AUDIO  SETUP =============================================================

    def publish_local_audio(self):
        with sounddevice.InputStream(callback=self.audio_callback, channels=1, samplerate=16000):
            while not rospy.is_shutdown():
                time.sleep(0.1)
            sounddevice.stop()
            subprocess.Popen('killall -9 roscore rosmaster')

    def audio_callback(self, indata, frames, time, status): 
        audio_msg = AudioBuffer()
        audio_data = (indata * 32767).astype(numpy.int16)
        audio_msg.data = audio_data.flatten().tolist()
        self.audio_pub.publish(audio_msg)

    # =============================================== PEPPER AUDIO  =============================================================

    def check_speaking(self,data):
        if data.status == "done":
            self.isTalking=False
        else:
            self.isTalking = True

    def turn_mic_pepper(self, enable):

        command = "enable" if enable else "disable"

        try:
            misc = rospy.ServiceProxy('/robot_toolkit/misc_tools_srv', misc_tools_srv)

            miscMessage = misc_tools_msg()
            miscMessage.command = "enable_all"
            misc(miscMessage)

            rospy.wait_for_service('/robot_toolkit/audio_tools_srv')
            audio = rospy.ServiceProxy('/robot_toolkit/audio_tools_srv', audio_tools_srv)

            audioMessage = audio_tools_msg()
            audioMessage.command = command
            audio(audioMessage)

            return True
        except rospy.ServiceException as e:
            print(f"Error al cambiar el estado del micrófono: {e}")
            return False

# =========================================================== MAIN ===============================================================

if __name__ == '__main__':
    consoleFormatter=ConsoleFormatter.ConsoleFormatter()
    speechUtilities = SpeechUtilities()
    try:
        print(consoleFormatter.format(" --- speech utilities node successfully initialized ---","OKGREEN"))
        rospy.spin()

    except rospy.ROSInterruptException:
        pass
