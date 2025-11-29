import yoloPose.poseAnalysis as poseAnalysis
import multiPlayerClassification.playerClass as playerClass
import llm
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
key=os.getenv("GEMINI_KEY")

#create gemini client
client = genai.Client(api_key=key)

def main():
    leftKneeAngles, rightKneeAngles = poseAnalysis.extractKeyFrames()
    json = llm.formatJson(focus="Shooting a soccer ball", kneeAngles=rightKneeAngles)
    prompt = llm.createLLMPrompt(json)
    response = llm.queryLLM(prompt=prompt)

    print(response)

def testMultiPlayer():
    playerClass.testYoloTracking()


    #pose analysis --> conv to json --> send downstream to llm --> return output

if __name__ == "__main__":
    #main()
    testMultiPlayer()