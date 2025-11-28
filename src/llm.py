from openai import OpenAI
from google import genai
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_KEY")
client = genai.Client(api_key=API_KEY)


def formatJson(focus: str = None, kneeAngles : list = None, centerOfGravity: list = None, speed: list = None, size : int = None) -> dict:
    results = {
        "focus": focus,
        "kneeAngles": kneeAngles,
        "centerOfGravity": centerOfGravity,
        "speed": speed,
        "size": size
    }

    return results


def createLLMPrompt(keyMetrics : dict) -> str:
    prompt = f"""
        You are a professional physical training specializing in soccer training. Given this players biomechanics and the focus for this session analyze and give them these two insights. Format each as cleanly formatted lists with a clear connection between which drills are for which weaknesses

        1. Current Weaknesses (e.g. Knee Angles suggest xyz, resulting in abc)
        2. Drills to Fix Each Weakness (e.g. I recommend drills xyz to help with abc)

        Biomechanics Data: {keyMetrics}
    """

    return prompt

def queryLLM(prompt : str):
    response = client.models.generate_content(
        model="gemini-2.5-flash", 
        contents=prompt
    )

    return response.text