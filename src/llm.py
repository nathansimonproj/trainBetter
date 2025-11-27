from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("API_KEY")
client = OpenAI(api_key=API_KEY)


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
    
    response = client.responses.create(

        model="gpt-4o-mini",
        input=prompt,
    )

    return response.output_text
    

