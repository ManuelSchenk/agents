from dotenv import load_dotenv
# from openai import OpenAI
from anthropic import Anthropic
import json
import os
import requests
from pypdf import PdfReader
import gradio as gr
from pathlib import Path

REL_PATH = Path(__file__).parent

load_dotenv(override=True)

def push(text):
    requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": os.getenv("PUSHOVER_TOKEN"),
            "user": os.getenv("PUSHOVER_USER"),
            "message": text,
        }
    )


def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"Recording {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}

def record_unknown_question(question):
    push(f"Recording {question}")
    return {"recorded": "ok"}

record_user_details_json = {
    "name": "record_user_details",
    "description": "Verwende dieses Tool, um zu erfassen, dass ein Benutzer Interesse an Kontaktaufnahme hat und eine E-Mail-Adresse bereitgestellt hat",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {
                "type": "string",
                "description": "Die E-Mail-Adresse dieses Benutzers"
            },
            "name": {
                "type": "string",
                "description": "Der Name des Benutzers, falls angegeben"
            }
            ,
            "notes": {
                "type": "string",
                "description": "Zusätzliche Informationen über das Gespräch, die es wert sind, aufgezeichnet zu werden, um Kontext zu geben"
            }
        },
        "required": ["email"],
        "additionalProperties": False
    }
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Verwende dieses Tool immer, um jede Frage aufzuzeichnen, die nicht beantwortet werden konnte, weil du die Antwort nicht wusstest",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "Die Frage, die nicht beantwortet werden konnte"
            },
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

tools = [{"type": "function", "function": record_user_details_json},
        {"type": "function", "function": record_unknown_question_json}]


class Me:

    def __init__(self):
        # self.openai = OpenAI(api_key=os.getenv('XAI_API_KEY'), base_url="https://api.x.ai/v1")
        self.anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        self.name = "Manuel Schenk"
        reader = PdfReader(REL_PATH / "me/Lebenslauf.pdf")
        self.linkedin = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                self.linkedin += text
        with open(REL_PATH / "me/selbstbeschreibung.txt", "r", encoding="utf-8") as f:
            self.summary = f.read()


    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"Tool called: {tool_name}", flush=True)
            tool = globals().get(tool_name)
            result = tool(**arguments) if tool else {}
            results.append({"role": "tool","content": json.dumps(result),"tool_call_id": tool_call.id})
        return results
    
    def system_prompt(self):
        system_prompt = f"Du agierst als {self.name} – einen Data Engineer aus Erfurt, der für seine Neugierde und schnelle Auffassungsgabe bekannt ist. \
Du beantwortest Fragen auf einer Job Vermittlungs Website, \
insbesondere Fragen zu {self.name}s Karriere, Hintergrund, Fähigkeiten und Erfahrungen. \
Deine Verantwortung ist es, {self.name} bei Interaktionen auf der Website so authentisch wie möglich zu repräsentieren. \
Du erhältst eine Zusammenfassung von {self.name}s Hintergrund und Lebenslauf, die du zum Beantworten von Fragen nutzen kannst. \
Sei professionell und ansprechend, als würdest du mit einem potenziellen Kunden oder zukünftigen Arbeitgeber sprechen, der auf die Website gestoßen ist. \
Wenn du die Antwort auf eine Frage nicht weißt, verwende dein record_unknown_question Tool, um die Frage aufzuzeichnen, die du nicht beantworten konntest, auch wenn es sich um etwas Triviales oder Karriere-Unabhängiges handelt. \
Wenn der Benutzer sich auf eine Diskussion einlässt, versuche ihn dazu zu bringen, per E-Mail Kontakt aufzunehmen; frage nach seiner E-Mail-Adresse und erfasse sie mit deinem record_user_details Tool. "

        system_prompt += f"\n\n## Zusammenfassung:\n{self.summary}\n\n## Lebenslauf:\n{self.linkedin}\n\n##LinkedIn-Profil: https://www.linkedin.com/in/manuel-schenk-48246117a/"
        system_prompt += f"Mit diesem Kontext chatte bitte mit dem Benutzer und bleibe dabei immer in der Rolle als {self.name}."
        return system_prompt
    
    def chat(self, message, history):
        messages = [{"role": "system", "content": self.system_prompt()}] + history + [{"role": "user", "content": message}]
        done = False
        while not done:
            # response = self.openai.chat.completions.create(model="grok-4", messages=messages, tools=tools)
            response = self.anthropic.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=2048,
                messages=messages,
                tools=tools
            )
            if response.choices[0].finish_reason=="tool_calls":
                message = response.choices[0].message
                tool_calls = message.tool_calls
                results = self.handle_tool_call(tool_calls)
                messages.append(message)
                messages.extend(results)
            else:
                done = True
        return response.choices[0].message.content
    

if __name__ == "__main__":
    me = Me()
    gr.ChatInterface(me.chat, type="messages").launch()
    