import requests
import json
import sys

API_URL = "http://localhost:11434/api/generate"

def chat():
    print("=== Antimatter AI Chat Interface ===")
    print("Type 'exit' to quit.\n")
    
    history = []
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        
        # Simple prompt construction
        prompt = user_input
        
        data = {
            "model": "antimatter",
            "prompt": prompt,
            "stream": True # Streaming response
        }
        
        print("Antimatter: ", end="", flush=True)
        
        try:
            with requests.post(API_URL, json=data, stream=True) as response:
                if response.status_code != 200:
                    print(f"Error: {response.text}")
                    continue
                    
                for line in response.iter_lines():
                    if line:
                        json_resp = json.loads(line)
                        token = json_resp.get("response", "")
                        print(token, end="", flush=True)
                        
                        if json_resp.get("done", False):
                            print() # Newline at end
        except requests.exceptions.ConnectionError:
            print("\nError: Could not connect to Ollama server. Is it running?")
            break

if __name__ == "__main__":
    chat()
