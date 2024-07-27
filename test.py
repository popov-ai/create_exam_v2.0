from llama_index.llms.groq import Groq
import os
groq_api = os.getenv('GROQ_API_KEY')
llm = Groq(model="llama3-70b-8192", api_key=groq_api)

print(llm)