from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from groq import Groq
from dotenv import load_dotenv # Importa a função para carregar .env

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv() 

# 1. Configuração do FastAPI
app = FastAPI()

# 2. Configuração do CORS (Para permitir que o index.html acesse o backend)
origins = ["*"] # Em um projeto real, você colocaria apenas o endereço do seu frontend

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Definição do modelo de dados para a requisição
class TextRequest(BaseModel):
    text: str

# 4. Inicialização do Cliente Groq
# O cliente Groq agora usa a chave GROQ_API_KEY carregada pelo load_dotenv()
client = Groq()

# 5. Endpoint da API
@app.post("/summarize")
def summarize_text(data: TextRequest):
    text_to_summarize = data.text
    
    # Validação simples
    if not text_to_summarize or len(text_to_summarize) < 10:
        raise HTTPException(status_code=400, detail="O texto é muito curto para resumir.")

    try:
        # Chamada à API Groq (AGORA COM O NOME DO MODELO ATIVO)
        response = client.chat.completions.create(
            # Modelo Llama 3 8B ativo - Verificado na documentação do Groq
            model="llama-3.3-70b-versatile", 
            messages=[
                {"role": "system", "content": "Você é um assistente de resumo. Resuma o texto fornecido em no máximo 50 palavras, focando nos pontos principais."},
                {"role": "user", "content": text_to_summarize}
            ],
            max_tokens=200 
        )
        
        # Extrai o resumo da resposta do Groq
        summary = response.choices[0].message.content
        
        # O backend retorna o resumo dentro da chave 'summary'
        return {"summary": summary}
        
    except Exception as e:
        # Captura e registra qualquer erro da API Groq
        print(f"Erro ao chamar a API Groq: {e}")
        # Retorna um erro 500 mais informativo para o frontend
        raise HTTPException(status_code=500, detail="Erro ao gerar o resumo. Verifique a chave ou o limite da API.")