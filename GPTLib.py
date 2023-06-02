from langchain.llms import OpenAI
from langchain.docstore.document import Document
import requests
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
import pathlib
import subprocess
import tempfile
import os
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import re
import os
import openai
import tiktoken


class GPT():
  def __init__(self):
    pass


  def load_search_indexes(self, url: str) -> str:
    # Извлекаем из URL идентификатор документа для доступа по прямой ссылке
    match = re.search('/document/d/([a-zA-Z0-9-_]+)', url)
    if match is None:
        raise ValueError('Некорректная ссылка Google Docs')
    docID = match.group(1)

    # Скачиваем документ как текстовый файл
    response = requests.get(f'https://docs.google.com/document/d/{docID}/export?format=txt')
    response.raise_for_status()
    text = response.text
    # Проведём векторизация документа
    return self.create_embedding(text)
  
  def update_token(self, value):
    openai.api_key = value
    os.environ["OPENAI_API_KEY"] = openai.api_key
    print(f'Ключ сохранен!')

  
  def create_embedding(self, data):
    
    # Получение числа токенов в строке
    def num_tokens_from_string(string: str, encoding_name: str) -> int:
      encoding = tiktoken.get_encoding(encoding_name)
      num_tokens = len(encoding.encode(string))
      return num_tokens

    #  Разбивает текст на куски
    source_chunks = []
    splitter = CharacterTextSplitter(separator="\n", chunk_size=1024, chunk_overlap=0)

    for chunk in splitter.split_text(data):
      source_chunks.append(Document(page_content=chunk, metadata={}))

    # Индексируем документ по частям и заносим в векторную базу данных
    search_index = Chroma.from_documents(source_chunks, OpenAIEmbeddings(), )
    # Посчитаем токены
    count_token = num_tokens_from_string(' '.join([x.page_content for x in source_chunks]), "cl100k_base")
    billing = 0.0004*(count_token/1000) 
    return search_index, count_token, billing

    
  def answer_index(self, system, topic, search_index, temp = 1): 
    # Поиск документов по схожести с вопросом 
    docs = search_index.similarity_search(topic, k=5)
    message_content = re.sub(r'\n{2}', ' ', '\n '.join([f'\nОтрывок документа №{i+1}\n=====================' + doc.page_content + '\\n' for i, doc in enumerate(docs)]))
   
    messages = [
      {"role": "system", "content": system + f"{message_content}"},
      {"role": "user", "content": topic}
      ]

    completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages,
    temperature=temp
    )

    return completion.choices[0].message.content, 0.002*(completion["usage"]["total_tokens"]/1000)
