import streamlit as st
import numpy as np
import pandas as pd
from GPTLib import GPT

gpt = GPT()

def preTrain(token, doc):
  try:
    if token is None:
      st.write('Для обучения нужен ключ от ChatGPT')
    else:
      gpt.update_token(token)
      ind, count_token, billing = gpt.load_search_indexes(doc)

      st.write(f"Количество токенов в документе: {count_token}") 
      st.write(f"СТОИМОСТЬ ОБУЧЕНИЯ МОДЕЛИ: {billing}$") 
      return ind

  except Exception as e:
    st.write(f"ОШИБКА: {e}")
    return None

def answer(prompt, query, indexes):
  if indexes is None:
    st.write('Модель необходимо обучить!')
  else:
    try:
      content, billing= gpt.answer_index(
        prompt,
        query,
        indexes
      )
      st.write(f"ОТВЕТ: {content}") 
      st.write(f"СТОИМОСТЬ ЗАПРОСА: {billing}$") 
    except Exception as e:
      st.write(f"ОШИБКА: {e}")


def main(): 

  if "learn" not in st.session_state:
    st.session_state.learn = None


  def answer_request():
    if len(st.session_state['prompt'])>16 and len(st.session_state['query'])>10:      
      answer(st.session_state['prompt'], st.session_state['query'], st.session_state.learn)

    
  
  def input_calback():
    if len(st.session_state['token'])>16 and len(st.session_state['doc'])>16:
      st.session_state.learn = preTrain(st.session_state['token'], st.session_state['doc'])

  st.header("Дообучение ChatGPT на документе Google Doc")
  st.subheader("Откройте левую боковую панель и задайте параметры")

  st.sidebar.text_input('Ключ доступа', '', key='token', on_change=input_calback)
  st.sidebar.text_input('Данные для обучения (ссылка на гугл документ)', 'https://docs.google.com/document/d/1MuQ02a3Kz6ysDN43SM5YrURS2Mg2trLNrM_TyIElZLQ/edit?usp=sharing', key='doc', on_change=input_calback)
  
  st.sidebar.text_area('Инструкции для модели', 
      """Инструкция по оценке диалога менеджера по продажам с клиентом

       Перед тобой диалог менеджера по продажам с клиентом
       Тебе надо проверить несколько критериев и заполнить отчёт
       Не пиши общее сообщение, только отчёт по форме, форма отчёта будет в конце

       Что надо проверить
       1. Говорил ли клиент о потребности в трудоустройстве
       2. Говорил ли клиент о потребности создать AI проект
       3. Говорил ли клиент возражение о стоимости обучения
       4. Говорил ли клиент возражение о времени на обучение

       Форма отчёта
       1. Потребность о трудоустройстве - ДА или НЕТ
       2. Потребность о создании AI проекте - ДА или НЕТ
       3. Возражение о стоимости - ДА или НЕТ
       4. Возражение о времени обучения - ДА или НЕТ

       Заполни отчёт и пришли в качестве ответа, коротко и ёмко """, key='prompt', disabled = st.session_state.learn == None)

  st.sidebar.text_area('Запрос пользователя', 'Пришли отчет',  key='query', disabled = st.session_state.learn == None)
  st.sidebar.button('Отправить запрос', key='request', on_click=answer_request, disabled = st.session_state.learn == None)
  

   
if __name__ == "__main__":
    main()
