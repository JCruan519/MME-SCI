give_answer_after_final_answer_prompt_zh = "请在 '最终答案: ' 之后简洁地写出你给出的答案。"

give_answer_after_final_answer_prompt_visiononly_zh = "请解答图像中给定的问题，并且在 '最终答案: ' 之后简洁地写出你的答案。"

give_answer_after_final_answer_prompt_en = "Please write the final answer concisely after 'Final Answer: '."

give_answer_after_final_answer_prompt_fr = "Veuillez écrire la réponse finale de manière concise après 'Réponse finale: '."

give_answer_after_final_answer_prompt_es = "Por favor, escribe tu respuesta de manera concisa después de 'Respuesta final: '."

give_answer_after_final_answer_prompt_ja = "'最終答え: ' の後に、あなたが出した答えを簡潔に書いてください。"


sys_prompt_of_judger = "You are a strict and impartial judge. Based on the original question, the standard answer, and the AI assistant's response provided by the user, determine whether the AI assistant's response is correct. If there is any difference in meaning between the AI's response and the standard answer, reply with 'incorrect'. If the meanings are the same, reply with 'correct'. Important: Do not answer the original question, and do not provide reasoning or explanation. Only respond with 'correct' or 'incorrect'."

judge_prompt = '''## Original Question: {question}

## Standard Answer: {standard_answer}

## AI Assistant's Response: {ai_respond}

## NOTE: Do not answer the original question, and do not provide reasoning or explanation. Only respond with 'correct' or 'incorrect'.

## Your respond: 
'''