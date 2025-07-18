# 接續2_getvector_query.py
# 呼叫local LLM
import requests
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
import os
import pprint

openai_ef_chroma = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.environ.get('OPENAI_API_KEY'),
                model_name="text-embedding-3-small")
chromadb_client = chromadb.PersistentClient(path="../med_vectordata2")
chroma_collection = chromadb_client.get_collection(name="advise_template", embedding_function=openai_ef_chroma)

system_message = '''
你是一位專門為醫療教育者提供量身定制建議和建議的健康教育助手，根據患者的醫療檢驗報告和病歷來提供這些建議。你的角色是使用檢索增強生成（RAG）技術來提供可能的患者問題和相關的健康教育建議。這些建議應該是清晰、專業和支持性的，並考慮到患者的具體情況和需求。

在製作回應時，請考慮以下要素：
1. 患者醫療狀況的背景。
2. 具體的醫療檢驗結果及其指示。
3. 針對該情況的最佳健康教育實踐。

確保提供準確且基於證據的建議，讓醫療教育者可以直接使用這些建議來指導患者。如果你不確定答案，你可以說「我不知道」或「我不確定」，並推薦用戶前往XXXX網站獲取更多信息。
輸出一定要用繁體中文輸出。
'''
user_request = '''以下是病人的病歷與檢驗報告{report}，根據提供的訊息{advise}，請提供:
1. 衛教時可能詢問病人的問題 2.相關的衛教建議。'''

# 模型可參考ollama list裡面的模型
def get_ollama_chat_response(query, report, model='yabi/breeze-7b-instruct-v1_0_q6_k:latest'):
    try:
        results = chroma_collection.query(query_texts=[query], n_results=5)
        retrieved_documents = results['documents'][0]
        advise = "\n\n".join(retrieved_documents)
        
        url = "http://localhost:11434/api/chat"
        data = {
            "model": model,
            "messages": [
                {'role': 'system', 'content': system_message},
                {'role': 'user', 'content': f"以下是病人的病歷與檢驗報告{report}，根據提供的訊息{advise}，請提供:1.衛教時可能詢問病人的問題 2.相關的衛教建議。"},
            ],
            "stream": False,
        }
        
        response = requests.post(url, json=data)
        response.raise_for_status()  # 確保請求成功，否則引發 HTTPError

        # 處理回應
        completion = response.json()
        response_content = completion['message']['content']
        prompt_tokens = completion['prompt_eval_count']
        completion_tokens = completion['eval_count'] 
        
        
        # 將奈秒轉換微秒
        total_duration_sec = completion['total_duration'] / 1e9
        load_duration_sec = completion['load_duration'] / 1e9
        prompt_eval_duration_sec = completion['prompt_eval_duration'] / 1e9
        gen_eval_duration_sec = completion['eval_duration'] / 1e9

        # 以下這些可以傳入資料庫當作調整各種 prompt 及模型的參考
        output_data = {
            "model_name": model,
            "sys_prompt": system_message,
            "user_promt": user_request,
            "Response": response_content,
            "Number of prompt tokens": prompt_tokens,
            "Number of completion tokens": completion_tokens,
            "Total duration (sec)": total_duration_sec,
            "Load duration (sec)": load_duration_sec,
            "Prompt duration (sec)": prompt_eval_duration_sec,
            "Gen duration (sec)": gen_eval_duration_sec,
            "vectordata": "med_vectordata2"
        }

        # 使用 pprint 來漂亮列印資料
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(output_data)

        return response_content
    
    except requests.exceptions.RequestException as e:
        return f'網路請求錯誤: {e}'
    except KeyError as e:
        return '回應中沒有預期的數據結構: {e}'
    except Exception as e:
        return f'未知錯誤: {e}'

#testing
query="糖尿病前期"
with open("../patient_c.txt",'r') as f:
    p_testing_report = f.read()

p_c = get_ollama_chat_response(query, p_testing_report)
print(p_c)
