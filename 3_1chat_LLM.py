# 接續2_2getvector_query.py
# 呼叫Openai GPT4o LLM

from openai import OpenAI

# 要準備自己的system prompt, user_prompt
GPT_MODEL = "gpt-4o" # "gpt-4-turbo-2024-04-09"or "gpt-3.5-turbo-1106" or "gpt-4o"
system_message = '''
你是一位專門為醫療教育者提供量身定制建議和建議的健康教育助手，根據患者的醫療檢驗報告和病歷來提供這些建議。你的角色是使用檢索增強生成（RAG）技術來提供可能的患者問題和相關的健康教育建議。這些建議應該是清晰、專業和支持性的，並考慮到患者的具體情況和需求。

在製作回應時，請考慮以下要素：
1. 患者醫療狀況的背景。
2. 具體的醫療檢驗結果及其指示。
3. 針對該情況的最佳健康教育實踐。

確保提供準確且基於證據的建議，讓醫療教育者可以直接使用這些建議來指導患者。如果你不確定答案，你可以說「我不知道」或「我不確定」，並推薦用戶前往成大醫院網站獲取更多信息。
輸出一定要用繁體中文輸出。
'''

user_request = '''以下是病人的病歷與檢驗報告{report}，根據提供的訊息{advise}，請提供:
1. 衛教時可能詢問病人的問題 2.相關的衛教建議。'''

openai_ef_chroma = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.environ.get('OPENAI_API_KEY'),
                model_name="text-embedding-3-small")
chromadb_client = chromadb.PersistentClient(path="./med_vectordata2")
chroma_collection = chromadb_client.get_collection(name="advise_template", embedding_function=openai_ef_chroma)

# 這部分傳入文本要針對不同專案去設計，retrieved_documents是檢索出的資料供LLM參考，此處query是屬於哪種個案，report當作context直接放入prompt，advise是檢索出來的chunking資料, gpt4o設定seed的話可以得到比較一致的效果
def get_chat_response(query, report, seed: int = None):
    try:
        results = chroma_collection.query(query_texts=query , n_results=2)
        retrieved_documents = results['documents'][0]
        advise = "\n\n".join(retrieved_documents)
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"以下是病人的病歷與檢驗報告{report}，根據提供的訊息{advise}，請提供:1.衛教時可能詢問病人的問題 2.相關的衛教建議。"},
        ]


        client = OpenAI(
            # This is the default and can be omitted
            api_key=os.environ.get('OPENAI_API_KEY'),
        )

        completion = client.chat.completions.create(
            model=GPT_MODEL,
            messages=messages,
            seed=seed,
            max_tokens=2000,
            temperature=0.7,

        )

        response_content = completion.choices[0].message.content
        # print(f"response content: {response_content}")
        system_fingerprint = completion.system_fingerprint
        # print(f"system_fingerprint: {system_fingerprint}")
        prompt_tokens = completion.usage.prompt_tokens #response["usage"]["prompt_tokens"]
        completion_tokens = (
            completion.usage.total_tokens - prompt_tokens
        )
        
        #以下這些可以傳入資料庫當作調整各種prompt及模型的參考
        output_data = {
                "model_name":GPT_MODEL,
                "sys_prompt":system_message,
                "user_promt":user_request,
                "Response": response_content,
                "System Fingerprint": system_fingerprint,
                "Number of prompt tokens": prompt_tokens,
                "Number of completion tokens": completion_tokens,
            "vectordata": "med_vectordata2"
        }

        # 使用 pprint 來漂亮列印資料
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(output_data)

        return response_content
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

#testing
query="糖尿病前期"
with open("patient_c.txt",'r') as f:
    p_testing_report = f.read()

p_c = get_chat_response(query, p_testing_report)