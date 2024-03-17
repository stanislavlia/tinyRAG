from openai import OpenAI
import os
from dotenv import load_dotenv, find_dotenv


class OpenAI_LLMGenerator():
    def __init__(self, openai_client, system_promt, max_token=1500, modelname="gpt-3.5-turbo"):
        
        self.system_promt = system_promt
        self.client = openai_client
        self.modelname = modelname
        self.max_token = max_token
        
    
    #NOW WORKS FOR SINGLE query
    def _create_userpromt_from_chunks(self, query, relevant_chunks):
        
        joint_relevant_chunks = "\n<EOD>\n".join(relevant_chunks["documents"][0])
        
        
        USER_PROMT_TEMPLATE = f""" You need to answer this question using provided information: {query}
                            
                            Here's the related chunks of documents. Each chunks ends with special token <EOD>:
                            
                            {joint_relevant_chunks}
                           
                      """
        
        return USER_PROMT_TEMPLATE
    
    def _get_chunkssources_info(self, relevant_chunks):
        
        sources = "\n".join([ str(d) for d in relevant_chunks["metadatas"][0]])
        
        return sources
    
    
        
    def generate_response(self, query_text, relevant_chunks):
        
        messages = [
        {
            "role": "system",
            "content": self.system_promt,
        },
            
        {
            "role": "user",
            "content": self._create_userpromt_from_chunks(query=query_text,
                                                          relevant_chunks=relevant_chunks)
        }   ]

        response = self.client.chat.completions.create(
            model=self.modelname,
            messages=messages,
           )
        
        
        content = response.choices[0].message.content
        content = content
        
        content += "\n\n\n [INFO] Related chunks used for generation:\n\n" + self._get_chunkssources_info(relevant_chunks)
        
        return content

        
        