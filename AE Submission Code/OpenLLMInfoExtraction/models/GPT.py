from openai import OpenAI
from openai import AzureOpenAI
import openai
import tiktoken
import time

from .Model import Model


class GPTAzureFT(Model):
    def __init__(self, config, ft_name=''):
        super().__init__(config)
        print('GPTAzureFT is in use.\n')
        self.set_API_key()
        
        # Add your own deployment name
        if self.name == 'gpt-4':
            self.deployment_name = f'gpt-4-turbo-0409'
        elif self.name == 'gpt-3.5-turbo':
            self.deployment_name = f'gpt-35-turbo'
        else:
            raise NotImplementedError()
        
        print(f'self.deployment_name = {self.deployment_name}')

    def set_API_key(self):
        self.client = AzureOpenAI(
            api_key="<YOUR_API_KEY>",  
            api_version="<YOUR_API_VERSION>",
            azure_endpoint = "<YOUR_ENDPOINT>"
        )

    def query(self, msg, try_num=0, icl_num=0):
        if try_num >= 3:
            return 'RateLimitError'
        
        try:
            return self.__do_query(msg, icl_num)

        except openai.BadRequestError:
            return 'BadRequestError'

        except openai.RateLimitError:
            time.sleep(10)
            return self.query(msg, try_num+1, icl_num)

    def __do_query(self, msg, icl_num=0):

        completion = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=[
                {"role": "user", "content": msg}
            ],
            temperature=self.temperature
        )
        response = completion.choices[0].message.content

        return response


