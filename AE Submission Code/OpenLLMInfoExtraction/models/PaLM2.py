import google.generativeai as palm
import google.ai.generativelanguage as gen_lang
import google

from .Model import Model


class PaLM2(Model):
    def __init__(self, config):
        super().__init__(config)
        api_keys = config["api_key_info"]["api_keys"]
        api_pos = int(config["api_key_info"]["api_key_use"])
        assert (0 <= api_pos <= len(api_keys)), "Please enter a valid API key to use"
        self.api_key = api_keys[api_pos]
        self.set_API_key()
        self.max_output_tokens = int(config["params"]["max_output_tokens"])
        if self.name == 'models/text-bison-001':
            self.name = self.name.replace('text-bison-001', 'text-bison@001')
        if self.name == 'models/chat-bison-001':
            self.name = self.name.replace('chat-bison-001', 'chat-bison@001')
        print(self.name)
        
    def set_API_key(self):
        palm.configure(api_key=self.api_key)

    def query(self, msg):
        if 'text' in self.name:
            completion = palm.generate_text(
                model=self.name,
                prompt=msg,
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
                safety_settings=[
                    {
                        "category": gen_lang.HarmCategory.HARM_CATEGORY_DEROGATORY,
                        "threshold": gen_lang.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
                    },
                    {
                        "category": gen_lang.HarmCategory.HARM_CATEGORY_TOXICITY,
                        "threshold": gen_lang.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
                    },
                    {
                        "category": gen_lang.HarmCategory.HARM_CATEGORY_VIOLENCE,
                        "threshold": gen_lang.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
                    },
                    {
                        "category": gen_lang.HarmCategory.HARM_CATEGORY_SEXUAL,
                        "threshold": gen_lang.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
                    },
                    {
                        "category": gen_lang.HarmCategory.HARM_CATEGORY_MEDICAL,
                        "threshold": gen_lang.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
                    },
                    {
                        "category": gen_lang.HarmCategory.HARM_CATEGORY_DANGEROUS,
                        "threshold": gen_lang.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
                    },
                ]
            )
            response = completion.result

        elif 'chat' in self.name:
            response = palm.chat(messages=msg, candidate_count=1).last
        return response