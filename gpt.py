import os

import openai


class OpenAIService:
    def __init__(self, model_name="gpt-3.5-turbo-0301"""):
        openai.api_key = os.environ["OPENAI_API_KEY"]
        self.model_name = model_name
        # self.model_name = "gpt-3.5-turbo"
        # self.model_name = "gpt-3.5-turbo-0301" # my paid yahoo account
        # self.model_name = "gpt-4-0314", # only nyc-hackathon-5 org https://platform.openai.com/account/rate-limits
        # self.model_name = "gpt-4", # only nyc-hackathon-5 org https://platform.openai.com/account/rate-limits

    def list_models(self):
        return openai.Model.list()

    def prompt(self, user_prompt, system_prompt=None, assistant_prompt=None):
        return openai.ChatCompletion.create(
            model=self.model_name,
            # messages=[
            #     {"role": "user", "content": user_prompt}
            #     # {"role": "system", "content": system_prompt} if system_prompt else None,
            #     # {"role": "assistant", "content": assistant_prompt} if assistant_prompt else None,
            #     ],
            messages= [{"role": "user", "content": user_prompt}] if not system_prompt else [
                {"role": "user", "content": user_prompt},
                {"role": "system", "content": system_prompt},
                ],
            timeout=60,
        )
