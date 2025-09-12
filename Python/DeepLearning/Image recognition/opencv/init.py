import os
import configparser

from langchain_community.llms.moonshot import Moonshot
from langchain_community.chat_models.moonshot import MoonshotChat
from openai import OpenAI

config = configparser.ConfigParser()
config.read("../Langchain/setting.ini")
model = config["Moonshot"]["model"]
api_key = config["Moonshot"]["OPENAI_API_KEY"]
base_url = config["Moonshot"]["url"]
os.environ["MOONSHOT_API_KEY"] = api_key
os.environ['HUGGINGFACEHUB_API_TOKEN'] = config["Huggingface"]["HUGGINGFACEHUB_API_TOKEN"]


# 可变max长度
def Reset(max):
    Rellm = Moonshot(
        model=model,
        temperature=0.8,
        max_tokens=max,
    )
    return Rellm


# 默认llm配置。若使用自己配置，直接import本包即可重新定义
llm = Moonshot(
    model=model,
    temperature=0.8,
    max_tokens=20,
)

Nllm = Moonshot(
    model=model,
    temperature=0.8,
)

chat = MoonshotChat(
    model=model,
    temperature=0.8,
    max_tokens=20, )

# openAi 默认的接口，不使用langchain
client = OpenAI(
    api_key=api_key,
    base_url=base_url,
)
