from langchain_community.chat_models.moonshot import MoonshotChat
from langchain.schema import HumanMessage, SystemMessage
import init

# 这里没有直接用init中的默认配置
# 可以直接这样再次定义，不需要再次导入key之类的东西了
# 虽然代码中没有直接体现init包，但是不要删，这部初始化调用到了init中设定的环境变量
chat = init.chat
messages = [
    SystemMessage(content="你是一个很棒的智能助手"),
    HumanMessage(content="请给我的花店起个名,多输出几个结果，直接输出名字，不要输出多余的语句")
]
response = chat(messages)
print(response.content)
