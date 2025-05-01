from dotenv import load_dotenv
import os

from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

llm_config = {
  "config_list": [
    {
      "model": "gpt-4",
      "api_key": os.environ["OPENAI_API_KEY"]
    }
  ]
}

pm = UserProxyAgent(
  name="ProjectManageer",
  human_input_mode="NEVER",
  code_execution_config={"use_docker": False}
)

developer = AssistantAgent(
  name="Developer",
  system_message="You are JS developer. Write good structured and understandable code",
  llm_config=llm_config
)

reviewer = AssistantAgent(
  name="Reviwer",
  system_message="You are code reviewer. Check code style, errors, mistakes and make code better",
  llm_config=llm_config
)

group_chat = GroupChat(
  agents=[pm, developer, reviewer],
  messages=[],
  max_round=3
)

chat_manager = GroupChatManager(groupchat=group_chat, llm_config=llm_config)

pm.initiate_chat(
  chat_manager,
  message="Create function to list of fibonachi"
)



print('hello')