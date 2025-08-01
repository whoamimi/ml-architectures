"""
Qwen 2.5-1.5b Instruct mode
https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct

"""

from transformers import pipeline

# model_name = "Qwen/Qwen2.5-1.5B-Instruct"
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

# pipe = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct")

#################### PROMPTS
BASE_AGENT_PROMPT = """You are {agent_name}. Given the chat history messages with a user and their recent response, your task is to respond to the user's message. Your response must align with the flow of dialogue and reflect on your identity profile given below:
Name: {agent_name}
Unique Identity number: 120127
Owner: {user_name}
Owner's identity profile:
{user_profile}

"""

PLAY_DATING_INTERVIEWER = """You are {agent_name}, a dating interviewer conducting sessions to assist your friend {user_name} in finding the most compatible romantic partner. In this role, your task is to ask insightful, thoughtful questions that reveal their intentions and characteristics that would make them suitable candidates as {user_name}'s romantic partner. Your responses must ensure the conversation remains focused at answering the given list of interviewing questions and ensure that all questions are answered by the user.

# Here are the questions to ask:
{questions}

# Below are background information for the chat:
## {user_name}'s profile:
{user_profile}

## {user_name}'s preferences in an ideal partner:
{user_preference}

# Example of chat dialogue:
{example_dialogue}

Your response should ensure the conversation flows naturally, and transitions between questions feel seamless. Your response should not exceed 2 sentences.
"""
#################### MOCK PROMPTS

class Mock:
    MAIN_USER_NAME = "Mimi"
    MAIN_USER_PROFILE = """
        First name: Mimi
        Last name: Phan
        Location: Sydney
        Occupation: University Student
        Occupation Notes: Studies at Univeristy of NSW, Majors mathematics
        Favourites: Likes color pink, enjoys programming, likes to catch the bus
        Dislikes: Curry
        Personalty Trait according to Myer Briggs: INTJ
        Relation: Has a pet dog.
        Summary facts: Impatient
        """
    AGENT_NAME = "Naomi"
    MAIN_USER_PREFERENCE = """
    - intelligence
    - deep philosophical insights
    - good sense of humor
    - competitive
    - cheeky and romantic
    """
    INTERVIEW_EXAMPLE = """
    User: [START]
    Assistant: Hey, thanks for coming to chat with me today. My name is Naomi. What is your name? :)
    User: My name is Alex.
    Assistant: Nice to meet you, Alex! What is your Instagram account ID?
    User: It's @alex_awesome.
    Assistant: Cool, @alex_awesome! What's your favorite color, Alex?
    User: My favorite color is blue.
    Assistant: Great choice! Blue is such a calming color. What is your intent in dating Mimi?
    User: I want to get to know Mimi better and see if we have a genuine connection.
    Assistant: That sounds wonderful, Alex. That concludes our interview. Thanks for sharing. And thank you for coming to this interview. Have a great day!
    User: [END]
    """
    INTERVIEW_QUESTIONS = """
    1. Exchange greeting: Hey, thanks for coming to chat with me today. My name is Naomi. What is your name? :)
    2. User's name: What is your name?
    3. User's social media account name: What is your instagram account id?
    4. User's favourite color: What is your favourite color?
    5. User's dating intention: What is your intent in dating Mimi?
    6. End chat: Thanks for coming to this interview.
    """

    @staticmethod
    def base_prompt():
        return BASE_AGENT_PROMPT.format(
            user_name=Mock.MAIN_USER_NAME,
            user_profile=Mock.MAIN_USER_PROFILE,
            agent_name=Mock.AGENT_NAME
        )

    @staticmethod
    def play_dating_prompt():
        return PLAY_DATING_INTERVIEWER.format(
            agent_name=Mock.AGENT_NAME,
            user_name=Mock.MAIN_USER_NAME,
            user_profile=Mock.MAIN_USER_PROFILE,
            questions=Mock.INTERVIEW_QUESTIONS,
            example_dialogue=Mock.INTERVIEW_EXAMPLE,
            user_preference=Mock.MAIN_USER_PREFERENCE
        )

#################### MINI HELPER PROMPTS
NARRATE_PROMPT = """You are an assistant bot. Given the chat history messages, your task is to summarise the user's messages"""

QUEST_PROMPT = """You are an assistant bot. Given the chat history messages, your task is to detect the type of question is asking by returning one of the following possible quetion types. If the input does not suit any listed question types below, return 'None', otherwise, return the question type that best describes the input's question.

The question types are listed below:
1. User is asking questions related to Mimi.
2. User is asking questions non-related to people.
3. User is asking questions related to Naomi.

# Example
### User's input
What does Mimi like?

### Your output
User is asking questions related to Mimi.
"""

HELPER_PROMPT = {
    "narrate": NARRATE_PROMPT,
    "quest": QUEST_PROMPT,

}

#################### MODEL CONFIGS

gen_config = dict(
    max_new_tokens=100,
    do_sample=False,
    use_cache=False,
    temperature=0.0,

)

class Agent:
    """Calls the pipeline ie: Contacts HuggingFace pipeline"""

    def __init__(self, model_card: str = model_name):
        self.pipe = pipeline("text-generation", model=model_card, device='mps')
        self.config = gen_config

    def __call__(self, messages: list, **kwargs: dict):
        if kwargs:
            self.config.update(kwargs)

        try:
            response = self.pipe(messages, **self.config)

            if not isinstance(response, list):
                raise TypeError(f'Expected list form but received: {type(response)}. Output: {response}')
            if response[0].get('generated_text', None) is None:
                raise ValueError(f"Expected dict returned with key 'generated_text' but received: {response[0]}")

            output = response[0]['generated_text'][-1]

            if output.get('role', None) != "assistant":
                raise ValueError(f"Expected role as assistant but received: {output}")

            return output['content']

        except Exception as e:
            return f"Error returned {e}"

class Chatterbox:
    """Stores / Appends Session chat history and features"""

    def __init__(self):
        self.messages = []
        self._chat_speaker = "user"

    def add_message(self, inputs: str):
        if len(self.messages) == 0:
            role = "system"
        else:
            role = self._chat_speaker
            self._chat_speaker = "user" if self._chat_speaker == "assistant" else "assistant"

        self.messages.append({
            "role": role,
            "content": inputs
        })

class Controller:
    """Controls the Chat session"""

    @staticmethod
    def chat(chat: Chatterbox, agent: Agent):
        while True:
            user_input = input("Enter message: ")
            chat.add_message(user_input)
            if user_input.startswith('\end'):
                return
            agent_response = agent.respond(chat.messages)
            chat.add_message(agent_response)
            print(f"Agent: {agent_response}")


CHAT_CONFIG = {
    "start_chat": "[START]",
    "end_chat": "[END]"
}
class ChatWindow:
    def __init__(self):
        self.agent = Agent()

        self.base_chat = Chatterbox()
        self.base_chat.add_message(Mock.base_prompt())

        self.roleplayer = Chatterbox()
        self.roleplayer.add_message(Mock.play_dating_prompt())
        self.roleplayer.add_message(CHAT_CONFIG['start_chat'])

    def start_step(self):
        response = self.agent(self.roleplayer.messages, use_cache=False)
        if isinstance(response, str) and not response.startswith("Error returned"):
            self.roleplayer.add_message(response)
        print(self.roleplayer.messages)

    def step(self, user_message: str):
        for attr_name in ['base_chat', 'roleplayer']:
            obj = getattr(self, attr_name)
            obj.add_message(user_message)
            response = self.agent(obj.messages)
            if isinstance(response, str) and not response.startswith("Error returned"):
                obj.add_message(response)
            else:
                return response

# agent = Agent()
# chat = Chatterbox()
# sys_prompt = Mock.base_prompt()
# chat.add_message(sys_prompt)
# Controller.chat(chat=chat, agent=agent)

inbox = ChatWindow()

