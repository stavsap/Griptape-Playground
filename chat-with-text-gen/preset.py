from griptape.structures import Agent
from griptape.drivers import TextGenPromptDriver
from griptape.tokenizers import TextGenTokenizer
from transformers import PreTrainedTokenizerFast

fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json")

prompt_driver = TextGenPromptDriver(
    preset="griptape",
    tokenizer=TextGenTokenizer(max_tokens=300, tokenizer=fast_tokenizer)
)

agent = Agent(
    prompt_driver=prompt_driver
)

Chat(agent).start()
