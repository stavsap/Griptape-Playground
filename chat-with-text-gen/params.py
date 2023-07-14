from griptape.structures import Agent
from griptape.drivers import TextGenPromptDriver
from griptape.tokenizers import TextGenTokenizer
from transformers import PreTrainedTokenizerFast

params = {
        'max_new_tokens': 250,
        'do_sample': True,
        'temperature': 0.7,
        'top_p': 0.1,
        'typical_p': 1,
        'epsilon_cutoff': 0,  # In units of 1e-4
        'eta_cutoff': 0,  # In units of 1e-4
        'tfs': 1,
        'top_a': 0,
        'repetition_penalty': 1.18,
        'repetition_penalty_range': 0,
        'top_k': 40,
        'min_length': 0,
        'no_repeat_ngram_size': 0,
        'num_beams': 1,
        'penalty_alpha': 0,
        'length_penalty': 1,
        'early_stopping': False,
        'mirostat_mode': 0,
        'mirostat_tau': 5,
        'mirostat_eta': 0.1,
        'seed': 235245345,
        'add_bos_token': True,
        'truncation_length': 2048,
        'ban_eos_token': False,
        'skip_special_tokens': True,
        'stopping_strings': []
    }

fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json")

prompt_driver = TextGenPromptDriver(
    params=params,
    tokenizer=TextGenTokenizer(max_tokens=params['max_new_tokens'], tokenizer=fast_tokenizer)
)

agent = Agent(
    prompt_driver=prompt_driver
)

Chat(agent).start()
