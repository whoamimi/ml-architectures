
# Embeddings from Instruct Model SMOLLM
# Script to setup my test on 17th of April, 2025. Further details should be somewhere in Notion.
# Conclusion
#
# - Fine-tuned from base model `llama3.1` instruct model
# - Can be used as a chat model, however more useful in question-answer tasks rather than conversational agents like DialoGPT
# - If I want to trial methods using the sentence embeddings from the model itself, with Transformers from HF you cannot directly encode and retrieve the hidden states (will need to preform some list slicing)
# - This model is probably best for trialing Hooking / Adapters on instruct based models but not methods related to handling residuals, more detailed architect framework
# - This model is NOT useful for steering towards target vs rejected responses


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

# Loads the model directly
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M-Instruct")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M-Instruct")

def _process_v1(message):
    """ decodes tokens into jibberish. Problem with tokenizer.apply_chat_template + model.forward pass. Do not use. """

    test_token = tokenizer.apply_chat_template(message, padding='max_length', truncation=True, tokenize=True, max_length=50, return_assistant_tokens_mask=True, return_dict=True, return_tensors="pt", add_generation_prompt=False)

    with torch.no_grad():
        output = model(**test_token, max_new_tokens=0)

    print('Logits shape: ', output.logits.shape)

    sequence =output.logits.squeeze(0).softmax(-1).argmax(-1)
    print('Output:', sequence)

    return output


def _process_v2(message):
    test_token = tokenizer(message, return_tensors="pt", return_token_type_ids=True, return_attention_mask=True, return_special_tokens_mask=True, add_special_tokens=True, padding_side="left", return_offsets_mapping=True)
    with torch.no_grad():
        output = model(**test_token)

    print(tokenizer.decode(output.logits.squeeze(0).argmax(-1), skip_special_tokens=False, clean_up_tokenization_spaces=True))

    return output

def prepare_encode(prompt, target, reject):
    """ Usage example:
        target, reject = prepare_encode(prompt="Hey, how are you?", target="I am okay...", reject="I am good. And yourself
    """
    msg = [
        {
            "role": "system",
            "content": "You are Naomi. Your task is to be Mimi's friend"
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    target_msg = msg + [{"role": "assistant", "content": target}]
    reject_msg = msg + [{"role": "assistant", "content": reject}]

    target_encoded = tokenizer.apply_chat_template(target_msg, tokenize=False)
    reject_encoded = tokenizer.apply_chat_template(reject_msg, tokenize=False)

    return target_encoded,reject_encoded

def encode(target, reject):
    # NOTE: Encoder function has been removed as I found a better approach but needs more validation or comparison tests.
    pass

