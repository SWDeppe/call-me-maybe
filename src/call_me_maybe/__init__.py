from llm_sdk import Small_LLM_Model
# import torch
from pathlib import Path
from typing import Any
import json
import pydantic
from .Function import Function

# def load_vocabulary(path: str) -> list[tuple[str, int]]:


def select_fonction_0(prompt: str, functions: list[str]) -> str:
    # V0 -> function is list[str]
    print("for user prompt:", prompt)
    sdk = Small_LLM_Model()
    # functions.append("None")
    prompt = f"you will have to find a function to use bettween {functions} with the prompt:\n" + prompt + "\nfirst tokens fonction choice:\n"
    tensor = sdk.encode(prompt)
    result = sdk.get_logits_from_input_ids(tensor.squeeze(0).tolist())
    functions_tokens_indexes: dict[str, list[int]] = {}
    for func in functions:
        functions_tokens_indexes[func] = sdk.encode(func).squeeze(0).tolist()

    for name, i in functions_tokens_indexes.items():
        print(f"for {name} with first token {sdk.decode([i[1]])}:", result[i[1]])

def select_fonction(prompt: str, functions: list[Function]) -> int:
    # define a new prompt to give the LLM more context
    new_prompt = ("We are in a constrained decoding context !\nA coder wants you to get the right function to use in a function set.\n" +
            "The function set in question is:")
    for func in functions:
        new_prompt += "\n\t" + func.__repr__()   
    new_prompt += "\nCoder ask to you: " + prompt + "you:"

    sdk = Small_LLM_Model()

    # functions -> tokens [prompt, tokenId, scoreForNextGoodToken]
    functions_tokens: list[list[int]] = [ sdk.encode(func.name).squeeze(0).tolist() for func in functions ]
    res_branches: list[list[tuple[str, int, float]]] = [ [] for _ in range(len(functions))] # TODO: Add a general score to each branch to know when to stop or to continue !
    i = 0
    while (True):
        count = 0
        for y in range(len(functions)):
            if i + 1 >= len(functions_tokens[y]):
                count += 1
                continue
            
            logits = sdk.get_logits_from_input_ids(sdk.encode(new_prompt if i == 0 else res_branches[y][i - 1][0] ).squeeze(0).tolist())

            res_branches[y].append(
                (
                    new_prompt if i == 0 else res_branches[y][i - 1][0] + sdk.decode([functions_tokens[y][i]]),  # prompt at this stage
                    functions_tokens[y][i],
                    logits[functions_tokens[y][i]]
                )
            )
        print(res_branches)
        if count == len(functions):
            break
        i += 1
    for i, branch in enumerate(res_branches):
        print(functions[i].name + ": ")
        print(*[ tokens[2] for tokens in branch])



    raise NotImplementedError


function_type = dict[str, str | dict[str, type]]


def parse(func_definition_path: str, prompts_path: str) -> tuple[list[function_type], list[dict[str, str]]]:
    with open(func_definition_path) as f:
        json_content = f.read()
    functions: list[dict[str, Any]] = json.decoder.JSONDecoder().decode(json_content)

    with open(prompts_path) as f:
        json_content = f.read()
    prompts: list[dict[str, str]] = json.decoder.JSONDecoder().decode(json_content)


    # assert()
    
    return (functions, prompts)
    # pydantic.typing

def main() -> None:
    # print(select_fonction("What is your name ?", ["fn_add", "fn_substract", "fn_greet"]))
    here = Path().absolute()
    functions, prompts = parse(str(here) + "/data/input/functions_definition.json",
          str(here) + "/data/input/function_calling_tests.json")
    my_functions: list[Function] = []
    for func in functions:
        my_functions.append(Function(func["name"], func["description"]))
    select_fonction(prompts[0]["prompt"], my_functions)

    