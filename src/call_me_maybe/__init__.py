from llm_sdk import Small_LLM_Model
# import torch
from pathlib import Path
from typing import Any
import json
import pydantic
from .Function import Function

# def load_vocabulary(path: str) -> list[tuple[str, int]]:


def select_function_v0(prompt: str, functions: list[str]) -> str:
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

def select_function_v1(prompt: str, functions: list[Function]) -> int:
    # define a new prompt to give the LLM more context
    new_prompt = ("We are in a constrained decoding context !\nA coder wants you to get the right function to use in a function set.\n" +
            "The function set in question is:")
    for func in functions:
        new_prompt += "\n\t" + func.__repr__()   
    new_prompt += "\nCoder ask to you: " + prompt + "\nyou:"

    sdk = Small_LLM_Model()

    # functions -> tokens [prompt, tokenId, scoreForNextGoodToken]
    functions_tokens: list[list[int]] = [ sdk.encode(func.name).squeeze(0).tolist() for func in functions ]
    res_branches: list[list[tuple[str, int, float]]] = [ [] for _ in range(len(functions))] # TODO: Add a general score to each branch to know when to stop or to continue !
    i = 0
    while (True):
        count = 0
        for y in range(len(functions)):
            if i >= len(functions_tokens[y]):
                count += 1
                continue
            
            logits = sdk.get_logits_from_input_ids(sdk.encode(new_prompt if i == 0 else res_branches[y][i - 1][0]).squeeze(0).tolist())

            res_branches[y].append(
                (
                    (new_prompt if i == 0 else res_branches[y][i - 1][0]) + sdk.decode([functions_tokens[y][i]]),  # prompt at this stage
                    functions_tokens[y][i],
                    logits[functions_tokens[y][i]]
                )
            )
        if count == len(functions):
            break
        i += 1

    scores: list[float] = []
    for i, branch in enumerate(res_branches):
        print(functions[i].name + ": ")
        print(*[ f"token {sdk.decode(tokens[1])}: {tokens[2]}" for tokens in branch])
        score = 0
        for y, token in enumerate(branch):
            score += token[2] / (y + 1)
        scores.append(score / len(branch))
        print("score:", scores[i])
        print("\n")
    raise NotImplementedError

# digit reaction
# 128
# 2 -> forget_digit = 3
# 2 and 20 to 29
# 2028
# 3 -> forget_digit = 3
# 2 and 20 to 29

import numpy as np

def select_function_v2(prompt: str, functions: list[Function]) -> int:
    """" use indexes and not the full function name that has scores that gets to varient the longer and increase too fast """
    # prepare a prompte
    # new_prompt = "Choose the function that best satisfies the coder's request.\nReturn only its index.\n"
    new_prompt = "Available functions:\n"
    new_prompt += "    0. No function matches\n"
    for i, func in enumerate(functions):
        new_prompt += f"    {i + 1}. {func.__repr__()}\n"
    new_prompt += f"Request: {prompt}"
    new_prompt += "Function Index: "

    # prepare the digits string
    max_str = str(len(functions))
    digits_count = len(max_str)
    max_first_digit: int = int(max_str[0])

    sdk = Small_LLM_Model()
    str_res = ""
    for i in range(len(max_str)):
        if i >= digits_count:
            break
        logits = sdk.get_logits_from_input_ids(sdk.encode(new_prompt).squeeze(0).tolist())
        possible_tokens: list[int] = [ sdk.encode(dig).squeeze(0).tolist()[0] for dig in "0123456789"]  # a digit token containes allways one single digit
        digits_scores = [ logits[id] for id in possible_tokens ]
        print("scores:", [ f"{functions[index].name}: {score}"  for index, score in enumerate(digits_scores[1:]) if index < len(functions)])
        print("None score:", digits_scores[0])
        choice = np.argmax(digits_scores)
        if (i == 0 and choice > max_first_digit):
            digits_count -= 1
        if (choice == 0):
            break
        new_prompt += str(choice)
        str_res += str(choice)

    return int(str_res)

def select_function(prompt: str, functions: list[Function]) -> int:
    """" use indexes and not the full function name that has scores that gets to varient the longer and increase too fast """
    # prepare a prompte
    # new_prompt = "Choose the function that best satisfies the coder's request.\nReturn only its index.\n"
    new_prompt = "Available functions:\n"
    new_prompt += "    0. No function matches\n"
    for i, func in enumerate(functions):
        new_prompt += f"    {i + 1}. {func.__repr__()}\n"
    new_prompt += f"Request: {prompt}"

    sdk = Small_LLM_Model()
    new_prompt += "\n\nFunction Name: "
    # test add the function name
    functions_token: list[list[int]] = [ sdk.encode(func.name).squeeze(0).tolist() for func in functions ]
    i = 0
    # max_len = len(max(functions_token))
    function_name = ""
    while True:
        logits = sdk.get_logits_from_input_ids(sdk.encode(new_prompt).squeeze(0).tolist())
        functions_tokens_scores = [ logits[tokens[i]] for tokens in functions_token if len(tokens) > i ]
        choice = np.argmax(functions_tokens_scores)
        function_name += sdk.decode([functions_token[choice][i]])
        new_prompt += sdk.decode([functions_token[choice][i]])
        print(f"choice {i}:", choice)
        # define the remaning possible functions to choose
        token = functions_token[choice][i]
        functions_token = [ tokens for tokens in functions_token if tokens[i] == token]
        if len(functions_token) == 1:
            function_name = sdk.decode(functions_token[0])
            break 
        i += 1

    print(f"For the prompt: `{prompt}` choosed function name: {function_name}")
    for i, func in enumerate(functions):
        if func.name == function_name:
            return i
    return -1

function_type = dict[str, str | dict[str, type]]


def parse(func_definition_path: str, prompts_path: str) -> tuple[list[function_type], list[dict[str, str]]]:
    with open(func_definition_path) as f:
        json_content = f.read()
    functions: list[dict[str, Any]] = json.decoder.JSONDecoder().decode(json_content)

    with open(prompts_path) as f:
        json_content = f.read()
    prompts: list[dict[str, str]] = json.decoder.JSONDecoder().decode(json_content)

    return (functions, prompts)


def convert_to_func(function: function_type) -> Function:
    parameters = {}
    for param_name, value in function["parameters"].items():
        parameters[param_name] = value["type"]
    return Function(function["name"], function["description"], function["returns"]["type"], parameters)

def main() -> None:
    # print(select_fonction("What is your name ?", ["fn_add", "fn_substract", "fn_greet"]))
    here = Path().absolute()
    functions, prompts = parse(str(here) + "/data/input/functions_definition.json",
          str(here) + "/data/input/function_calling_tests.json")
    my_functions: list[Function] = []
    for func in functions:
        my_functions.append(convert_to_func(func))
    for prompt in prompts:
        select_function(prompt["prompt"], my_functions)
        print("\n\n")

    