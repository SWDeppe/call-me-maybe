from llm_sdk import Small_LLM_Model
# import torch
from pathlib import Path
from typing import Any
import json
import pydantic
from .Function import Function

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

    