from typing import Literal

from pydantic import BaseModel, ConfigDict


class Function:
    name: str
    description: str
    return_value: str
    parameters: dict[str, Literal["number", "string"]]

    def __init__(self,
                 name: str,
                 description: str,
                 return_value: str,
                 parameters: dict[str, Literal["number", "string"]]):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.return_value = return_value

    def __repr__(self) -> str:
        """ This will be feed to the prompte keep it short ! """
        return f"{self.name}:\n\t\tdescription: {self.description}\n" \
            f"\t\treturn value: {self.return_value}\n" \
            f"\t\tparameters: {self.parameters}"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ValueDefinition(StrictModel):
    type: Literal["number", "string"]


class FunctionDefinition(StrictModel):
    name: str
    description: str
    parameters: dict[str, ValueDefinition]
    returns: ValueDefinition


class PromptDefinition(StrictModel):
    prompt: str
