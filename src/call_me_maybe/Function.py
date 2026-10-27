class Function:
    # TODO: add parameters/return and type validation by doing this a pydantic class !
        # !!!                              TYPE VALIDATION EXAMPLE FOUND ON STACK OVERFLOW                                           !!! #
        # from pydantic import BaseModel, field_validator, TypeAdapter, ConfigDict
        # from typing import Any


        # class State(BaseModel):
        #     model_config  = ConfigDict(validate_assignment=True)
        #     value_type: type
        #     value: Any

        #     @field_validator("value", mode="before")
        #     @classmethod
        #     def validate_value(cls, value, info):
        #         adapter = TypeAdapter(info.data["value_type"])
        #         return adapter.validate_python(value)


        # state1 = State(value=True, value_type=bool)
        # state2 = State(value=7, value_type=int)

        # state1.value = 7
    name: str
    description: str
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def __repr__(self) -> str:
        """ This will be feed to the prompte keep it short ! """
        return f"the function {self.name} do: {self.description}"