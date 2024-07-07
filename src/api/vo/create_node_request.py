from pydantic import BaseModel, constr, field_validator
from consts import node_mapping


class CreateNodeRequest(BaseModel):
    node_type: str
    node_name: str
    node_description: str
    set_relationship: bool = False
    current_node_type: str | None = None
    current_node_name: str | None = None

    # node_type: constr()
    #
    # @field_validator("node_type")
    # def check_acceptable_values(cls, v):
    #     acceptable_node_values = node_mapping.keys()
    #     if v not in acceptable_node_values:
    #         raise ValueError(f'{v} is not an acceptable value. Must be one of {acceptable_node_values}')
    #     return v
