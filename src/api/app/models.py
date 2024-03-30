"""Fast API models"""

from pydantic import BaseModel

class ConfigItem(BaseModel):
    """Config Item of GetCResponse model"""
    name: str = None
    descriptions: str = None
    attributes_name: str
    options: list | str

class GetCResponse(BaseModel):
    """Get Config Response Model"""
    items: list[ConfigItem] = []
