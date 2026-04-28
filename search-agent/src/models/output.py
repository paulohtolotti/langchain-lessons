from typing import List
from pydantic import BaseModel, Field

class Source(BaseModel):
    """Scheme for a source used by the agent"""
    url:str = Field(description="Source's URL")


class AgentResponse(BaseModel):
    """Schema for the agent response"""
    response:str = Field(description="Agent's answer")
    sources:List[Source] = Field(default_factory=list, description="URL of sources to generate the answer")
