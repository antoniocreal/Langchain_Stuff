from typing import List, Dict, Any

from langchain_core.output_parsers import PydanticOutputParser # Pydantic helps with validation and data management. It allows to define schemas and validate inputs agains those schemas
from pydantic import BaseModel, Field


class Summary(BaseModel):
    summary: str = Field(description="summary")
    facts: List[str] = Field(description="interesting facts about them")

    def to_dict(self) -> Dict[str, Any]:
        return {"summary": self.summary, "facts": self.facts}


summary_parser = PydanticOutputParser(pydantic_object=Summary)