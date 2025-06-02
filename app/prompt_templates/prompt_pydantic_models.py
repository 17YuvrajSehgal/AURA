from typing import List

from pydantic import BaseModel, Field


class ImportantSectionsOfReadme(BaseModel):
    sections_list: List[str] = Field(
        ...,
        description="All the important section, subsections that must be present in the readme file according to "
                    "conference guidelines",
    )


class LicenceFile(BaseModel):
    is_acceptable: bool = Field(description="Boolean value true if the license is acceptable, false otherwise")
    reasoning: str = Field(description="Reason why the license was accepted or rejected")
    acceptable_licence_list: List[str] = Field(
        description="List of all the acceptable licence types mentioned in conference if any.")


class Entities(BaseModel):
    names: List[str] = Field(
        ...,
        description="All the person, organization, or business entities that appear in the text",
    )


class ListOfFiles(BaseModel):
    names: List[str] = Field(
        ...,
        description="Full paths of the files.",
    )
