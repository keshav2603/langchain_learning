from pydantic import BaseModel,Field
from typing import Optional
class Student(BaseModel):
    name:str="paras"
    age:Optional[int]=None
    cgpa:float=Field(gt=0,lt=10,default=1,description="value representiong cgpa of student")
new_student={"age":"10"}

stu=Student(**new_student)

print(stu)