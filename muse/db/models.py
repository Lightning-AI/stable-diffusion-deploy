from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel


class RequestMonitor(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    prompt: str
    request_count: int
    model_server_process_time: float
    load_balancer_process_time: float
    created_at: datetime = datetime.now()