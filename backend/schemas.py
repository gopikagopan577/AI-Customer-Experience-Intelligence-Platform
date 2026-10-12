
from pydantic import BaseModel, Field


class SentimentRequest(BaseModel):
    review_text: str = Field(
        min_length=1,
        max_length=10000,
        description="Customer review text"
    )


class CustomerRequest(BaseModel):
    age: int = Field(ge=0, le=120)
    location: str = Field(min_length=1, max_length=100)
    total_orders: float = Field(ge=0)
    avg_order_value: float = Field(ge=0)
    days_since_last_order: float = Field(ge=0)
    support_tickets: float = Field(ge=0)
    customer_satisfaction: float = Field(ge=1.0, le=5.0)


class SegmentationRequest(BaseModel):
    total_orders: float = Field(ge=0)
    avg_order_value: float = Field(ge=0)
    days_since_last_order: float = Field(ge=0)
    support_tickets: float = Field(ge=0)
    customer_satisfaction: float = Field(ge=1.0, le=5.0)
