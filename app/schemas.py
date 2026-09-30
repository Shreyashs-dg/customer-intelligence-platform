from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    Recency: int = Field(..., ge=0, description="Days since the customer's last purchase")
    Frequency: int = Field(..., ge=1, description="Number of distinct orders placed")
    Monetary: float = Field(..., ge=0, description="Total amount spent across all orders")
    Tenure: int = Field(..., ge=0, description="Days between first and last purchase")
    AOV: float = Field(..., ge=0, description="Average order value (Monetary / Frequency)")
    Purchase_Variability: float = Field(..., ge=0, description="Std deviation of days between consecutive orders")
    Product_Diversity: int = Field(..., ge=1, description="Number of distinct products purchased")

    class Config:
        json_schema_extra = {
            "example": {
                "Recency": 46,
                "Frequency": 18,
                "Monetary": 10198.64,
                "Tenure": 591,
                "AOV": 545.27,
                "Purchase_Variability": 50.07,
                "Product_Diversity": 209,
            }
        }


class SegmentResponse(BaseModel):
    cluster: int
    segment_name: str


class ChurnResponse(BaseModel):
    cluster: int
    segment_name: str
    churn_probability: float
    churned: bool


class ForecastRequest(BaseModel):
    year: int = Field(..., ge=2009, description="Target year to forecast")
    month: int = Field(..., ge=1, le=12, description="Target month to forecast (1-12)")

    class Config:
        json_schema_extra = {"example": {"year": 2012, "month": 9}}


class ForecastResponse(BaseModel):
    target_month: str
    predicted_revenue: float
    method: str

