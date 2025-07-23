from typing import Optional

from pydantic import BaseModel


class MetricsData(BaseModel):
    """Base class for all metrics data.

    Parameters:
        processor: Name of the processor generating the metrics.
        model: Optional model name associated with the metrics.
    """

    processor: str
    model: Optional[str] = None
