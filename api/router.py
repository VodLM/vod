from fastapi import APIRouter
from api.models import CustomRequest, CustomResponse


router = APIRouter()


@router.post("/custom_request", summary="Custom request")
async def custom_request(request: CustomRequest) -> CustomResponse:
    """Perform the endpoint action."""
    content = CustomResponse(content=request.content)
    return content

